"""Optimised batched best-of-N Chatterbox voice conversion + SIDON + scoring.

Design goals (the LeVo lesson: the model was never the bottleneck, dispatch was):
  * ONE model load per process, never per item.
  * Generation batches (M sources x N candidates) in a single s3gen forward pass,
    with length-bucketed padding so the pad waste stays small.
  * SIDON runs BATCHED over candidates (the reference implementation loops one
    candidate at a time -> 32 separate torchscript calls per source).
  * All resampling happens on the GPU (torchaudio.functional.resample) instead of
    librosa/numpy on the CPU.
  * Speaker embedders (ECAPA / WavLM-tbr) are fed batches, not single clips.
  * MP3 encoding uses PyAV in a thread pool, not one ffmpeg subprocess per clip.

Nothing here mutates the upstream chatterbox package.
"""
import os, sys, time, json, math, io
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

S3_SR = 16000
OUT_SR = 24000
SIDON_SR = 48000


# --------------------------------------------------------------------------- #
#  Chatterbox VC
# --------------------------------------------------------------------------- #
def load_vc(device="cuda:0"):
    from chatterbox.vc import ChatterboxVC
    vc = ChatterboxVC.from_pretrained(device)
    vc.s3gen.eval()
    return vc


def set_target_from_wav(vc, wav, sr, peak_norm=None):
    """Set the target voice from an in-memory waveform.

    peak_norm=None keeps the incoming level (use this when the reference has
    already been loudness-normalised); peak_norm=0.97 reproduces the upstream
    demo behaviour, which throws that normalisation away.
    """
    w = np.asarray(wav, np.float32)
    if peak_norm is not None:
        p = np.abs(w).max()
        if p > 0:
            w = w * (peak_norm / p)
    t = torch.from_numpy(w).float()
    if sr != OUT_SR:
        t = torchaudio.functional.resample(t, sr, OUT_SR)
    t = t[: 10 * OUT_SR]                       # DEC_COND_LEN
    vc.ref_dict = vc.s3gen.embed_ref(t.numpy(), OUT_SR, device=vc.device)
    return vc.ref_dict


@torch.inference_mode()
def tokenize(vc, wav16_list, device):
    """S3-tokenize a list of 16k mono float arrays. Returns (tokens[B,T], lens[B])."""
    L = max(len(w) for w in wav16_list)
    x = torch.zeros(len(wav16_list), L, dtype=torch.float32, device=device)
    for i, w in enumerate(wav16_list):
        t = torch.as_tensor(np.asarray(w, np.float32), device=device)
        x[i, : t.numel()] = t
    toks, lens = vc.s3gen.tokenizer(x)
    # tokenizer pads to the batch max; recover true per-item token counts from
    # the individual sample counts (25 tokens/s at 16 kHz => 640 samples/token)
    true_lens = torch.tensor([max(1, len(w) // 640) for w in wav16_list],
                             dtype=torch.long, device=device)
    lens = torch.minimum(lens.to(device), true_lens)
    return toks, lens


@torch.inference_mode()
def generate_batch(vc, tokens, tok_lens, n_cand, ref_dict=None, seed=None):
    """(M sources, N candidates) -> one s3gen forward pass.

    tokens   : LongTensor [M, T]  (already padded)
    tok_lens : LongTensor [M]
    returns  : wav tensor [M*N, samples] on GPU, ordered source-major
    """
    ref_dict = ref_dict if ref_dict is not None else vc.ref_dict
    M = tokens.shape[0]
    B = M * n_cand
    tok = tokens.repeat_interleave(n_cand, dim=0).contiguous()
    lens = tok_lens.repeat_interleave(n_cand, dim=0).contiguous()
    rd = _expand_ref(ref_dict, B)
    if seed is not None:
        torch.manual_seed(seed)
    wavs = vc.s3gen.inference(speech_tokens=tok, ref_dict=rd,
                              speech_token_lens=lens)
    if isinstance(wavs, (tuple, list)):
        wavs = wavs[0]
    return wavs


def _expand_ref(ref_dict, B):
    out = {}
    for k, v in ref_dict.items():
        if torch.is_tensor(v):
            if v.dim() >= 1 and v.shape[0] == 1 and B > 1:
                v = v.expand(B, *v.shape[1:]) if v.is_contiguous() else v.repeat(B, *([1] * (v.dim() - 1)))
        out[k] = v
    return out


# --------------------------------------------------------------------------- #
#  SIDON  (batched)
# --------------------------------------------------------------------------- #
MEDIATHEK_SIDON = "/e/data1/datasets/playground/mmlaion/schuhmann1/mediathek_sidon/code"


class Sidon:
    """Batched SIDON, on top of this project's audited batching.

    `mediathek_sidon/code/sidon_batch.py` already solved this properly: it
    featurises per item (the HF stacked-frame extractor's attention_mask
    disagrees with the real frame count for some lengths, which a naive
    `padding=True` batch gets wrong on 15 of 16 items), batches the w2v-BERT
    encoder (safe — causal depthwise conv, re-masked per layer), and groups the
    DAC decoder by *identical* frame count (the decoder is not causal, so a
    padded batch decode contaminates tails). It ran the whole Mediathek corpus
    at ~172x realtime on one GH200.

    The one thing added here: `featurize` is pure CPU/numpy and was the serial
    tail of that design, so it runs in a thread pool overlapped with the GPU.
    """

    def __init__(self, device="cuda:0", threads=8):
        if MEDIATHEK_SIDON not in sys.path:
            sys.path.insert(0, MEDIATHEK_SIDON)
        import sidon_batch as SB
        self.SB = SB
        self.device = device
        self.model = SB.get_model(device)
        self.threads = threads
        self._pool = None

    def _featurize_parallel(self, xs):
        """Same contract as SB.featurize, but the per-item numpy work is spread
        over threads (the HF extractor releases the GIL in its numpy inner loops)."""
        from concurrent.futures import ThreadPoolExecutor
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.threads)
        proc = self.model[2]
        SB = self.SB

        def one(x):
            o = proc(x, sampling_rate=SB.SR_IN, return_tensors="np", padding=True)
            f = o["input_features"][0]
            m = o.get("attention_mask")
            m = np.asarray(m[0]) if m is not None else np.ones(f.shape[0], dtype=np.int64)
            if m.shape[0] != f.shape[0]:
                mm = np.zeros(f.shape[0], dtype=np.int64)
                k = min(m.shape[0], f.shape[0])
                mm[:k] = m[:k]
                m = mm
            return f, m.astype(np.int64)

        res = list(self._pool.map(one, xs))
        fs = [r[0] for r in res]
        ms = [r[1] for r in res]
        T = [f.shape[0] for f in fs]
        Tm = max(T)
        b = np.zeros((len(fs), Tm, fs[0].shape[1]), dtype=np.float32)
        am = np.zeros((len(fs), Tm), dtype=np.int64)
        for i, (f, m) in enumerate(zip(fs, ms)):
            b[i, : T[i]] = f
            am[i, : T[i]] = m
        return b, am, T

    def restore(self, wavs, sr, max_frames=8000, max_items=32, parallel_fe=True):
        """wavs: list of 1-D float arrays at `sr`. Returns list of 48 kHz arrays."""
        SB = self.SB
        if sr != SB.SR_IN:
            xs = []
            for w in wavs:
                t = torch.as_tensor(np.asarray(w, np.float32))
                xs.append(torchaudio.functional.resample(t, sr, SB.SR_IN).numpy())
        else:
            xs = [np.asarray(w, np.float32) for w in wavs]
        old = SB.featurize
        if parallel_fe:
            SB.featurize = lambda proc, a: self._featurize_parallel(a)
        try:
            return SB.enhance_segments(xs, device=self.device, model=self.model,
                                       max_frames=max_frames, max_items=max_items)
        finally:
            SB.featurize = old


# --------------------------------------------------------------------------- #
#  Loudness  (EBU R128 / ITU-R BS.1770 via pyloudnorm)
# --------------------------------------------------------------------------- #
_METERS = {}


def measure_lufs(wav, sr):
    import pyloudnorm as pyln
    if sr not in _METERS:
        _METERS[sr] = pyln.Meter(sr)
    try:
        v = _METERS[sr].integrated_loudness(np.asarray(wav, np.float64))
        return float(v) if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def loudness_normalize(wav, sr, target_lufs=-23.0, peak_ceiling=0.95):
    """EBU R128 integrated loudness -> target, with a peak ceiling.

    Returns (normalised wav, lufs_pre, gain_db, lufs_post, clipped_by_ceiling).
    The ceiling is reported because it is the trap: SIDON hands back audio
    already peak-normalised to 0.9, so a target above the material's natural
    loudness makes the ceiling eat the entire gain and the "normalisation"
    becomes a silent no-op.
    """
    w = np.asarray(wav, np.float32)
    pre = measure_lufs(w, sr)
    gain_db = 0.0
    if np.isfinite(pre) and pre < 0:
        gain_db = target_lufs - pre
        w = w * (10.0 ** (gain_db / 20.0))
    peak = float(np.abs(w).max())
    clipped = False
    if peak > peak_ceiling:
        w = w * (peak_ceiling / peak)
        clipped = True
    post = measure_lufs(w, sr)
    return w.astype(np.float32), pre, gain_db, post, clipped


# --------------------------------------------------------------------------- #
#  Speaker embedders  (batched)
# --------------------------------------------------------------------------- #
class SpeakerSim:
    """ECAPA-TDNN (speechbrain) + Orange/Speaker-wavLM-tbr, both batched.

    Two independent embedders on purpose: the 0.40 floor is an ECAPA-scale
    convention, and WavLM disagrees with it on a large fraction of takes.
    """

    def __init__(self, device="cuda:0", savedir=None, spk_emb_path=None):
        from speechbrain.inference.speaker import EncoderClassifier
        self.device = device
        self.ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=savedir or "/tmp/ecapa_ckpt",
            run_opts={"device": device})
        self.orange = None
        if spk_emb_path:
            sys.path.insert(0, spk_emb_path)
            try:
                from spk_embeddings import EmbeddingsModel
                EmbeddingsModel.all_tied_weights_keys = {}
                self.orange = EmbeddingsModel.from_pretrained(
                    "Orange/Speaker-wavLM-tbr").to(device).eval()
            except Exception as e:
                print(f"[SpeakerSim] Orange/WavLM unavailable: {type(e).__name__}: {e}",
                      flush=True)

    @staticmethod
    def _pad_stack(wavs, device):
        L = max(len(w) for w in wavs)
        x = torch.zeros(len(wavs), L, device=device)
        lens = torch.zeros(len(wavs), device=device)
        for i, w in enumerate(wavs):
            t = torch.as_tensor(np.asarray(w, np.float32), device=device)
            x[i, : t.numel()] = t
            lens[i] = t.numel() / L
        return x, lens

    @torch.inference_mode()
    def ecapa_emb(self, wavs16, max_batch=64):
        out = []
        for s in range(0, len(wavs16), max_batch):
            grp = wavs16[s: s + max_batch]
            x, lens = self._pad_stack(grp, self.device)
            e = self.ecapa.encode_batch(x, wav_lens=lens).squeeze(1)
            out.append(F.normalize(e, dim=-1))
        return torch.cat(out, 0)

    @torch.inference_mode()
    def orange_emb(self, wavs16, max_batch=32):
        if self.orange is None:
            return None
        out = []
        for s in range(0, len(wavs16), max_batch):
            grp = wavs16[s: s + max_batch]
            x, _ = self._pad_stack(grp, self.device)
            e = self.orange(x)
            out.append(F.normalize(e, dim=-1))
        return torch.cat(out, 0)


# --------------------------------------------------------------------------- #
#  Emotion / quality scorer  (BUD-E-Whisper + Empathic-Insight experts)
# --------------------------------------------------------------------------- #
SEQ_LEN, EMB, PROJ = 1500, 768, 64
HID, DROP = [64, 32, 16], [0.0, 0.1, 0.1, 0.1]

EMO40 = ["Affection", "Amusement", "Anger", "Astonishment_Surprise", "Awe", "Bitterness",
         "Concentration", "Confusion", "Contemplation", "Contempt", "Contentment",
         "Disappointment", "Disgust", "Distress", "Doubt", "Elation", "Embarrassment",
         "Emotional_Numbness", "Fatigue_Exhaustion", "Fear", "Helplessness",
         "Hope_Enthusiasm_Optimism", "Impatience_and_Irritability", "Infatuation", "Interest",
         "Intoxication_Altered_States_of_Consciousness", "Jealousy_&_Envy", "Longing",
         "Malevolence_Malice", "Pain", "Pleasure_Ecstasy", "Pride", "Relief", "Sadness",
         "Sexual_Lust", "Shame", "Sourness", "Teasing", "Thankfulness_Gratitude", "Triumph"]
QUALITY = {"Overall_Quality": "model_score_overall_quality_best.pth",
           "Speech_Quality": "model_score_speech_quality_best.pth",
           "Background_Quality": "model_score_background_quality_best.pth",
           "Content_Enjoyment": "model_score_content_enjoyment_best.pth"}


class _FullMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.proj = torch.nn.Linear(SEQ_LEN * EMB, PROJ)
        layers = [torch.nn.ReLU(), torch.nn.Dropout(DROP[0])]
        cur = PROJ
        for i, h in enumerate(HID):
            layers += [torch.nn.Linear(cur, h), torch.nn.ReLU(), torch.nn.Dropout(DROP[i + 1])]
            cur = h
        layers.append(torch.nn.Linear(cur, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        return self.mlp(self.proj(self.flatten(x)))


class _PooledMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3072, PROJ)
        layers = [torch.nn.ReLU(), torch.nn.Dropout(DROP[0])]
        cur = PROJ
        for i, h in enumerate(HID):
            layers += [torch.nn.Linear(cur, h), torch.nn.ReLU(), torch.nn.Dropout(DROP[i + 1])]
            cur = h
        layers.append(torch.nn.Linear(cur, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(self.proj(x))


def _clean(sd):
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    return sd


class Scorer:
    """Frozen BUD-E-Whisper encoder feeding the Empathic-Insight expert heads.

    `emo_names` selects which of the 40 emotion experts to keep resident; the
    full 40 cost ~40 x 46 M params of fp16 linear layers, which is fine on a
    GH200 but pointless if only a handful of dimensions are needed.
    """

    def __init__(self, whisper_dir, emo_dir, qual_dir, device="cuda:0", emo_names=None,
                 extra_dims=()):
        import glob
        from transformers import WhisperModel, WhisperFeatureExtractor
        self.device = device
        self.fe = WhisperFeatureExtractor.from_pretrained(whisper_dir)
        wm = WhisperModel.from_pretrained(whisper_dir, torch_dtype=torch.float16,
                                          low_cpu_mem_usage=True)
        self.enc = wm.get_encoder().to(device).eval()
        files = {os.path.basename(p): p for p in glob.glob(emo_dir + "/model_*_best.pth")}
        qfiles = {os.path.basename(p): p for p in glob.glob(qual_dir + "/model_*_best.pth")}
        files.update({k: v for k, v in qfiles.items() if k not in files})

        def find(name):
            for c in (f"model_{name}_best.pth",
                      f"model_{name.replace('&', 'and')}_best.pth",
                      f"model_{name.replace('&', '_and_').replace('__', '_')}_best.pth"):
                if c in files:
                    return files[c]
            key = name.lower().replace("&", "and").replace("_", "").replace(".", "")
            for b, p in files.items():
                if b[6:-9].lower().replace("&", "and").replace("_", "").replace(".", "") == key:
                    return p
            return None

        self.emo = {}
        for e in list(emo_names if emo_names is not None else EMO40) + list(extra_dims):
            p = find(e)
            if p is None:
                print(f"[Scorer] missing expert {e}", flush=True)
                continue
            m = _FullMLP().to(device)
            m.load_state_dict(_clean(torch.load(p, map_location=device, weights_only=True)))
            self.emo[e] = m.eval().half()
        self.qual = {}
        for lab, fn in QUALITY.items():
            p = os.path.join(qual_dir, fn)
            if not os.path.exists(p):
                continue
            m = _PooledMLP().to(device)
            m.load_state_dict(_clean(torch.load(p, map_location=device, weights_only=True)))
            self.qual[lab] = m.eval().half()

    @torch.inference_mode()
    def embed(self, wavs16, max_batch=64):
        out = []
        for s in range(0, len(wavs16), max_batch):
            inp = self.fe([np.asarray(w, np.float32) for w in wavs16[s: s + max_batch]],
                          sampling_rate=S3_SR, return_tensors="pt",
                          padding="max_length", truncation=True)
            f = inp.input_features.to(self.device, dtype=torch.float16)
            out.append(self.enc(f, return_dict=True).last_hidden_state)
        return torch.cat(out, 0)

    @torch.inference_mode()
    def score(self, emb, names=None):
        names = names if names is not None else list(self.emo)
        res = {e: self.emo[e](emb).squeeze(-1).float().cpu().numpy() for e in names if e in self.emo}
        if self.qual:
            e = emb.float()
            pooled = torch.cat([e.mean(1), e.min(1).values, e.max(1).values, e.std(1)], 1).half()
            for lab, m in self.qual.items():
                res[lab] = m(pooled).squeeze(-1).float().cpu().numpy()
        return res


# --------------------------------------------------------------------------- #
#  IO helpers
# --------------------------------------------------------------------------- #
def gpu_resample_batch(wav_t, src_sr, dst_sr):
    """wav_t: [B, T] on GPU -> [B, T'] on GPU."""
    return torchaudio.functional.resample(wav_t, src_sr, dst_sr)


def encode_mp3(wav, sr, path, bitrate=160000):
    import av
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = np.asarray(wav, np.float32)
    c = av.open(path, "w")
    s = c.add_stream("mp3", rate=sr)
    s.bit_rate = bitrate
    s.layout = "mono"
    fsz = s.codec_context.frame_size or 1152
    pts = 0
    for o in range(0, len(x), fsz):
        ch = np.ascontiguousarray(x[o: o + fsz].clip(-1, 1))
        fr = av.AudioFrame.from_ndarray(ch[None, :], format="flt", layout="mono")
        fr.sample_rate = sr
        fr.pts = pts
        pts += len(ch)
        for pk in s.encode(fr):
            c.mux(pk)
    for pk in s.encode(None):
        c.mux(pk)
    c.close()


def encode_mp3_bytes(wav, sr, bitrate=160000):
    import av
    buf = io.BytesIO()
    x = np.asarray(wav, np.float32)
    c = av.open(buf, "w", format="mp3")
    s = c.add_stream("mp3", rate=sr)
    s.bit_rate = bitrate
    s.layout = "mono"
    fsz = s.codec_context.frame_size or 1152
    pts = 0
    for o in range(0, len(x), fsz):
        ch = np.ascontiguousarray(x[o: o + fsz].clip(-1, 1))
        fr = av.AudioFrame.from_ndarray(ch[None, :], format="flt", layout="mono")
        fr.sample_rate = sr
        fr.pts = pts
        pts += len(ch)
        for pk in s.encode(fr):
            c.mux(pk)
    for pk in s.encode(None):
        c.mux(pk)
    c.close()
    return buf.getvalue()


def decode_audio_bytes(raw, target_sr=None):
    """Decode mp3/wav bytes -> (mono float32 np array, sr). PyAV, no ffmpeg binary."""
    import av
    c = av.open(io.BytesIO(raw))
    st = c.streams.audio[0]
    sr = st.rate
    chunks = []
    rs = None
    if target_sr and target_sr != sr:
        rs = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=target_sr)
    for fr in c.decode(st):
        if rs is not None:
            for f2 in rs.resample(fr):
                chunks.append(f2.to_ndarray().reshape(-1))
        else:
            a = fr.to_ndarray()
            if a.ndim > 1 and a.shape[0] > 1:
                a = a.mean(0)
            chunks.append(np.asarray(a, np.float32).reshape(-1))
    if rs is not None:
        for f2 in rs.resample(None) or []:
            chunks.append(f2.to_ndarray().reshape(-1))
        sr = target_sr
    c.close()
    x = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
    return x.astype(np.float32), sr
