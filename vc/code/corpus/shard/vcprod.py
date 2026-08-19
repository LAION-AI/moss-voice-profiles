"""Scenario 1 — best-of-4 voice conversion + SIDON + reward ranking, one shard.

Pipeline order is fixed by the owner and is NOT the pilot's order:

    generate 4 candidates  ->  SIDON all 4  ->  rank the SIDON-enhanced 4
                           ->  store all 4 outputs + every score + every reward

The ranker therefore scores exactly the audio that ships. The pilot's
"SIDON is harmful" number was measured on select-then-enhance, a different
pipeline, and does not transfer.

Everything is stored. `is_winner_*` columns are a VIEW over stored scores:
three different rewards are written side by side, so changing the reward later
is a parquet query, not 3,588 GPU-h of recomputation.

Measured facts carried from the pilot (do not rediscover):
  * SIDON via mediathek_sidon/sidon_batch.py  (38.3 ms/clip; the upstream
    TorchScript loop measures 11,017 ms/clip, 290x slower, and is bit-identical)
  * TF32 on s3gen is free (identical to 4 dp) and gives 1.60x
  * TF32 on SIDON costs ~15 dB -> it is explicitly disabled around SIDON
  * length-bucket before batching: pad waste 51.4% -> 8.2%
  * saturation at ~64 items in flight -> pack 16 sources x 4 candidates
  * references are prepared at EBU R128 -23 LUFS, ceiling 0.95
"""
import os, sys, io, json, time, glob, tarfile, argparse, hashlib, traceback
import numpy as np, torch, torchaudio, pandas as pd
from concurrent.futures import ThreadPoolExecutor

NB = "/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox"
sys.path.insert(0, f"{NB}/vcbon/code")
sys.path.insert(0, NB)
import vcengine as E

VC_ROOT = f"{NB}/vprof/vc500"
RUN_TAG = "VC1"
OUT_SR = 24000            # s3gen native
SIDON_SR = 48000          # SIDON native, and the shipped rate
MP3_KBPS = 160000         # same as the source corpus and the pilot
MIN_TOK = 32              # 1.28 s: below this s3gen's flow convolutions cannot pad

EMO40 = ["Affection", "Amusement", "Anger", "Astonishment_Surprise", "Awe", "Bitterness",
         "Concentration", "Confusion", "Contemplation", "Contempt", "Contentment",
         "Disappointment", "Disgust", "Distress", "Doubt", "Elation", "Embarrassment",
         "Emotional_Numbness", "Fatigue_Exhaustion", "Fear", "Helplessness",
         "Hope_Enthusiasm_Optimism", "Impatience_and_Irritability", "Infatuation", "Interest",
         "Intoxication_Altered_States_of_Consciousness", "Longing", "Malevolence_Malice",
         "Pain", "Pleasure_Ecstasy", "Pride", "Relief", "Sadness", "Sexual_Lust", "Shame",
         "Sourness", "Teasing", "Thankfulness_Gratitude", "Triumph", "Jealousy_&_Envy"]
QUAL4 = ["content_enjoyment", "overall_quality", "speech_quality", "background_quality"]


# --------------------------------------------------------------------------- #
#  target dimension: which single number is "target emotion strength" here
# --------------------------------------------------------------------------- #
def gid_target(gid, block, emotion):
    """Returns (kind, name, sign).

    The corpus mixes two kinds of conditioning. `voicenet` blocks aim a
    continuous VoiceNet dimension high or low (the gid encodes which), so the
    "strength" is the signed regression value. Everything else aims an emotion
    label, so the strength is that emotion expert's score. The two live on
    different scales and are z-scored against different groups accordingly.
    """
    p = str(gid).split("|")
    if block == "voicenet" and len(p) >= 5:
        return ("dim", p[2], 1.0 if "high" in p[3] else -1.0)
    if emotion and str(emotion) != "nan" and pd.notna(emotion):
        return ("emo", str(emotion), 1.0)
    return ("free", "", 1.0)


def z_set(x):
    """z-score within one candidate set (the pilot's reward). Stored for
    comparison only; see NORM.md for why it is not the production reward."""
    x = np.asarray(x, np.float64)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)


def mm_set(x):
    x = np.asarray(x, np.float64)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(x)


class Norm:
    """Frozen group-level z-score constants.

    Estimated once from the smoke shards, then frozen for the whole run and
    recorded in norm_stats.json. Frozen, not per-shard, because a reward whose
    normalisation drifts between shards is not comparable across the corpus.
    """

    def __init__(self, path):
        self.d = None
        if path and os.path.exists(path):
            try:
                d = json.load(open(path))
                self.d = d if d.get("emo") and d.get("qual") else None
            except Exception as e:
                print(f"[Norm] unusable {path}: {type(e).__name__}: {e}; "
                      f"falling back to within-set ranking", flush=True)

    def emo(self, kind, name, sgn, x):
        """Group key includes the SIGN.

        A `voicenet` gid aims a dimension either high or low, and the stored
        strength is `sign x regression`. Pooling both directions into one group
        gives a bimodal distribution (measured: -2.32 and +2.64 for the same
        dimension), whose sd is a distance between two modes rather than a
        spread — z-scoring against it under-weights the emotion term wherever
        the corpus happens to ask for both directions. Splitting by sign makes
        each group unimodal and the z-score mean what it says.
        """
        if self.d is None:
            return np.zeros_like(np.asarray(x, np.float64))
        e = self.d["emo"]
        st = (e.get(f"{kind}:{name}:{int(sgn)}") or e.get(f"{kind}:{name}")
              or e.get(f"{kind}:*:{int(sgn)}") or e.get(f"{kind}:*") or e["*"])
        return (np.asarray(x, np.float64) - st["mean"]) / max(st["sd"], 1e-9)

    def qual(self, x):
        if self.d is None:
            return np.zeros_like(np.asarray(x, np.float64))
        st = self.d["qual"]
        return (np.asarray(x, np.float64) - st["mean"]) / max(st["sd"], 1e-9)


# --------------------------------------------------------------------------- #
def decode_shard(tar_path, keys, threads=16):
    """Sequential tar read + threaded mp3 decode -> {key: 16 kHz mono float32}.

    Read is sequential (one pass, no seeks); decode is the expensive half and
    PyAV releases the GIL, so it goes to a pool. Decoding a whole ~10 k-sample
    shard up front costs ~2-3 % of the shard's GPU time and removes all random
    access from the hot loop.
    """
    want = set(keys)
    out = {}
    pool = ThreadPoolExecutor(max_workers=threads)
    futs = {}
    with tarfile.open(tar_path) as tf:
        for m in tf:
            if m.name not in want:
                continue
            raw = tf.extractfile(m).read()
            futs[pool.submit(E.decode_audio_bytes, raw, 16000)] = m.name
    for f, k in futs.items():
        try:
            out[k] = f.result()[0]
        except Exception as e:
            print(f"[decode] FAILED {k}: {type(e).__name__}: {e}", flush=True)
    pool.shutdown(wait=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, help="<voice>/<sh>  e.g. anime_000/002")
    ap.add_argument("--n-cand", type=int, default=4)
    ap.add_argument("--pack", type=int, default=16, help="sources per s3gen forward pass")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--norm", default=f"{NB}/vcbon/prod/index/norm_stats.json")
    ap.add_argument("--out-root", default=VC_ROOT)
    ap.add_argument("--run-tag", default=RUN_TAG)
    ap.add_argument("--ab-presidon", type=int, default=0,
                    help="also score the PRE-SIDON candidates (diagnostic A/B, costs a scoring pass)")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--no-marker", type=int, default=0)
    a = ap.parse_args()

    voice, sh = a.shard.split("/")
    sh = int(sh)
    dev = "cuda:0"
    run_dir = f"{NB}/vprof/vp500/{voice}/PPILOT2"
    out_dir = f"{a.out_root}/{voice}/{a.run_tag}"
    os.makedirs(out_dir, exist_ok=True)
    tmp_tar = f"{out_dir}/.vc-{sh:03d}.tar.partial"
    fin_tar = f"{out_dir}/vc-{sh:03d}.tar"
    marker = f"{out_dir}/done-{sh:03d}.json"

    if os.path.exists(marker) and not a.no_marker:
        print(f"[vcprod] {a.shard} already complete", flush=True)
        return 0

    T0 = time.time()
    stage = dict(decode=0.0, tokenize=0.0, generate=0.0, sidon=0.0, score=0.0,
                 spk=0.0, mp3=0.0, io=0.0, presidon_score=0.0)

    # ------------------------------------------------------------- sources ---
    metas = sorted(glob.glob(f"{run_dir}/meta-{sh:03d}-*.parquet"))
    cols = ["audio_key", "gid", "cand", "dur", "block", "lang", "emotion", "condition",
            "dim", "level", "spk_sim", "emo_strength", "quality", "blend", "genuineness",
            "dim_target", "profile", "subset", "text"]
    import pyarrow.parquet as pq
    avail = set(pq.ParquetFile(metas[0]).schema.names)
    use = [c for c in cols if c in avail]
    mf = pd.concat([pd.read_parquet(m, columns=use) for m in metas], ignore_index=True)
    if a.limit and a.limit < len(mf):
        # STRIDE, not head. The meta is ordered by gid, so head(n) returns one
        # block's takes and nothing else -- a 320-row head of anime_000/000 was
        # 100 % `burst_isolated` / target_kind `free`. A stride sample spans the
        # blocks, languages and durations in their real proportions, which is
        # what both the normalisation constants and the throughput number need.
        mf = mf.iloc[:: max(1, len(mf) // a.limit)].head(a.limit).reset_index(drop=True)
    assert mf["audio_key"].is_unique, "audio_key repeats inside a shard"
    print(f"[vcprod] {a.shard}: {len(mf)} sources from {len(metas)} meta files", flush=True)

    # ------------------------------------------------------------- decode ----
    # Decode runs on a background thread while the models load. Both are ~40-400 s
    # per shard and neither needs the other; serialising them would waste ~28 GPU-h
    # of pure model-load time across 2,000 shards.
    t_dec0 = time.time()
    _dec_pool = ThreadPoolExecutor(max_workers=1)
    dec_fut = _dec_pool.submit(decode_shard, f"{run_dir}/cands-{sh:03d}.tar",
                               list(mf.audio_key), a.threads)

    # ------------------------------------------------------------- models ----
    load = {}
    t0 = time.time(); vc = E.load_vc(dev); load["vc"] = time.time() - t0
    t0 = time.time(); sd = E.Sidon(dev, threads=a.threads); load["sidon"] = time.time() - t0
    t0 = time.time()
    sp = E.SpeakerSim(dev, savedir=f"{NB}/vcbon/ecapa_ckpt", spk_emb_path=f"{NB}/vprof/idloop/code")
    load["spk"] = time.time() - t0
    t0 = time.time()
    from pp_scores_fast import FastScorer
    fs = FastScorer("cuda")
    load["scorer"] = time.time() - t0
    print(f"[vcprod] models loaded {json.dumps({k: round(v,1) for k,v in load.items()})}", flush=True)

    # ---- join the decode that has been running underneath the model load ----
    src16 = dec_fut.result()
    _dec_pool.shutdown(wait=True)
    stage["decode"] = time.time() - t_dec0
    have = mf.audio_key.isin(src16.keys())
    if (~have).any():
        print(f"[vcprod] WARNING {int((~have).sum())} sources missing from tar", flush=True)
        mf = mf[have].reset_index(drop=True)
    if not len(mf):
        print("[vcprod] no decodable sources in shard", flush=True)
        return 4
    audio_s_in = float(sum(len(v) / 16000 for v in src16.values()))
    print(f"[vcprod] decoded {len(src16)} clips, {audio_s_in/3600:.2f} h, "
          f"{stage['decode']:.0f}s wall incl. model load "
          f"({audio_s_in/max(stage['decode'],1e-9):.0f}x rt)", flush=True)

    # ------------------------------------------------------------ reference --
    with tarfile.open(f"{NB}/vcbon/refs500/refs_prepared.tar") as tf:
        rw, rsr = E.decode_audio_bytes(tf.extractfile(f"{voice}.prep.mp3").read())
    # TF32 ON for s3gen only. Measured free (scores identical to 4 dp), 1.60x.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    E.set_target_from_wav(vc, rw, rsr, peak_norm=None)   # ref already loudness-prepped
    r16 = torchaudio.functional.resample(torch.as_tensor(rw), rsr, 16000).numpy()
    tgt_ec = sp.ecapa_emb([r16])[0]
    tgt_tb = sp.orange_emb([r16])[0]
    man = json.load(open(f"{NB}/vprof/work/manifest_{voice}.json"))
    import librosa
    o16, _ = librosa.load(man["ref_wav"], sr=16000, mono=True)
    tgt_ec0 = sp.ecapa_emb([o16])[0]
    tgt_tb0 = sp.orange_emb([o16])[0]

    ref_json = dict(
        voice=voice, run_tag=a.run_tag, scenario=1,
        target_reference="vcbon/refs500/refs_prepared.tar :: %s.prep.mp3" % voice,
        reference_prep="SIDON restore -> EBU R128 integrated -23.0 LUFS, peak ceiling 0.95",
        reference_original=man["ref_wav"],
        model="ResembleAI/chatterbox s3gen VC",
        sidon="mediathek_sidon/code/sidon_batch.py (batched); TF32 DISABLED for SIDON",
        s3gen_tf32=True, n_candidates=a.n_cand, pack=a.pack,
        out_sample_rate=SIDON_SR, mp3_bitrate=MP3_KBPS,
        pipeline="generate N -> SIDON all N -> rank SIDON-enhanced -> store all N",
        reward="normalise(target emotion strength) + normalise(quality); "
               "production reward = frozen group z-score (reward_group). "
               "reward_set (within-set z) and reward_minmax (within-set min-max) "
               "are stored alongside so selection is a view.",
        join_key="(source_run_dir, source_audio_key) -- audio_key is NOT unique across runs",
        key_format="<target>/<source>/<gid>.c<NNN>.k<J>",
    )
    with open(f"{out_dir}/REF.json", "w") as f:
        json.dump(ref_json, f, indent=2)

    NORM = Norm(a.norm)
    have_norm = NORM.d is not None

    # ------------------------------------------------------------- the loop --
    mf["_n"] = mf["audio_key"].map(lambda k: len(src16[k]))
    mf = mf.sort_values("_n").reset_index(drop=True)   # length-bucketing
    keys = list(mf.audio_key)
    tar = tarfile.open(tmp_tar, "w")
    rows, prov_rows = [], []
    n_cand_total = 0
    n_short_padded = 0
    failed_packs = []
    mp3_pool = ThreadPoolExecutor(max_workers=max(4, a.threads // 2))
    # mp3 encoding is CPU and the tar write must be serial, so the drain is
    # deferred by one pack: this pack's encodes run on the pool while the GPU
    # does the next pack's generate/SIDON, and only then are they muxed.
    pending_mp3 = []
    t_run0 = time.time()

    for s in range(0, len(keys), a.pack):
      # One pack is the fault-isolation unit. Two different data-dependent
      # crashes showed up in the first four full shards, both of which would
      # otherwise have thrown away hours of completed work in the same shard and
      # been re-queued straight back into the same input. A failed pack is
      # recorded by source key and skipped; the shard still completes, its
      # verification is computed against what was actually produced, and the
      # marker carries the failure list so the gap is visible and re-runnable
      # rather than silent.
      try:
        # snapshot so a failed pack leaves no half-written rows behind
        _n_rows0, _n_prov0 = len(rows), len(prov_rows)
        _pending0, _ncand0 = list(pending_mp3), n_cand_total
        ks = keys[s: s + a.pack]
        sub = mf.iloc[s: s + a.pack].reset_index(drop=True)
        ws = [src16[k] for k in ks]
        M, N = len(ks), a.n_cand

        # ---- very short sources: pad up, generate, trim back ----
        # 1.08 % of the corpus is under 1.28 s and 0.20 % is under 0.16 s
        # (measured over 604,478 takes). s3gen's flow convolutions cannot run on
        # them: a 0.08 s clip is 2 tokens and the decoder tries to pad (4,4) into
        # a dimension of 4. Because packs are length-sorted the shortest clips in
        # a shard all land in its FIRST pack, so this is not a rare tail case —
        # it kills the shard in the first minute. `mediathek_0051/003` (minimum
        # duration 0.08 s) died exactly there.
        #
        # Sources are padded with silence to MIN_TOK tokens for generation AND
        # for SIDON, and the audio is trimmed back to its true length only after
        # restoration. Padding rather than skipping, because the spec is to
        # convert all 20,125,736 takes. Trimming after SIDON rather than before,
        # because SIDON's stacked-frame feature extractor is not safe on a 0.08 s
        # input either. The cost is that SIDON sees a little trailing silence and
        # is non-causal, so a very short clip's restoration is not bit-identical
        # to restoring it alone — accepted, and recorded, as the alternative is
        # not converting these takes at all.
        orig_tok = [max(1, len(w) // 640) for w in ws]
        need = MIN_TOK * 640
        n_padded = 0
        for i, w in enumerate(ws):
            if len(w) < need:
                ws[i] = np.concatenate([w, np.zeros(need - len(w), np.float32)])
                n_padded += 1
        n_short_padded += n_padded

        torch.cuda.synchronize(); t0 = time.time()
        tok, ln = E.tokenize(vc, ws, dev)
        # --- latent bug in the shared tokenizer path, fixed here rather than in
        # vcengine.py so the pilot's artefact stays byte-identical ---
        # `E.tokenize` pads `tok` to whatever width the S3 tokenizer returns for
        # the batch, then clamps `ln` down to the true per-clip token count
        # (samples // 640). When the clamp lowers the batch MAXIMUM, the padded
        # width no longer equals max(ln). s3gen's flow builds its mask from
        # `token_len` and multiplies it against the embedded tokens, so the two
        # disagree by one frame and it dies with
        #   "size of tensor a (230) must match ... tensor b (229)".
        # It needs a specific length combination, which is why the pilot, the
        # 320-source smoke and 17 stride-sampled 2,000-source shards all missed
        # it and the first full 9,968-source shard hit it 348 s in.
        m = int(ln.max().item())
        if tok.shape[1] > m:
            tok = tok[:, :m].contiguous()
        elif tok.shape[1] < m:
            ln = torch.clamp(ln, max=tok.shape[1])
        torch.cuda.synchronize(); stage["tokenize"] += time.time() - t0

        # ---- 1. generate N candidates (TF32 on) ----
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        t0 = time.time()
        seed = int(hashlib.blake2b(f"{a.shard}|{s}".encode(), digest_size=4).hexdigest(), 16)
        wav = E.generate_batch(vc, tok, ln, N, seed=seed)
        torch.cuda.synchronize(); stage["generate"] += time.time() - t0
        n_cand_total += M * N

        t0 = time.time()
        cands = []
        for i in range(M):
            n = int(ln[i]) * 960
            cands.append(wav[i * N:(i + 1) * N, :n])
        torch.cuda.synchronize(); stage["io"] += time.time() - t0

        # ---- optional diagnostic: score the PRE-SIDON candidates ----
        pre = None
        if a.ab_presidon:
            t0 = time.time()
            pre16 = [torchaudio.functional.resample(cands[i][j].float(), OUT_SR, 16000).cpu().numpy()
                     for i in range(M) for j in range(N)]
            psc = fs.score_batch([torch.as_tensor(x) for x in pre16])
            pec = sp.ecapa_emb(pre16); ptb = sp.orange_emb(pre16)
            pre = dict(sc=psc, ec=(pec @ tgt_ec).cpu().numpy(), tb=(ptb @ tgt_tb).cpu().numpy(),
                       ec0=(pec @ tgt_ec0).cpu().numpy(), tb0=(ptb @ tgt_tb0).cpu().numpy())
            stage["presidon_score"] += time.time() - t0

        # ---- 2. SIDON on ALL N candidates (TF32 OFF: measured ~15 dB cost) ----
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        t0 = time.time()
        flat = [cands[i][j].float().cpu().numpy() for i in range(M) for j in range(N)]
        sw = sd.restore(flat, OUT_SR, max_frames=8000, max_items=32)
        torch.cuda.synchronize(); stage["sidon"] += time.time() - t0
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # trim padded-short sources back to their true length, at 48 kHz
        # (24 kHz generation -> 48 kHz SIDON output, so 960 samples/token x 2)
        for i in range(M):
            n48 = orig_tok[i] * 960 * 2
            for j in range(N):
                k = i * N + j
                if len(sw[k]) > n48:
                    sw[k] = sw[k][:n48]

        sw16 = [torchaudio.functional.resample(torch.as_tensor(x), SIDON_SR, 16000).numpy()
                for x in sw]

        # ---- 3. score the SIDON-enhanced candidates: this is what ships ----
        t0 = time.time()
        SC = fs.score_batch([torch.as_tensor(x) for x in sw16])
        stage["score"] += time.time() - t0
        t0 = time.time()
        ec = sp.ecapa_emb(sw16); tb = sp.orange_emb(sw16)
        ecs = (ec @ tgt_ec).cpu().numpy(); tbs = (tb @ tgt_tb).cpu().numpy()
        ecs0 = (ec @ tgt_ec0).cpu().numpy(); tbs0 = (tb @ tgt_tb0).cpu().numpy()
        stage["spk"] += time.time() - t0

        # ---- 4. reward + rank, then encode all N ----
        t0 = time.time()
        mp3_futs = []
        for i in range(M):
            r = sub.iloc[i]
            sl = slice(i * N, (i + 1) * N)
            S = SC[sl]
            kind, name, sgn = gid_target(r["gid"], r["block"], r.get("emotion"))
            if kind == "dim":
                strength = np.array([sgn * float(x["voicenet"].get(name, {}).get("reg", np.nan))
                                     if isinstance(x["voicenet"].get(name), dict) else np.nan
                                     for x in S])
            elif kind == "emo":
                strength = np.array([float(x["emonet"].get(name, np.nan)) for x in S])
            else:
                strength = np.array([float(max(x["emonet"].values())) for x in S])
            ovq = np.array([float(x["quality"].get("overall_quality", np.nan)) for x in S])
            st = np.nan_to_num(strength, nan=float(np.nanmean(strength))
                               if np.isfinite(strength).any() else 0.0)
            oq = np.nan_to_num(ovq, nan=float(np.nanmean(ovq))
                               if np.isfinite(ovq).any() else 0.0)

            zg_e, zg_q = NORM.emo(kind, name, sgn, st), NORM.qual(oq)
            rw_group = zg_e + zg_q
            zs_e, zs_q = z_set(st), z_set(oq)
            rw_set = zs_e + zs_q
            mm_e, mm_q = mm_set(st), mm_set(oq)
            rw_mm = mm_e + mm_q

            order_g = np.argsort(-rw_group) if have_norm else np.argsort(-rw_set)
            win_g = int(order_g[0])
            win_s = int(np.argsort(-rw_set)[0])
            win_m = int(np.argsort(-rw_mm)[0])

            src_key = str(r["audio_key"])
            base_key = src_key[:-4] if src_key.endswith(".mp3") else src_key
            for j in range(N):
                X = S[j]
                # self-describing key: <target>/<source>/<gid>.c<NNN>.k<J>
                key = f"{voice}/{voice}/{base_key}.k{j}"
                mp3_futs.append((key, mp3_pool.submit(
                    E.encode_mp3_bytes, sw[i * N + j], SIDON_SR, MP3_KBPS)))
                row = dict(
                    shard_id=a.shard, voice=voice, sh=sh,
                    source_run_dir=run_dir, source_audio_key=src_key,
                    gid=str(r["gid"]), source_cand=int(r["cand"]),
                    block=str(r["block"]), lang=str(r.get("lang")),
                    emotion=(None if pd.isna(r.get("emotion")) else str(r.get("emotion"))),
                    condition=(None if pd.isna(r.get("condition")) else str(r.get("condition"))),
                    target_voice=voice, target_kind=kind, target_name=name, target_sign=float(sgn),
                    cand_idx=j, out_key=key + ".mp3",
                    dur_src=float(r["dur"]),
                    # ---- source-side sensors, for before/after on the same instrument
                    spk_sim_src=float(r["spk_sim"]) if pd.notna(r.get("spk_sim")) else None,
                    emo_strength_src=float(r["emo_strength"]) if pd.notna(r.get("emo_strength")) else None,
                    quality_src=float(r["quality"]) if pd.notna(r.get("quality")) else None,
                    blend_src=float(r["blend"]) if pd.notna(r.get("blend")) else None,
                    genuineness_src=float(r["genuineness"]) if pd.notna(r.get("genuineness")) else None,
                    dim_target_src=float(r["dim_target"]) if pd.notna(r.get("dim_target")) else None,
                    # ---- converted, SIDON-enhanced: raw scores, nothing discarded
                    emo_target_raw=float(strength[j]) if np.isfinite(strength[j]) else None,
                    qual_raw=float(ovq[j]) if np.isfinite(ovq[j]) else None,
                    blend_vc=float(X["blend_0_10"]), genuineness_vc=float(X["genuineness_0_6"]),
                    ecapa_to_prepared_ref=float(ecs[i * N + j]),
                    wavlm_to_prepared_ref=float(tbs[i * N + j]),
                    ecapa_to_original_ref=float(ecs0[i * N + j]),
                    wavlm_to_original_ref=float(tbs0[i * N + j]),
                    # ---- reward components, all of them
                    z_emo_group=float(zg_e[j]), z_qual_group=float(zg_q[j]),
                    reward_group=float(rw_group[j]),
                    z_emo_set=float(zs_e[j]), z_qual_set=float(zs_q[j]),
                    reward_set=float(rw_set[j]),
                    mm_emo_set=float(mm_e[j]), mm_qual_set=float(mm_q[j]),
                    reward_minmax=float(rw_mm[j]),
                    rank_group=int(np.where(order_g == j)[0][0]),
                    is_winner_group=bool(j == win_g),
                    is_winner_set=bool(j == win_s),
                    is_winner_minmax=bool(j == win_m),
                    sidon=True,
                )
                for e in EMO40:
                    row["emo_" + e] = float(X["emonet"].get(e, np.nan))
                for q in QUAL4:
                    row["q_" + q] = float(X["quality"].get(q, np.nan))
                for dnm, dv in X["voicenet"].items():
                    row["vn_" + dnm] = float(dv["reg"])
                if pre is not None:
                    P = pre["sc"][i * N + j]
                    if kind == "dim":
                        pv = P["voicenet"].get(name)
                        ps = sgn * float(pv["reg"]) if isinstance(pv, dict) else np.nan
                    elif kind == "emo":
                        ps = float(P["emonet"].get(name, np.nan))
                    else:
                        ps = float(max(P["emonet"].values()))
                    row.update(
                        pre_emo_target_raw=(float(ps) if np.isfinite(ps) else None),
                        pre_qual_raw=float(P["quality"].get("overall_quality", np.nan)),
                        pre_blend=float(P["blend_0_10"]),
                        pre_genuineness=float(P["genuineness_0_6"]),
                        pre_speech_q=float(P["quality"].get("speech_quality", np.nan)),
                        pre_ecapa=float(pre["ec"][i * N + j]),
                        pre_wavlm=float(pre["tb"][i * N + j]),
                        pre_ecapa_origref=float(pre["ec0"][i * N + j]),
                        pre_wavlm_origref=float(pre["tb0"][i * N + j]),
                    )
                rows.append(row)

            prov_rows.append(dict(
                shard_id=a.shard, source_run_dir=run_dir, source_audio_key=src_key,
                source_gid=str(r["gid"]), source_cand=int(r["cand"]),
                target_voice=voice, source_voice=voice, arm="self", scenario=1,
                n_candidates=N, out_tar=f"vc-{sh:03d}.tar",
                out_keys=[f"{voice}/{voice}/{base_key}.k{jj}.mp3" for jj in range(N)],
                winner_group=int(win_g), winner_set=int(win_s), winner_minmax=int(win_m),
                block=str(r["block"]), lang=str(r.get("lang")),
                target_kind=kind, target_name=name,
                dur_src=float(r["dur"]),
                model="ResembleAI/chatterbox s3gen VC", sidon=True,
                sidon_impl="mediathek_sidon/sidon_batch.py",
                out_sample_rate=SIDON_SR, mp3_bitrate=MP3_KBPS,
                target_ref="prepared: SIDON + EBU R128 -23 LUFS, ceiling 0.95",
                run_tag=a.run_tag, seed=int(seed),
            ))

        stage["io"] += time.time() - t0
        t0 = time.time()
        for key, fut in pending_mp3:            # drain the PREVIOUS pack
            b = fut.result()
            ti = tarfile.TarInfo(key + ".mp3"); ti.size = len(b); ti.mtime = 0
            tar.addfile(ti, io.BytesIO(b))
        pending_mp3 = mp3_futs
        stage["mp3"] += time.time() - t0

        if (s // a.pack) % 25 == 0:
            el = time.time() - t_run0
            done = s + M
            print(f"[vcprod] {a.shard} {done}/{len(keys)}  {el:.0f}s  "
                  f"{el/max(done,1):.3f} s/sample  eta {el/max(done,1)*(len(keys)-done)/60:.1f} min",
                  flush=True)

      except Exception as exc:
        # Roll the pack back completely: stored rows, provenance, the candidate
        # counter, and this pack's un-muxed mp3 futures. The previous pack's
        # pending futures are restored so they are still written by the next
        # drain. Then carry on with the next pack.
        del rows[_n_rows0:]
        del prov_rows[_n_prov0:]
        pending_mp3, n_cand_total = _pending0, _ncand0
        bad = list(keys[s: s + a.pack])
        failed_packs.append(dict(pack_start=s, n=len(bad), keys=bad,
                                 error=f"{type(exc).__name__}: {exc}"))
        print(f"[vcprod] PACK FAILED at {s} ({len(bad)} sources): "
              f"{type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        torch.cuda.empty_cache()

    t0 = time.time()
    for key, fut in pending_mp3:                # final drain
        b = fut.result()
        ti = tarfile.TarInfo(key + ".mp3"); ti.size = len(b); ti.mtime = 0
        tar.addfile(ti, io.BytesIO(b))
    stage["mp3"] += time.time() - t0
    tar.close()
    mp3_pool.shutdown(wait=True)
    wall = time.time() - t_run0

    # ------------------------------------------------------------- persist ---
    t0 = time.time()
    cand = pd.DataFrame(rows)
    prov = pd.DataFrame(prov_rows)
    cand.to_parquet(f"{out_dir}/.cand-{sh:03d}.parquet.partial", index=False,
                    compression="zstd")
    prov.to_parquet(f"{out_dir}/.prov-{sh:03d}.parquet.partial", index=False,
                    compression="zstd")

    # ---- verify BEFORE the marker: tar readable, member count right, parquet joins ----
    n_expect = len(prov) * a.n_cand
    with tarfile.open(tmp_tar) as tf:
        names = tf.getnames()
    ok_tar = len(names) == n_expect
    ok_join = set(cand.out_key) == set(names)
    ok_rows = len(cand) == n_expect
    ok_uni = not cand.duplicated(["source_run_dir", "source_audio_key", "cand_idx"]).any()
    ok_win = bool((cand.groupby(["source_audio_key"])["is_winner_group"].sum() == 1).all())
    n_failed_src = sum(f["n"] for f in failed_packs)
    # Verification is against what was actually produced, plus an explicit
    # accounting of what was not: sources = produced + failed must equal the
    # shard's input. A shard that quietly converted 9,000 of 10,000 takes and
    # called itself done is the failure this guards against.
    ok_account = (len(prov) + n_failed_src) == len(keys)
    # A shard is only allowed to lose a small tail this way; anything more is a
    # systematic fault pretending to be a few bad packs, and must not be marked
    # complete.
    ok_loss = n_failed_src <= max(64, int(0.01 * len(keys)))
    verify = dict(tar_members=len(names), expected=n_expect, ok_tar=ok_tar, ok_join=ok_join,
                  ok_rows=ok_rows, ok_unique=ok_uni, ok_one_winner=ok_win,
                  ok_accounting=ok_account, ok_loss_within_tolerance=ok_loss,
                  n_sources_in=len(keys), n_sources_done=len(prov),
                  n_sources_failed=n_failed_src, n_packs_failed=len(failed_packs),
                  n_short_padded=n_short_padded)
    if not all([ok_tar, ok_join, ok_rows, ok_uni, ok_win, ok_account, ok_loss]):
        print(f"[vcprod] VERIFY FAILED {json.dumps(verify)}", flush=True)
        with open(f"{out_dir}/failed-{sh:03d}.json", "w") as fh:
            json.dump(dict(shard=a.shard, verify=verify, failed_packs=failed_packs), fh, indent=2)
        return 3
    if failed_packs:
        with open(f"{out_dir}/failed-{sh:03d}.json", "w") as fh:
            json.dump(dict(shard=a.shard, verify=verify, failed_packs=failed_packs), fh, indent=2)

    os.replace(tmp_tar, fin_tar)
    os.replace(f"{out_dir}/.cand-{sh:03d}.parquet.partial", f"{out_dir}/cand-{sh:03d}.parquet")
    os.replace(f"{out_dir}/.prov-{sh:03d}.parquet.partial", f"{out_dir}/prov-{sh:03d}.parquet")
    stage["io"] += time.time() - t0

    n_samp = len(prov)
    rec = dict(shard=a.shard, n_samples=n_samp, n_candidates=n_cand_total,
               n_cand=a.n_cand, pack=a.pack, audio_s_in=audio_s_in,
               wall_s=wall, total_s=time.time() - T0, load_s=load, stage_s=stage,
               s_per_sample=wall / max(n_samp, 1),
               gpu_h_per_1k=wall / max(n_samp, 1) * 1000 / 3600,
               realtime_x=audio_s_in / max(wall, 1e-9),
               sidon_ms_per_clip=stage["sidon"] * 1000 / max(n_cand_total, 1),
               out_bytes=os.path.getsize(fin_tar),
               verify=verify, have_norm=have_norm, ab_presidon=bool(a.ab_presidon),
               n_short_padded=n_short_padded, n_packs_failed=len(failed_packs),
               n_sources_failed=int(sum(f['n'] for f in failed_packs)),
               host=os.environ.get("SLURMD_NODENAME", ""), jobid=os.environ.get("SLURM_JOB_ID", ""),
               finished_at=time.time())
    if not a.no_marker:
        with open(marker, "w") as f:
            json.dump(rec, f, indent=2)
    print("[vcprod] DONE " + json.dumps(rec, default=float), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
