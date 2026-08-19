"""Does the prepared reference actually make better conversions?

A reference is a silent failure mode: a bad one degrades every conversion made
from it and nothing in the output says so. So the preparation chain is ablated
rather than trusted, on real corpus takes, with the same scorers used
everywhere else.

Arms (all use the same sources, same seeds):
  raw_peak    original reference, peak-normalised to 0.97   (upstream default)
  raw_lufs    original reference, EBU R128 -23 LUFS
  sidon_peak  SIDON-restored, then peak-normalised to 0.97  (throws the level away)
  sidon_18    SIDON-restored, EBU R128 -18 LUFS             (ceiling eats the gain)
  sidon_23    SIDON-restored, EBU R128 -23 LUFS             (the chosen chain)

Also: TF32 on/off at the chosen arm, since TF32 is a 1.6x throughput lever and
the project has already measured it costing ~15 dB elsewhere.
"""
import os, sys, io, json, glob, time, tarfile
import numpy as np, torch, torchaudio, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox")
import vcengine as E

NB = "/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox"
DEV = "cuda:0"
N = 8
NSRC = int(os.environ.get("NSRC", "24"))
CONS = sorted(glob.glob(f"{NB}/hfcache/.cache/dramabox/"
                        f"datasets--TTS-AGI--moss-reference-voices-consolidated/snapshots/*"))[-1]

mf = pd.read_parquet(f"{NB}/vcbon/pilot/sources.parquet")
src16 = {}
with tarfile.open(f"{NB}/vcbon/pilot/sources.tar") as tf:
    for m in tf:
        src16[m.name] = E.decode_audio_bytes(tf.extractfile(m).read(), target_sr=16000)[0]

vc = E.load_vc(DEV)
sd = E.Sidon(DEV, threads=16)
sp = E.SpeakerSim(DEV, savedir=f"{NB}/vcbon/ecapa_ckpt", spk_emb_path=f"{NB}/vprof/idloop/code")
from pp_scores_fast import FastScorer
fs = FastScorer("cuda")
vp = pd.read_parquet(f"{NB}/vprof/work/voices.parquet")
shard_of = dict(zip(vp["cid"], vp["shard"]))

rows = []
for VOICE in sorted(mf.src_voice.unique()):
    sub = mf[mf.src_voice == VOICE].sort_values("dur").head(NSRC)
    ws = [src16[k] for k in sub.key]

    with tarfile.open(f"{CONS}/data/voices-{int(shard_of[VOICE]):04d}.tar") as tf:
        raw = None
        for m in tf:
            if os.path.basename(m.name) == f"{VOICE}.orig.mp3":
                raw = tf.extractfile(m).read(); break
    ow, osr = E.decode_audio_bytes(raw)
    o16 = torchaudio.functional.resample(torch.as_tensor(ow), osr, 16000).numpy()
    sw48 = sd.restore([o16], 16000)[0]

    variants = {}
    p = np.abs(ow).max(); variants["raw_peak"] = (ow * (0.97 / p), osr)
    variants["raw_lufs"] = (E.loudness_normalize(ow, osr, -23.0)[0], osr)
    p = np.abs(sw48).max(); variants["sidon_peak"] = (sw48 * (0.97 / p), 48000)
    variants["sidon_18"] = (E.loudness_normalize(sw48, 48000, -18.0)[0], 48000)
    variants["sidon_23"] = (E.loudness_normalize(sw48, 48000, -23.0)[0], 48000)

    # identity anchor: every arm is scored against the SAME reference embedding,
    # taken from the untouched original, so the arms are comparable
    anch_ec = sp.ecapa_emb([o16])[0]; anch_tb = sp.orange_emb([o16])[0]

    for arm, (rw, rsr) in variants.items():
        for tf32 in ((0, 1) if arm == "sidon_23" else (0,)):
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            E.set_target_from_wav(vc, rw, rsr, peak_norm=None)
            t_gen = 0.0
            for s in range(0, len(ws), 8):
                grp = ws[s:s + 8]
                tok, ln = E.tokenize(vc, grp, DEV)
                torch.cuda.synchronize(); t0 = time.time()
                w = E.generate_batch(vc, tok, ln, N, seed=777 + s)
                torch.cuda.synchronize(); t_gen += time.time() - t0
                c16 = []
                for i in range(len(grp)):
                    n = int(ln[i]) * 960
                    blk = w[i * N:(i + 1) * N, :n].float()
                    c16 += list(torchaudio.functional.resample(blk, 24000, 16000).cpu().numpy())
                sc = fs.score_batch([torch.as_tensor(x) for x in c16])
                ec = (sp.ecapa_emb(c16) @ anch_ec).cpu().numpy()
                tb = (sp.orange_emb(c16) @ anch_tb).cpu().numpy()
                for j in range(len(c16)):
                    r = sub.iloc[s + j // N]
                    rows.append(dict(voice=VOICE, arm=arm, tf32=tf32, key=r["key"],
                                     block=r["block"], emotion=r["emotion"],
                                     ecapa=float(ec[j]), wavlm=float(tb[j]),
                                     overall_q=float(sc[j]["quality"].get("overall_quality", np.nan)),
                                     speech_q=float(sc[j]["quality"].get("speech_quality", np.nan)),
                                     blend=float(sc[j]["blend_0_10"]),
                                     genuineness=float(sc[j]["genuineness_0_6"]),
                                     peak_emo=float(max(sc[j]["emonet"].values())),
                                     emo_tgt=(float(sc[j]["emonet"].get(r["emotion"], np.nan))
                                              if r["emotion"] else np.nan),
                                     gen_s=t_gen))
            print(f"[ablate] {VOICE} {arm} tf32={tf32}: gen {t_gen:.1f}s  "
                  f"ecapa {np.mean([x['ecapa'] for x in rows[-len(ws)*N:]]):.4f}", flush=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

df = pd.DataFrame(rows)
df.to_parquet(f"{NB}/vcbon/out/ref_ablate.parquet", index=False)
g = df[df.tf32 == 0].groupby("arm").agg(
    n=("ecapa", "size"), ecapa=("ecapa", "mean"), wavlm=("wavlm", "mean"),
    overall_q=("overall_q", "mean"), speech_q=("speech_q", "mean"),
    blend=("blend", "mean"), genuine=("genuineness", "mean"),
    peak_emo=("peak_emo", "mean"), emo_tgt=("emo_tgt", "mean"),
    above_floor=("ecapa", lambda x: (x >= 0.40).mean()))
print("\n=== reference preparation ablation (tf32 off) ===")
print(g.to_string())
t = df[df.arm == "sidon_23"].groupby("tf32").agg(
    ecapa=("ecapa", "mean"), overall_q=("overall_q", "mean"), peak_emo=("peak_emo", "mean"),
    blend=("blend", "mean"), gen_s=("gen_s", "max"))
print("\n=== TF32 at sidon_23 ===")
print(t.to_string())
json.dump({"ablation": g.reset_index().to_dict("records"),
           "tf32": t.reset_index().to_dict("records")},
          open(f"{NB}/vcbon/out/ref_ablate.json", "w"), indent=2, default=float)
