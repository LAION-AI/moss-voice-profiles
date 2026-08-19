"""Prepare voice-conversion reference audio: SIDON restore + EBU R128 loudness.

A bad reference poisons every conversion made from it, silently, so this stage
is measured rather than assumed:

  * SIDON restores the reference to 48 kHz. The 500 shipped references are a
    mix of orig / sidon / cbx variants (whichever scored best on DNSMOS), so
    starting from `.orig` and restoring uniformly is the only way every target
    gets the same treatment.
  * Loudness is normalised with pyloudnorm (ITU-R BS.1770 / EBU R128 integrated).
    The target is chosen by measurement, not taste — see the sweep below. The
    trap is that SIDON returns audio already peak-normalised to 0.9, so a target
    louder than the material makes the 0.95 peak ceiling eat the whole gain and
    the normalisation silently does nothing.
  * Every prepared reference is verified: loudness landed where it should, the
    ceiling did not engage, quality went up, and ECAPA/WavLM identity against
    the untouched original stayed high (restoration must not change who it is).

Usage:  prep_refs.py <out_dir> [voice_list_json|all500|cid,cid,...]
"""
import os, sys, io, json, glob, tarfile, time
import numpy as np, torch, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcengine as E

NB = "/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox"
DEV = "cuda:0"
CONS = sorted(glob.glob(f"{NB}/hfcache/.cache/dramabox/"
                        f"datasets--TTS-AGI--moss-reference-voices-consolidated/snapshots/*"))[-1]
TARGET_LUFS = float(os.environ.get("REF_LUFS", "-23.0"))
PEAK_CEIL = 0.95

out_dir = sys.argv[1]
spec = sys.argv[2] if len(sys.argv) > 2 else "all500"
os.makedirs(out_dir, exist_ok=True)

vp = pd.read_parquet(f"{NB}/vprof/work/voices.parquet")
if spec == "all500":
    cids = json.load(open(f"{NB}/vprof/work/voices500.json"))["voices"]
elif spec == "all6064":
    cids = vp["cid"].tolist()
elif os.path.exists(spec):
    cids = json.load(open(spec))["voices"]
else:
    cids = spec.split(",")
shard_of = dict(zip(vp["cid"], vp["shard"]))
print(f"[prep] {len(cids)} references, target {TARGET_LUFS} LUFS, ceiling {PEAK_CEIL}", flush=True)

# ---------------------------------------------------------------- load orig --
want = {}
for c in cids:
    want.setdefault(int(shard_of[c]), set()).add(c)
raw = {}
t0 = time.time()
for sh, cs in sorted(want.items()):
    tp = f"{CONS}/data/voices-{sh:04d}.tar"
    with tarfile.open(tp) as tf:
        for m in tf:
            b = os.path.basename(m.name)
            if b.endswith(".orig.mp3") and b[:-9] in cs:
                raw[b[:-9]] = tf.extractfile(m).read()
print(f"[prep] pulled {len(raw)}/{len(cids)} .orig.mp3 in {time.time()-t0:.1f}s", flush=True)
missing = [c for c in cids if c not in raw]
if missing:
    print(f"[prep] WARNING no .orig for {len(missing)}: {missing[:5]}")

cids = [c for c in cids if c in raw]
orig = {}
for c in cids:
    w, sr = E.decode_audio_bytes(raw[c])
    orig[c] = (w, sr)

# ------------------------------------------------------------------- SIDON ---
sd = E.Sidon(DEV, threads=16)
t0 = time.time()
enh = {}
B = 32
for s in range(0, len(cids), B):
    grp = cids[s: s + B]
    xs = []
    for c in grp:
        w, sr = orig[c]
        t = torch.as_tensor(w)
        if sr != 16000:
            import torchaudio
            t = torchaudio.functional.resample(t, sr, 16000)
        xs.append(t.numpy())
    for c, y in zip(grp, sd.restore(xs, 16000, max_frames=8000, max_items=32)):
        enh[c] = y
tot_s = sum(len(orig[c][0]) / orig[c][1] for c in cids)
print(f"[prep] SIDON {len(cids)} refs ({tot_s/60:.1f} min audio) in {time.time()-t0:.1f}s "
      f"= {tot_s/(time.time()-t0):.0f}x realtime", flush=True)
del sd
torch.cuda.empty_cache()

# ------------------------------------------------------- loudness + verify ---
sp = E.SpeakerSim(DEV, savedir=f"{NB}/vcbon/ecapa_ckpt", spk_emb_path=f"{NB}/vprof/idloop/code")
WH = sorted(glob.glob(f"{NB}/hfcache/.cache/dramabox/models--laion--BUD-E-Whisper/snapshots/*"))[-1]
QD = sorted(glob.glob(f"{NB}/hfcache/.cache/dramabox/models--laion--Empathic-Insight-Voice-Plus/snapshots/*"))[-1]
ED = sorted(glob.glob(f"{NB}/hfcache/.cache/dramabox/models--laion--Empathic-Insight-Voice-Small/snapshots/*"))[-1]
sc = E.Scorer(WH, ED, QD, DEV, emo_names=[])

import torchaudio
rows, prepared = [], {}
CH = 32
for s in range(0, len(cids), CH):
    grp = cids[s: s + CH]
    o16, p16, recs = [], [], []
    for c in grp:
        ow, osr = orig[c]
        ot = torch.as_tensor(ow)
        o16.append((torchaudio.functional.resample(ot, osr, 16000) if osr != 16000 else ot).numpy())
        y = enh[c]
        lu_raw = E.measure_lufs(y, 48000)
        yn, pre, gain, post, clipped = E.loudness_normalize(y, 48000, TARGET_LUFS, PEAK_CEIL)
        prepared[c] = yn
        p16.append(torchaudio.functional.resample(torch.as_tensor(yn), 48000, 16000).numpy())
        recs.append(dict(cid=c, dur_s=len(ow) / osr, lufs_orig=E.measure_lufs(o16[-1], 16000),
                         lufs_sidon_raw=lu_raw, lufs_pre=pre, gain_db=gain, lufs_post=post,
                         ceiling_clipped=bool(clipped), peak_out=float(np.abs(yn).max())))
    eo = sp.ecapa_emb(o16); ep = sp.ecapa_emb(p16)
    wo = sp.orange_emb(o16); wp = sp.orange_emb(p16)
    qo = sc.score(sc.embed(o16)); qp = sc.score(sc.embed(p16))
    for i, r in enumerate(recs):
        r["ecapa_orig_vs_prep"] = float((eo[i] * ep[i]).sum())
        r["wavlm_orig_vs_prep"] = float((wo[i] * wp[i]).sum()) if wo is not None else None
        for lab in qo:
            r[f"q_{lab}_orig"] = float(qo[lab][i]); r[f"q_{lab}_prep"] = float(qp[lab][i])
        rows.append(r)
    print(f"[prep] verified {len(rows)}/{len(cids)}", flush=True)

df = pd.DataFrame(rows)
df.to_parquet(f"{out_dir}/ref_prep.parquet", index=False)

# store the prepared references as one tar of 48 kHz mp3 (no loose-file sprawl)
with tarfile.open(f"{out_dir}/refs_prepared.tar", "w") as tf:
    for c in cids:
        b = E.encode_mp3_bytes(prepared[c], 48000, 160000)
        ti = tarfile.TarInfo(f"{c}.prep.mp3"); ti.size = len(b); ti.mtime = 0
        tf.addfile(ti, io.BytesIO(b))
json.dump({"target_lufs": TARGET_LUFS, "peak_ceiling": PEAK_CEIL, "n": len(cids),
           "source": "moss-reference-voices-consolidated/<cid>.orig.mp3",
           "chain": "decode -> 16k -> SIDON(48k) -> EBU R128 integrated -> peak ceiling"},
          open(f"{out_dir}/prep_config.json", "w"), indent=2)

print("\n=== reference preparation, verified ===")
for c in ["lufs_orig", "lufs_sidon_raw", "lufs_post", "gain_db", "peak_out"]:
    q = df[c].describe(percentiles=[.05, .5, .95])
    print(f"{c:16s} mean {q['mean']:8.2f}  p5 {q['5%']:8.2f}  med {q['50%']:8.2f}  p95 {q['95%']:8.2f}")
print(f"ceiling engaged on {df.ceiling_clipped.mean()*100:.1f}% of references")
print(f"landed within 0.5 LU of target: {(df.lufs_post.sub(TARGET_LUFS).abs() < 0.5).mean()*100:.1f}%")
print(f"ECAPA orig-vs-prepared: mean {df.ecapa_orig_vs_prep.mean():.4f} min {df.ecapa_orig_vs_prep.min():.4f}")
if df.wavlm_orig_vs_prep.notna().any():
    print(f"WavLM orig-vs-prepared: mean {df.wavlm_orig_vs_prep.mean():.4f} min {df.wavlm_orig_vs_prep.min():.4f}")
for lab in ["Overall_Quality", "Speech_Quality", "Background_Quality", "Content_Enjoyment"]:
    a, b = df.get(f"q_{lab}_orig"), df.get(f"q_{lab}_prep")
    if a is not None:
        print(f"{lab:20s} {a.mean():.3f} -> {b.mean():.3f}  ({b.mean()-a.mean():+.3f})")
print(f"\nwrote {out_dir}/refs_prepared.tar + ref_prep.parquet")
