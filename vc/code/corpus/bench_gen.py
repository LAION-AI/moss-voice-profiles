"""Throughput sweep for batched best-of-N Chatterbox voice conversion.

Three axes, measured on real corpus takes (mean 9.72 s, the corpus mean):

  N  candidates per source     — the best-of-N width
  M  sources packed per pass   — several *different* sources in one forward
  bucketing                    — sources sorted by length so padding is small

Plus correctness: packing several sources into one padded batch is only a win
if the padding does not change the audio. That is checked here rather than
assumed, against the natural seed-to-seed spread as the control.
"""
import os, sys, glob, json, time, tarfile
import numpy as np, torch, torchaudio, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcengine as E

NB = "/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox"
DEV = "cuda:0"
PIL = f"{NB}/vcbon/pilot"
OUT = f"{NB}/vcbon/out"
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ sources --
mf = pd.read_parquet(f"{PIL}/sources.parquet")
srcs = {}
with tarfile.open(f"{PIL}/sources.tar") as tf:
    for m in tf:
        srcs[m.name] = E.decode_audio_bytes(tf.extractfile(m).read(), target_sr=16000)[0]
mf["n16"] = mf["key"].map(lambda k: len(srcs[k]))
print(f"[bench] {len(mf)} sources, mean {mf.dur.mean():.2f}s", flush=True)

vc = E.load_vc(DEV)
print(f"[bench] s3gen dtype={vc.s3gen.dtype}", flush=True)

# prepared reference for the first pilot voice
V0 = mf.src_voice.iloc[0]
with tarfile.open(f"{NB}/vcbon/refs/refs_prepared.tar") as tf:
    rb = tf.extractfile(f"{V0}.prep.mp3").read()
rw, rsr = E.decode_audio_bytes(rb)
E.set_target_from_wav(vc, rw, rsr, peak_norm=None)

pool = mf.sort_values("n16").reset_index(drop=True)
med = pool.iloc[len(pool) // 2 - 24: len(pool) // 2 + 24]           # 48 near-median clips
results = []


def run(keys, N, seed=1234):
    ws = [srcs[k] for k in keys]
    torch.cuda.synchronize(); t0 = time.time()
    tok, ln = E.tokenize(vc, ws, DEV)
    torch.cuda.synchronize(); t_tok = time.time() - t0
    t0 = time.time()
    w = E.generate_batch(vc, tok, ln, N, seed=seed)
    torch.cuda.synchronize(); t_gen = time.time() - t0
    return w, ln, t_tok, t_gen


# --------------------------------------------------- axis 1: N, single source --
print("\n=== axis 1: candidates per source, M=1 (the upstream design) ===", flush=True)
k = med.key.iloc[24]
d = float(med.dur.iloc[24])
run([k], 2)                                                          # warm
for N in (1, 2, 4, 8, 16, 32, 64):
    ts = []
    for r in range(3):
        torch.cuda.reset_peak_memory_stats()
        _, _, tt, tg = run([k], N, seed=1000 + r)
        ts.append(tg)
    t = float(np.median(ts))
    results.append(dict(axis="N", M=1, N=N, dur=d, gen_s=t, per_cand_ms=t / N * 1000,
                        rtf=t / (N * d), mem_gb=torch.cuda.max_memory_allocated() / 1e9))
    print(f"  N={N:3d}  {t:6.3f}s  {t/N*1000:7.1f} ms/cand  RTF {t/(N*d):.4f}  "
          f"{N*d/t:6.1f}x realtime  mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)

# ------------------------------------------- axis 2: pack M sources, N fixed=8 --
print("\n=== axis 2: sources packed per forward pass, N=8, length-bucketed ===", flush=True)
for M in (1, 2, 4, 8, 16, 32):
    ks = list(med.key.iloc[:M])
    ds = float(med.dur.iloc[:M].sum())
    ts = []
    for r in range(3):
        torch.cuda.reset_peak_memory_stats()
        try:
            _, _, tt, tg = run(ks, 8, seed=2000 + r)
        except torch.OutOfMemoryError:
            tg = float("nan"); torch.cuda.empty_cache(); break
        ts.append(tg)
    if not ts:
        print(f"  M={M:3d}  OOM"); continue
    t = float(np.median(ts))
    nc = M * 8
    results.append(dict(axis="M", M=M, N=8, dur=ds, gen_s=t, per_cand_ms=t / nc * 1000,
                        rtf=t / (8 * ds), mem_gb=torch.cuda.max_memory_allocated() / 1e9))
    print(f"  M={M:3d}  {nc:4d} cand  {t:6.3f}s  {t/nc*1000:7.1f} ms/cand  RTF {t/(8*ds):.4f}  "
          f"{8*ds/t:6.1f}x realtime  mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)

# ------------------------------------ axis 3: bucketed vs unbucketed packing --
print("\n=== axis 3: length bucketing (M=8, N=8) ===", flush=True)
rng = np.random.default_rng(0)
for mode in ("bucketed", "shuffled"):
    tot_gen, tot_aud, tot_pad = 0.0, 0.0, 0.0
    idx = np.arange(len(pool)) if mode == "bucketed" else rng.permutation(len(pool))
    for s in range(0, 64, 8):
        ks = list(pool.key.iloc[idx[s:s + 8]])
        ds = float(pool.dur.iloc[idx[s:s + 8]].sum())
        w, ln, _, tg = run(ks, 8, seed=3000 + s)
        tot_gen += tg; tot_aud += ds * 8
        tot_pad += (int(ln.max()) * len(ks) - int(ln.sum())) / max(int(ln.sum()), 1)
    results.append(dict(axis="bucket", mode=mode, M=8, N=8, gen_s=tot_gen,
                        aud_s=tot_aud, rtf=tot_gen / tot_aud, pad_waste=tot_pad / 8))
    print(f"  {mode:10s} {tot_gen:6.2f}s for {tot_aud:7.1f}s audio  RTF {tot_gen/tot_aud:.4f}  "
          f"mean pad waste {tot_pad/8*100:5.1f}%", flush=True)

# ------------------------------------------------------- axis 4: dtype / TF32 --
print("\n=== axis 4: precision ===", flush=True)
ks = list(med.key.iloc[:8])
ds = float(med.dur.iloc[:8].sum())
for name, setup in [("fp32 (default)", lambda: torch.backends.cuda.matmul.__setattr__("allow_tf32", False)),
                    ("tf32 matmul", lambda: torch.backends.cuda.matmul.__setattr__("allow_tf32", True))]:
    setup()
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    run(ks, 8)
    ts = [run(ks, 8, seed=4000 + r)[3] for r in range(3)]
    t = float(np.median(ts))
    results.append(dict(axis="dtype", mode=name, M=8, N=8, gen_s=t, rtf=t / (8 * ds)))
    print(f"  {name:16s} {t:6.3f}s  RTF {t/(8*ds):.4f}  {8*ds/t:6.1f}x realtime", flush=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# --------------------------------------------------- correctness of packing --
print("\n=== correctness: does packing change the audio? ===", flush=True)
WH = sorted(glob.glob(f"{NB}/hfcache/.cache/dramabox/models--laion--BUD-E-Whisper/snapshots/*"))[-1]
QD = sorted(glob.glob(f"{NB}/hfcache/.cache/dramabox/models--laion--Empathic-Insight-Voice-Plus/snapshots/*"))[-1]
ED = sorted(glob.glob(f"{NB}/hfcache/.cache/dramabox/models--laion--Empathic-Insight-Voice-Small/snapshots/*"))[-1]
sc = E.Scorer(WH, ED, QD, DEV, emo_names=[])


def env_corr(a, b, hop=480):
    n = min(len(a), len(b)) // hop * hop
    ea = np.sqrt((a[:n].reshape(-1, hop) ** 2).mean(1) + 1e-12)
    eb = np.sqrt((b[:n].reshape(-1, hop) ** 2).mean(1) + 1e-12)
    return float(np.corrcoef(np.log(ea), np.log(eb))[0, 1])


tk = list(med.key.iloc[:8])
solo = {}
for i, k in enumerate(tk):
    w, ln, _, _ = run([k], 2, seed=5000)
    n = int(ln[0]) * 960
    solo[k] = [w[j, :n].float().cpu().numpy() for j in range(2)]
w, ln, _, _ = run(tk, 1, seed=6000)
packed = {k: w[i, : int(ln[i]) * 960].float().cpu().numpy() for i, k in enumerate(tk)}

ctrl_env = [env_corr(solo[k][0], solo[k][1]) for k in tk]            # seed-vs-seed control
pack_env = [env_corr(solo[k][0], packed[k]) for k in tk]
lens_ok = all(len(packed[k]) == len(solo[k][0]) for k in tk)
e_solo = sc.embed([torchaudio.functional.resample(torch.as_tensor(solo[k][0]), 24000, 16000).numpy() for k in tk])
e_solo2 = sc.embed([torchaudio.functional.resample(torch.as_tensor(solo[k][1]), 24000, 16000).numpy() for k in tk])
e_pack = sc.embed([torchaudio.functional.resample(torch.as_tensor(packed[k]), 24000, 16000).numpy() for k in tk])
cs = torch.nn.functional.cosine_similarity(e_solo.mean(1).float(), e_solo2.mean(1).float()).cpu().numpy()
cp = torch.nn.functional.cosine_similarity(e_solo.mean(1).float(), e_pack.mean(1).float()).cpu().numpy()
print(f"  output lengths identical: {lens_ok}")
print(f"  log-energy envelope corr : control(seed-vs-seed) {np.mean(ctrl_env):.4f}  "
      f"packed-vs-solo {np.mean(pack_env):.4f}")
print(f"  BUD-E encoder cosine     : control {cs.mean():.4f}  packed-vs-solo {cp.mean():.4f}")
results.append(dict(axis="pack_correct", lens_ok=bool(lens_ok),
                    ctrl_env=float(np.mean(ctrl_env)), pack_env=float(np.mean(pack_env)),
                    ctrl_emb=float(cs.mean()), pack_emb=float(cp.mean())))

json.dump(results, open(f"{OUT}/bench_gen.json", "w"), indent=2, default=float)
print(f"\nwrote {OUT}/bench_gen.json")
