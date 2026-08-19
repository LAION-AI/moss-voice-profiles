"""Validate the whole stack end-to-end on a handful of real corpus takes."""
import os, sys, time, tarfile, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcengine as E

NB = "/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox"
DEV = "cuda:0"
VOICE = sys.argv[1] if len(sys.argv) > 1 else "anime_000"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8

t0 = time.time()
tar = f"{NB}/vprof/vp500/{VOICE}/PPILOT2/cands-000.tar"
srcs = []
with tarfile.open(tar) as tf:
    for m in tf:
        if not m.name.endswith(".mp3"):
            continue
        w, sr = E.decode_audio_bytes(tf.extractfile(m).read(), target_sr=16000)
        srcs.append((m.name, w))
        if len(srcs) >= 6:
            break
print(f"[smoke] {len(srcs)} sources from {VOICE}, decode {time.time()-t0:.1f}s", flush=True)
for n, w in srcs:
    print(f"   {n}  {len(w)/16000:.2f}s")

t0 = time.time()
vc = E.load_vc(DEV)
print(f"[smoke] VC loaded {time.time()-t0:.1f}s  dtype={vc.s3gen.dtype}  sr={vc.sr}", flush=True)

ref_mp3 = None
import glob
for v in ("sidon", "orig", "cbx"):
    p = f"{NB}/vprof/work/refs/{VOICE}.{v}.mp3"
    if os.path.exists(p):
        ref_mp3 = p
        break
print("[smoke] ref:", ref_mp3)
rw, rsr = E.decode_audio_bytes(open(ref_mp3, "rb").read())
E.set_target_from_wav(vc, rw, rsr, peak_norm=0.97)
print("[smoke] ref_dict:", {k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in vc.ref_dict.items()})

# ---- generation: single source x N, then multi-source packing ----
toks, lens = E.tokenize(vc, [srcs[0][1]], DEV)
print("[smoke] tokens", tuple(toks.shape), "lens", lens.tolist())
torch.cuda.synchronize(); t0 = time.time()
w = E.generate_batch(vc, toks, lens, N, seed=1234)
torch.cuda.synchronize()
print(f"[smoke] 1 src x {N} cand -> {tuple(w.shape)} in {time.time()-t0:.2f}s", flush=True)

M = 4
toks, lens = E.tokenize(vc, [s[1] for s in srcs[:M]], DEV)
print("[smoke] multi tokens", tuple(toks.shape), "lens", lens.tolist())
torch.cuda.synchronize(); t0 = time.time()
w2 = E.generate_batch(vc, toks, lens, N, seed=1234)
torch.cuda.synchronize()
dt = time.time() - t0
print(f"[smoke] {M} src x {N} cand = {M*N} -> {tuple(w2.shape)} in {dt:.2f}s "
      f"({dt/(M*N)*1000:.0f} ms/cand)", flush=True)
print("[smoke] peak GPU mem %.1f GB" % (torch.cuda.max_memory_allocated() / 1e9))

# variation check
a = w2[:N].float().cpu().numpy()
L = min(x.shape[-1] for x in a)
cc = np.mean([np.corrcoef(a[i][:L], a[j][:L])[0, 1] for i in range(4) for j in range(i + 1, 4)])
print(f"[smoke] mean pairwise waveform corr among candidates: {cc:.3f}")

# does padding change the result? compare src0 alone vs src0 in the M-batch
t1, l1 = E.tokenize(vc, [srcs[0][1]], DEV)
wa = E.generate_batch(vc, t1, l1, 1, seed=7)[0].float().cpu().numpy()
tb, lb = E.tokenize(vc, [s[1] for s in srcs[:M]], DEV)
wb = E.generate_batch(vc, tb, lb, 1, seed=7)[0].float().cpu().numpy()
L = min(len(wa), len(wb))
print(f"[smoke] padded-vs-solo len {len(wa)} vs {len(wb)}  corr={np.corrcoef(wa[:L], wb[:L])[0,1]:.4f}")

# ---- SIDON batched ----
t0 = time.time()
sd = E.Sidon(DEV, cache_dir=os.environ.get("HF_HUB_CACHE"))
print(f"[smoke] SIDON loaded {time.time()-t0:.1f}s", flush=True)
cands = [w[i].float().cpu().numpy() for i in range(min(8, w.shape[0]))]
torch.cuda.synchronize(); t0 = time.time()
r = sd.restore(cands, 24000, max_batch=8)
torch.cuda.synchronize()
print(f"[smoke] SIDON batched {len(cands)} in {time.time()-t0:.2f}s "
      f"({(time.time()-t0)/len(cands)*1000:.0f} ms/cand) out {[len(x)/48000 for x in r[:3]]}")
torch.cuda.synchronize(); t0 = time.time()
r1 = [sd.restore([c], 24000)[0] for c in cands]
torch.cuda.synchronize()
print(f"[smoke] SIDON serial  {len(cands)} in {time.time()-t0:.2f}s "
      f"({(time.time()-t0)/len(cands)*1000:.0f} ms/cand)")
d = [float(np.corrcoef(a[:min(len(a),len(b))], b[:min(len(a),len(b))])[0,1]) for a, b in zip(r, r1)]
print(f"[smoke] batched-vs-serial SIDON corr: {np.mean(d):.4f} min {np.min(d):.4f}")

# ---- speaker sim ----
t0 = time.time()
sp = E.SpeakerSim(DEV, savedir=f"{NB}/vcbon/ecapa_ckpt",
                  spk_emb_path=f"{NB}/vprof/idloop/code")
print(f"[smoke] SpeakerSim loaded {time.time()-t0:.1f}s orange={sp.orange is not None}", flush=True)
import torchaudio
w16 = torchaudio.functional.resample(w[:8].float(), 24000, 16000).cpu().numpy()
rw16, _ = E.decode_audio_bytes(open(ref_mp3, "rb").read(), target_sr=16000)
t0 = time.time()
ce = sp.ecapa_emb(list(w16)); re_ = sp.ecapa_emb([rw16])
print(f"[smoke] ECAPA batch {tuple(ce.shape)} in {time.time()-t0:.2f}s  sims={(ce@re_.T).squeeze().tolist()}")
if sp.orange is not None:
    t0 = time.time()
    oe = sp.orange_emb(list(w16)); ro = sp.orange_emb([rw16])
    print(f"[smoke] WavLM batch {tuple(oe.shape)} in {time.time()-t0:.2f}s  sims={(oe@ro.T).squeeze().tolist()}")
print("[smoke] OK")
