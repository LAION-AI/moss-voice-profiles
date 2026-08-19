"""SIDON: reference torchscript path vs the project's batched path."""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcengine as E

DEV = "cuda:0"
t0 = time.time()
sd = E.Sidon(DEV, threads=16)
print(f"[sidon] loaded {time.time()-t0:.1f}s", flush=True)

rng = np.random.default_rng(0)
for dur in (5.0, 9.72, 16.0):
    n = int(dur * 24000)
    for B in (1, 4, 8, 16, 32, 64):
        wavs = [(rng.standard_normal(n) * 0.03).astype(np.float32) for _ in range(B)]
        sd.restore(wavs[:1], 24000)                     # warm
        torch.cuda.synchronize(); t0 = time.time()
        out = sd.restore(wavs, 24000, max_items=B, max_frames=max(8000, B * int(dur * 50) + 10))
        torch.cuda.synchronize(); dt = time.time() - t0
        print(f"dur={dur:5.2f}s B={B:3d}  {dt:7.2f}s  {dt/B*1000:8.1f} ms/clip  "
              f"RTF={dt/(B*dur):.4f}  x-realtime={B*dur/dt:7.1f}  out={len(out[0])/48000:.2f}s",
              flush=True)
    print()

# serial-vs-batched equality
wavs = [(rng.standard_normal(int(9.72 * 24000)) * 0.03).astype(np.float32) for _ in range(8)]
a = sd.restore(wavs, 24000, max_items=8)
b = [sd.restore([w], 24000)[0] for w in wavs]
cc = [float(np.corrcoef(x[:min(len(x), len(y))], y[:min(len(x), len(y))])[0, 1]) for x, y in zip(a, b)]
print(f"batched-vs-serial corr: mean {np.mean(cc):.6f} min {np.min(cc):.6f}")

# parallel-FE vs serial-FE equality
c = sd.restore(wavs, 24000, max_items=8, parallel_fe=False)
cc2 = [float(np.abs(np.asarray(x) - np.asarray(y)).max()) for x, y in zip(a, c)]
print(f"parallelFE-vs-serialFE max abs diff: {max(cc2):.2e}")
print("[sidon] peak GPU mem %.1f GB" % (torch.cuda.max_memory_allocated() / 1e9))
