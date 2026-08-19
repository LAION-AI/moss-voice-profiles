"""End-to-end cost of the *recommended* configuration vs the pilot's.

The pilot measured a deliberately conservative setup (fp32, full corpus-grade
scoring on all 8 candidates, SIDON on the winner, single-threaded MP3). Three
of those turned out to be free or harmful, so the production config is:

  A  pilot           fp32, score all 8 fully, SIDON winner, serial mp3
  B  +tf32           identical outputs (measured: scores match to 4 dp), 1.6x gen
  C  +threaded mp3   PyAV in a thread pool; encoding was 15% of the pilot budget
  D  -sidon out      SIDON on converted output measured NEGATIVE on quality
  E  +cheap ranking  full scorer on the winner only; rank on quality + target dim

Each arm is timed on the same clips, same order, so the deltas are the levers.
"""
import os, sys, io, json, glob, time, tarfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch, torchaudio, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox")
import vcengine as E

NB = "/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox"
DEV = "cuda:0"
N = 8
PACK = 8
NSRC = int(os.environ.get("NSRC", "96"))

mf = pd.read_parquet(f"{NB}/vcbon/pilot/sources.parquet")
src16 = {}
with tarfile.open(f"{NB}/vcbon/pilot/sources.tar") as tf:
    for m in tf:
        src16[m.name] = E.decode_audio_bytes(tf.extractfile(m).read(), target_sr=16000)[0]
# a duration-REPRESENTATIVE subset (evenly spaced through the sorted durations),
# then length-bucketed, so the measured ms/sample is for the corpus's real mix
mf = mf.assign(_n=mf["key"].map(lambda k: len(src16[k]))).sort_values("_n").reset_index(drop=True)
mf = mf.iloc[np.linspace(0, len(mf) - 1, min(NSRC, len(mf))).astype(int)].sort_values("_n")
keys = list(mf["key"])
audio_s = sum(len(src16[k]) / 16000 for k in keys)
print(f"[bench] {len(keys)} sources, {audio_s:.0f}s audio, mean {audio_s/len(keys):.2f}s", flush=True)

vc = E.load_vc(DEV)
sd = E.Sidon(DEV, threads=16)
sp = E.SpeakerSim(DEV, savedir=f"{NB}/vcbon/ecapa_ckpt", spk_emb_path=f"{NB}/vprof/idloop/code")
from pp_scores_fast import FastScorer
fs = FastScorer("cuda")

V = mf.src_voice.iloc[0]
with tarfile.open(f"{NB}/vcbon/refs500/refs_prepared.tar") as tf:
    rw, rsr = E.decode_audio_bytes(tf.extractfile(f"{V}.prep.mp3").read())
E.set_target_from_wav(vc, rw, rsr, peak_norm=None)
tgt_ec = sp.ecapa_emb([torchaudio.functional.resample(torch.as_tensor(rw), rsr, 16000).numpy()])[0]
pool = ThreadPoolExecutor(max_workers=16)


def run(tf32=False, thread_mp3=False, sidon_out=True, cheap_rank=False, warm=False,
        n_cand=N, pack=PACK):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    st = dict(gen=0.0, score=0.0, spk=0.0, sidon=0.0, rescore=0.0, mp3=0.0, misc=0.0)
    t_all = time.time()
    ks = keys[:16] if warm else keys
    for s in range(0, len(ks), pack):
        grp = ks[s:s + pack]
        ws = [src16[k] for k in grp]
        t0 = time.time()
        tok, ln = E.tokenize(vc, ws, DEV)
        torch.cuda.synchronize(); st["misc"] += time.time() - t0
        t0 = time.time()
        w = E.generate_batch(vc, tok, ln, n_cand, seed=11 + s)
        torch.cuda.synchronize(); st["gen"] += time.time() - t0
        t0 = time.time()
        cands, c16 = [], []
        for i in range(len(grp)):
            n = int(ln[i]) * 960
            blk = w[i * n_cand:(i + 1) * n_cand, :n].float()
            cands.append(blk)
            c16 += list(torchaudio.functional.resample(blk, 24000, 16000).cpu().numpy())
        torch.cuda.synchronize(); st["misc"] += time.time() - t0
        t0 = time.time()
        ec = sp.ecapa_emb(c16); ecs = (ec @ tgt_ec).cpu().numpy()
        torch.cuda.synchronize(); st["spk"] += time.time() - t0
        # ---- ranking ----
        t0 = time.time()
        if cheap_rank:
            # quality experts only (4 pooled heads) + speaker sim; the full
            # 57-VoiceNet + 40-emonet pass is deferred to the winner
            q = fs._budE_batch([fs._prep(torch.as_tensor(x)) for x in c16])[1]
            with torch.no_grad():
                ov = fs.qual["overall_quality"](q).reshape(-1).float().detach().cpu().numpy()
            rank_score = ov
        else:
            sc = fs.score_batch([torch.as_tensor(x) for x in c16])
            ov = np.array([float(x["quality"]["overall_quality"]) for x in sc])
            rank_score = ov
        torch.cuda.synchronize(); st["score"] += time.time() - t0
        rk = rank_score + 0.0 * ecs
        wins = [int(np.argmax(rk[i * n_cand:(i + 1) * n_cand])) for i in range(len(grp))]
        # ---- winner path ----
        outs = []
        if sidon_out:
            t0 = time.time()
            outs = sd.restore([cands[i][j].cpu().numpy() for i, j in enumerate(wins)], 24000,
                              max_frames=8000, max_items=32)
            torch.cuda.synchronize(); st["sidon"] += time.time() - t0
            osr = 48000
        else:
            outs = [cands[i][j].cpu().numpy() for i, j in enumerate(wins)]
            osr = 24000
        t0 = time.time()
        o16 = [torchaudio.functional.resample(torch.as_tensor(x), osr, 16000).numpy() for x in outs]
        _ = fs.score_batch([torch.as_tensor(x) for x in o16])
        _ = sp.ecapa_emb(o16)
        torch.cuda.synchronize(); st["rescore"] += time.time() - t0
        t0 = time.time()
        if thread_mp3:
            list(pool.map(lambda x: E.encode_mp3_bytes(x, osr, 160000), outs))
        else:
            [E.encode_mp3_bytes(x, osr, 160000) for x in outs]
        st["mp3"] += time.time() - t0
    wall = time.time() - t_all
    return wall, st, len(ks)


print("[bench] warm-up", flush=True)
run(warm=True); run(tf32=True, warm=True)

arms = [("A pilot config", dict()),
        ("B +tf32", dict(tf32=True)),
        ("C +threaded mp3", dict(tf32=True, thread_mp3=True)),
        ("D -sidon on output", dict(tf32=True, thread_mp3=True, sidon_out=False)),
        ("E +cheap ranking", dict(tf32=True, thread_mp3=True, sidon_out=False, cheap_rank=True))]
res = []
for name, kw in arms:
    wall, st, n = run(**kw)
    per = wall / n
    res.append(dict(arm=name, **kw, wall_s=wall, n=n, s_per_sample=per,
                    gpu_h_per_1k=per * 1000 / 3600,
                    samples_per_gpu_h=3600 / per,
                    realtime_x=audio_s / wall,
                    stage_ms={k: v / n * 1000 for k, v in st.items()}))
    print(f"{name:22s} {per*1000:7.1f} ms/sample  {3600/per:7.0f} samples/GPU-h  "
          f"{per*1000/3600:.4f} GPU-h/1k  | " +
          "  ".join(f"{k} {v/n*1000:6.1f}" for k, v in st.items()), flush=True)

print(f"\nspeedup A->E: {res[0]['s_per_sample']/res[-1]['s_per_sample']:.2f}x")

# --- how much does best-of-N actually cost, end to end, in config E? ---
print("\n=== end-to-end cost vs N (config E), and packing ===", flush=True)
ncurve = []
best = dict(tf32=True, thread_mp3=True, sidon_out=False, cheap_rank=True)
for nc, pk in [(1, 32), (1, 8), (2, 16), (4, 16), (8, 8), (8, 16), (16, 4), (32, 2)]:
    try:
        wall, st, n = run(n_cand=nc, pack=pk, **best)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache(); print(f"  N={nc} pack={pk}: OOM"); continue
    per = wall / n
    ncurve.append(dict(n_cand=nc, pack=pk, in_flight=nc * pk, s_per_sample=per,
                       gpu_h_per_1k=per * 1000 / 3600, samples_per_gpu_h=3600 / per,
                       ms_per_candidate=st["gen"] / n / nc * 1000,
                       stage_ms={k: v / n * 1000 for k, v in st.items()}))
    print(f"  N={nc:3d} pack={pk:3d} ({nc*pk:3d} in flight)  {per*1000:7.1f} ms/sample  "
          f"{3600/per:7.0f} samples/GPU-h  {per*1000/3600:.4f} GPU-h/1k  "
          f"gen {st['gen']/n/nc*1000:6.1f} ms/cand", flush=True)

json.dump({"arms": res, "n_curve": ncurve},
          open(f"{NB}/vcbon/out/bench_pipeline.json", "w"), indent=2, default=float)
print(f"wrote {NB}/vcbon/out/bench_pipeline.json")
