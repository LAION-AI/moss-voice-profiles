"""Why is SIDON costing 70 ms/clip when the pilot measured 38.3?

Two candidate explanations, and they have different consequences:

  (a) clip length. The cost model assumes a 9.72 s mean clip. SIDON cost is
      linear in duration, so a shard of longer takes costs proportionally more
      and nothing is wrong.
  (b) TF32. The pilot's throughput arms B/C ran with `tf32=true` set GLOBALLY,
      which means the 36-38 ms/clip figure was measured with TF32 active inside
      SIDON. This run disables TF32 for SIDON because the project measured that
      leaving it on costs ~15 dB. If that is the cause, then the 857 GPU-h SIDON
      budget and the "no TF32 on SIDON" quality rule are mutually inconsistent,
      and the budget is the one that has to move.

This measures both: SIDON wall time per clip at TF32 on vs off on identical
input, the duration scaling, and the actual numerical difference between the two
outputs in dB — so the quality claim is re-checked here rather than trusted.
"""
import os, sys, json, time, tarfile, argparse
import numpy as np, torch, torchaudio

NB = "/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox"
sys.path.insert(0, f"{NB}/vcbon/code")
import vcengine as E


def snr_db(a, b):
    n = min(len(a), len(b))
    a, b = np.asarray(a[:n], np.float64), np.asarray(b[:n], np.float64)
    d = a - b
    p, q = float((a ** 2).mean()), float((d ** 2).mean())
    return 10 * np.log10(p / q) if q > 0 else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="anime_000/001")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=f"{NB}/vcbon/prod/out/sidon_tf32.json")
    a = ap.parse_args()

    voice, sh = a.shard.split("/")
    import glob, pandas as pd
    metas = sorted(glob.glob(f"{NB}/vprof/vp500/{voice}/PPILOT2/meta-{int(sh):03d}-*.parquet"))
    mf = pd.concat([pd.read_parquet(m, columns=["audio_key", "dur"]) for m in metas],
                   ignore_index=True)
    mf = mf.iloc[:: max(1, len(mf) // a.n)].head(a.n)
    want = set(mf.audio_key)
    wavs, durs = [], []
    with tarfile.open(f"{NB}/vprof/vp500/{voice}/PPILOT2/cands-{int(sh):03d}.tar") as tf:
        for m in tf:
            if m.name in want:
                x, _ = E.decode_audio_bytes(tf.extractfile(m).read(), 24000)
                wavs.append(x); durs.append(len(x) / 24000)
    print(f"[bench] {len(wavs)} clips, mean {np.mean(durs):.2f} s "
          f"(cost model assumes 9.72 s)", flush=True)

    sd = E.Sidon("cuda:0", threads=16)

    def run(tf32):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        sd.restore(wavs[:8], 24000, max_frames=8000, max_items=a.batch)   # warm
        torch.cuda.synchronize()
        t0 = time.time()
        out = sd.restore(wavs, 24000, max_frames=8000, max_items=a.batch)
        torch.cuda.synchronize()
        dt = time.time() - t0
        return out, dt

    off, t_off = run(False)
    on, t_on = run(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    snrs = [snr_db(o, n) for o, n in zip(off, on)]
    mean_dur = float(np.mean(durs))
    res = dict(
        shard=a.shard, n_clips=len(wavs), mean_clip_s=mean_dur, batch=a.batch,
        tf32_off=dict(wall_s=t_off, ms_per_clip=t_off * 1000 / len(wavs),
                      ms_per_clip_at_972s=t_off * 1000 / len(wavs) * 9.72 / mean_dur,
                      realtime_x=sum(durs) / t_off),
        tf32_on=dict(wall_s=t_on, ms_per_clip=t_on * 1000 / len(wavs),
                     ms_per_clip_at_972s=t_on * 1000 / len(wavs) * 9.72 / mean_dur,
                     realtime_x=sum(durs) / t_on),
        speedup_tf32_on=t_off / t_on,
        snr_off_vs_on_db=dict(mean=float(np.mean(snrs)), median=float(np.median(snrs)),
                              p05=float(np.percentile(snrs, 5)),
                              p95=float(np.percentile(snrs, 95)),
                              min=float(np.min(snrs)), max=float(np.max(snrs))),
        pilot_claim_ms_per_clip=38.3,
        note="ms_per_clip_at_972s rescales to the cost model's assumed mean clip "
             "so the comparison against 38.3 ms is like-for-like on duration.")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
