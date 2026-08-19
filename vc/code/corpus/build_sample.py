"""Build the pilot's source-take sample from the real vp500 corpus.

Selection is stratified so the pilot answers the questions that matter:
  * below-floor vs above-floor takes (spk_sim < 0.40 is the repair population)
  * every block (emotion / voicenet / edge / burst / character), both languages
  * the real duration distribution, because generation cost is length-linear

Everything lands in ONE tar + ONE parquet — no loose-file sprawl.
"""
import os, sys, io, json, glob, tarfile, time
import numpy as np, pandas as pd

NB = "/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox"
OUT = f"{NB}/vcbon/pilot"
VOICES = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    "anime_000", "emolia_c0019", "mediathek_0051", "k10_age3_bg1"]
PER_VOICE = int(sys.argv[2]) if len(sys.argv) > 2 else 96

os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20260816)


def load_meta(run):
    fs = sorted(glob.glob(f"{run}/meta-*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df = df.drop_duplicates(subset=["gid", "cand"], keep="last")
    return df


rows, members = [], []
for v in VOICES:
    run = f"{NB}/vprof/vp500/{v}/PPILOT2"
    if not os.path.isdir(run):
        print(f"[skip] {v} missing"); continue
    t0 = time.time()
    df = load_meta(run)
    df = df[(~df["empty"]) & (df["dur"] > 1.0) & (df["dur"] < 25.0)].copy()
    df["below"] = df["spk_sim"] < 0.40
    print(f"[{v}] {len(df)} usable takes, below-floor {df['below'].mean()*100:.1f}%, "
          f"mean dur {df['dur'].mean():.2f}s, load {time.time()-t0:.1f}s", flush=True)

    # stratify: block x below-floor, proportional, then fill
    picks = []
    for (blk, bel), g in df.groupby(["block", "below"]):
        n = max(1, int(round(PER_VOICE * len(g) / len(df))))
        picks.append(g.sample(min(n, len(g)), random_state=int(rng.integers(1 << 30))))
    sel = pd.concat(picks).drop_duplicates(subset=["gid", "cand"])
    if len(sel) > PER_VOICE:
        # keep the below/above balance closer to 50/50 than the corpus is, so
        # both arms of the comparison have enough n
        b = sel[sel.below]; a = sel[~sel.below]
        nb = min(len(b), PER_VOICE // 2); na = PER_VOICE - nb
        sel = pd.concat([b.sample(nb, random_state=1), a.sample(min(na, len(a)), random_state=1)])
    sel = sel.reset_index(drop=True)

    want = set(sel["audio_key"])
    found = {}
    for t in sorted(glob.glob(f"{run}/cands-*.tar")):
        with tarfile.open(t) as tf:
            for m in tf:
                if m.name in want and m.name not in found:
                    found[m.name] = tf.extractfile(m).read()
        if len(found) == len(want):
            break
    sel = sel[sel["audio_key"].isin(found)].reset_index(drop=True)
    print(f"[{v}] selected {len(sel)}  below {int(sel.below.sum())}  "
          f"blocks {sorted(sel.block.unique())}", flush=True)

    for _, r in sel.iterrows():
        key = f"{v}/{r['audio_key']}"
        members.append((key, found[r["audio_key"]]))
        rows.append({
            "src_voice": v, "run_dir": run, "audio_key": r["audio_key"], "key": key,
            "gid": r["gid"], "cand": int(r["cand"]), "block": r["block"], "lang": r["lang"],
            "emotion": r.get("emotion"), "condition": r.get("condition"),
            "cond_key": r.get("cond_key"), "edge": r.get("edge"),
            "dur": float(r["dur"]), "spk_sim_src": float(r["spk_sim"]),
            "below_floor_src": bool(r["below"]),
            "blend_src": float(r["blend"]), "strength_raw_src": float(r["strength_raw"]),
            "emo_strength_src": float(r["emo_strength"]),
            "genuineness_src": float(r["genuineness"]), "wer_src": float(r["wer"]),
            "dim_target_src": (float(r["dim_target"]) if pd.notna(r["dim_target"]) else None),
            "score_src": float(r["score"]), "rank_src": int(r["rank"]),
            "text": r.get("text"),
        })

with tarfile.open(f"{OUT}/sources.tar", "w") as tf:
    for name, raw in members:
        ti = tarfile.TarInfo(name); ti.size = len(raw); ti.mtime = 0
        tf.addfile(ti, io.BytesIO(raw))
mf = pd.DataFrame(rows)
mf.to_parquet(f"{OUT}/sources.parquet", index=False)
print(f"\n[sample] {len(mf)} takes -> {OUT}/sources.tar ({os.path.getsize(OUT+'/sources.tar')/1e6:.1f} MB)")
print(mf.groupby(["src_voice", "below_floor_src"]).size())
print(f"dur: mean {mf.dur.mean():.2f}s median {mf.dur.median():.2f}s "
      f"p10 {mf.dur.quantile(.1):.2f} p90 {mf.dur.quantile(.9):.2f}")
print(f"spk_sim_src: mean {mf.spk_sim_src.mean():.3f}  below-floor {mf.below_floor_src.mean()*100:.1f}%")
