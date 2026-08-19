"""Build the Scenario-1 shard manifest.

A shard is one (voice, tar-index) pair: exactly the unit the corpus was
generated in, so the source audio for a shard is one contiguous tar and the
source metadata is the 1-3 incremental `meta-<sh>-<ts>.parquet` files for it.
500 voices x 4 tars = 2000 shards. Nothing is copied or re-materialised here;
the manifest only records where each shard's inputs live and how big it is,
so the workers can claim work without a 20 M-row index in memory.
"""
import os, sys, glob, json, time
import pandas as pd, pyarrow.parquet as pq

NB = "/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox"
OUT = f"{NB}/vcbon/prod/index"
os.makedirs(OUT, exist_ok=True)

voices = json.load(open(f"{NB}/vprof/work/voices500.json"))["voices"]
refs = set()
import tarfile
with tarfile.open(f"{NB}/vcbon/refs500/refs_prepared.tar") as tf:
    refs = {os.path.basename(n).replace(".prep.mp3", "") for n in tf.getnames()}

rows, missing = [], []
t0 = time.time()
for vi, v in enumerate(voices):
    d = f"{NB}/vprof/vp500/{v}/PPILOT2"
    if not os.path.isdir(d):
        missing.append((v, "no PPILOT2")); continue
    if v not in refs:
        missing.append((v, "no prepared reference")); continue
    for sh in range(4):
        tar = f"{d}/cands-{sh:03d}.tar"
        metas = sorted(glob.glob(f"{d}/meta-{sh:03d}-*.parquet"))
        if not os.path.exists(tar) or not metas:
            missing.append((v, f"shard {sh}: tar={os.path.exists(tar)} metas={len(metas)}"))
            continue
        n = sum(pq.ParquetFile(m).metadata.num_rows for m in metas)
        rows.append(dict(shard_id=f"{v}/{sh:03d}", voice=v, sh=sh,
                         run_dir=d, tar=tar, n_meta=len(metas), n_samples=int(n),
                         tar_bytes=os.path.getsize(tar)))
    if (vi + 1) % 50 == 0:
        print(f"[index] {vi+1}/{len(voices)} voices  {len(rows)} shards  {time.time()-t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
df.to_parquet(f"{OUT}/shards.parquet", index=False)
summary = dict(n_voices=int(df.voice.nunique()), n_shards=len(df),
               n_samples=int(df.n_samples.sum()),
               samples_per_shard_mean=float(df.n_samples.mean()),
               samples_per_shard_min=int(df.n_samples.min()),
               samples_per_shard_max=int(df.n_samples.max()),
               source_bytes=int(df.tar_bytes.sum()),
               missing=missing, built_at=time.time())
json.dump(summary, open(f"{OUT}/shards_summary.json", "w"), indent=2)
print(json.dumps({k: v for k, v in summary.items() if k != "missing"}, indent=2))
print(f"missing: {len(missing)}")
for m in missing[:20]:
    print("  ", m)
