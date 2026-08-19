"""Export a small set of listenable examples to the HF *dataset* repo.

The Space has a 1 GB quota it has already overflowed once, so audio never goes
there — it goes to TTS-AGI/moss-voice-pilot-audio under its own namespaced
prefix (`vcbon/`), because audio_key is not unique across runs and a previous
page served the wrong run's clips.
"""
import os, sys, io, json, glob, tarfile
import numpy as np, pandas as pd

NB = "/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox"
sys.path.insert(0, f"{NB}/vcbon/code")
import vcengine as E

RUN = f"{NB}/vcbon/pilot/vc_v1"
STAGE = f"{NB}/gh/audio_local/vcbon"
PREFIX = "vcbon"
os.makedirs(STAGE, exist_ok=True)

df = pd.read_parquet(f"{RUN}/candidates.parquet")
win = df[df.is_winner & df.sidon]
src = pd.read_parquet(f"{NB}/vcbon/pilot/sources.parquet").set_index("key")

self_ = win[win.arm == "self"].set_index("src_key")
picks = []
# 1. below-floor takes that conversion rescued the hardest
g = self_[self_.below_floor_src].copy()
g["gain"] = g.ecapa_vc_origref - g.spk_sim_src
picks += [("rescued", k) for k in g.nlargest(4, "gain").index]
# 2. emotion-block takes: where the strength cost shows
ge = self_[(self_.target_kind == "emo") & self_.emo_strength_src.gt(1.5)]
picks += [("emotion", k) for k in ge.nlargest(4, "emo_strength_src").index]
# 3. takes conversion did NOT rescue
gb = self_[self_.ecapa_vc_origref < 0.40]
picks += [("still_below", k) for k in gb.nsmallest(2, "ecapa_vc_origref").index]
# 4. a burst / edge case
gx = self_[self_.block.isin(["edge", "burst_isolated"])]
picks += [("edge", k) for k in gx.nlargest(2, "blend_vc").index]

keys = []
seen = set()
for tag, k in picks:
    if k in seen:
        continue
    seen.add(k); keys.append((tag, k))
print(f"[export] {len(keys)} examples")

# original corpus takes
srcbytes = {}
with tarfile.open(f"{NB}/vcbon/pilot/sources.tar") as tf:
    for m in tf:
        if m.name in seen:
            srcbytes[m.name] = tf.extractfile(m).read()

# converted takes, per arm
conv = {}
for arm in ("self", "nn", "far"):
    with tarfile.open(f"{RUN}/converted-{arm}.tar") as tf:
        for m in tf:
            if not m.name.endswith(".mp3"):
                continue
            base = os.path.basename(m.name).rsplit(".k", 1)[0] + ".mp3"
            voice = m.name.split("/")[1].split("__")[0]
            conv[(arm, f"{voice}/{base}")] = tf.extractfile(m).read()

rows = []
for tag, k in keys:
    stem = k.replace("/", "__").replace(".mp3", "")
    open(f"{STAGE}/{stem}__source.mp3", "wb").write(srcbytes[k])
    r = self_.loc[k]
    rec = dict(tag=tag, key=k, stem=stem, block=r["block"], gid=r["gid"],
               emotion=r["emotion"], lang=r["lang"], dur=float(r["dur"]),
               spk_sim_src=float(r["spk_sim_src"]),
               emo_strength_src=float(r["emo_strength_src"]),
               blend_src=float(r["blend_src"]), text=src.loc[k, "text"])
    for arm in ("self", "nn", "far"):
        b = conv.get((arm, k))
        if b is None:
            continue
        open(f"{STAGE}/{stem}__{arm}.mp3", "wb").write(b)
        w = win[(win.arm == arm) & (win.src_key == k)]
        if len(w):
            w = w.iloc[0]
            rec[f"{arm}_ecapa"] = float(w["ecapa_vc_origref"])
            rec[f"{arm}_wavlm"] = float(w["wavlm_vc_origref"])
            rec[f"{arm}_strength"] = float(w["strength_vc"]) if pd.notna(w["strength_vc"]) else None
            rec[f"{arm}_blend"] = float(w["blend_vc"])
            rec[f"{arm}_q"] = float(w["overall_q_vc"])
            rec[f"{arm}_target"] = w["target_voice"]
    rows.append(rec)

# the prepared reference for each voice in play
with tarfile.open(f"{NB}/vcbon/refs500/refs_prepared.tar") as tf:
    have = {os.path.basename(m.name).replace(".prep.mp3", ""): m for m in tf.getmembers()}
    voices = sorted({r["key"].split("/")[0] for r in rows} |
                    {r.get(f"{a}_target") for r in rows for a in ("self", "nn", "far")} - {None})
    for v in voices:
        if v in have:
            open(f"{STAGE}/ref__{v}.mp3", "wb").write(tf.extractfile(have[v]).read())

json.dump(rows, open(f"{NB}/vcbon/out/examples.json", "w"), indent=2, default=float)
n = len(os.listdir(STAGE))
sz = sum(os.path.getsize(f"{STAGE}/{f}") for f in os.listdir(STAGE))
print(f"[export] staged {n} files, {sz/1e6:.1f} MB in {STAGE}")

if "--upload" in sys.argv:
    from huggingface_hub import HfApi
    api = HfApi(token=open(f"{NB}/.hf_token").read().strip())
    api.upload_folder(folder_path=STAGE, path_in_repo=PREFIX,
                      repo_id="TTS-AGI/moss-voice-pilot-audio", repo_type="dataset",
                      commit_message="Best-of-N voice conversion pilot: examples")
    print("[export] uploaded to TTS-AGI/moss-voice-pilot-audio/" + PREFIX)
