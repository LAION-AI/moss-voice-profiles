"""Publish the Scenario-1 conversion corpus to the Hub.

NOT RUN. `$NB/.hf_token` was rotated and is dead, so this is written, committed
and left unpushed. When a token appears:

    python vchfpublish.py --token-file $NB/.hf_token --card-only     # dry run first
    python vchfpublish.py --token-file $NB/.hf_token

`--card-only` uploads the dataset card and the manifest without the ~15.6 TB of
audio, which is the sane first push: it makes the repo describable and lets the
card be reviewed before anything large moves.

The audio goes as the WebDataset tars exactly as written, because they are
already the shipping artefact. `upload_large_folder` is used rather than
`upload_folder` for the same reason the rest of this project uses it: it
resumes, and a 15.6 TB upload will be interrupted.
"""
import os, sys, json, glob, argparse, time

NB = "/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox"
PROD = f"{NB}/vcbon/prod"
VC = f"{NB}/vprof/vc500"
REPO = "laion/moss-voice-conversion-bestof4"


def build_card(idx, an, norm):
    th = (an or {}).get("throughput") or {}
    return f"""---
license: apache-2.0
task_categories:
- audio-to-audio
- text-to-speech
language:
- en
- de
tags:
- voice-conversion
- speech-restoration
- best-of-n
size_categories:
- 10M<n<100M
---

# MOSS voice conversion, best-of-4 with SIDON restoration

{idx.get('n_samples', 0):,} source takes across {idx.get('n_voices', 0)} voices,
each converted to its own voice's prepared reference **four times**, each
candidate restored with SIDON, and all four kept.

## What is here

For every source take there are four converted candidates. Nothing was selected
and thrown away: the selection is a *view* over stored scores, so a different
reward can be applied later without regenerating anything.

```
<voice>/VC1/vc-<NNN>.tar        4 candidates per source, 48 kHz mp3 (WebDataset)
<voice>/VC1/cand-<NNN>.parquet  every score for every candidate, no audio
<voice>/VC1/prov-<NNN>.parquet  provenance, one row per source take
<voice>/VC1/REF.json            the target reference and exactly how it was prepared
```

Sample key: `<target>/<source>/<gid>.c<NNN>.k<J>.mp3`.

**Join on `(source_run_dir, source_audio_key)`, not on `audio_key` alone** —
`audio_key` is `<gid>.c<NNN>.mp3` and is *not* unique across generation runs.
`prov` carries `source_run_dir` for exactly this reason.

## Pipeline

1. 4 voice-conversion candidates (ResembleAI chatterbox `s3gen`), TF32 enabled.
2. **SIDON restoration on all four**, TF32 disabled (TF32 inside SIDON costs
   quality; it also costs 1.39x throughput to leave it off, which is the price).
3. The four **SIDON-enhanced** candidates are ranked by
   `normalise(target emotion strength) + normalise(quality)`.
4. Everything stored.

Ranking happens after restoration so the ranker sees the audio that ships,
rather than selecting on a pre-enhancement signal.

## The reward

`reward_group` = frozen group z-score of target emotion strength + frozen group
z-score of Overall-Quality. Group = `(target kind, target name)` for the emotion
term (each emotion expert and each VoiceNet dimension is separately calibrated),
one global group for quality.

Two alternative rewards are stored alongside with their own winner columns, so
the choice is reversible without recomputation:

| column | normalisation |
|---|---|
| `reward_group` / `is_winner_group` | frozen group z-score (**production**) |
| `reward_set` / `is_winner_set` | z-score within the 4-candidate set |
| `reward_minmax` / `is_winner_minmax` | min-max within the 4-candidate set |

Within-set normalisation was rejected for production because it forces the two
terms to contribute equally in every set regardless of whether either actually
varies there, and at n=4 the sd estimate that drives it is ~40% noise.

## Per-candidate columns

All 40 EmoNet expert scores, all 4 quality heads, all 57 VoiceNet regressions,
speaker similarity under two independent embedders (ECAPA and Orange WavLM-tbr)
against two anchors (the prepared reference and the untouched original the
corpus generator used), blend, genuineness, and the source take's own stored
scores for before/after on the same instrument.

## Reference preparation

Every target reference was SIDON-restored and loudness-normalised to
**EBU R128 / ITU-R BS.1770 integrated −23 LUFS with a 0.95 peak ceiling**. The
ceiling matters: SIDON returns audio already peak-normalised to 0.9, so a louder
target makes the ceiling eat the gain and the normalisation silently does
nothing.

## Provenance

Sources are the `PPILOT2` corpus under `vprof/vp500/<voice>/`, untouched. This
release is additive.

Measured throughput: {th.get('s_per_sample_mean', 0):.3f} s per source sample,
{th.get('projected_gpu_h_full_run', 0):,.0f} GPU-h on NVIDIA GH200
(JUPITER Booster). Built {time.strftime('%Y-%m')}.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--token-file", default=f"{NB}/.hf_token")
    ap.add_argument("--card-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    idx = json.load(open(f"{PROD}/index/shards_summary.json"))
    an = None
    for p in (f"{PROD}/out/analysis.json", f"{PROD}/out/analysis_normpass.json"):
        if os.path.exists(p):
            an = json.load(open(p)); break
    norm = json.load(open(f"{PROD}/index/norm_stats.json")) if \
        os.path.exists(f"{PROD}/index/norm_stats.json") else {}

    card = build_card(idx, an, norm)
    os.makedirs(f"{PROD}/release", exist_ok=True)
    open(f"{PROD}/release/README.md", "w").write(card)

    tars = sorted(glob.glob(f"{VC}/*/VC1/vc-*.tar"))
    cands = sorted(glob.glob(f"{VC}/*/VC1/cand-*.parquet"))
    manifest = dict(repo=a.repo, built_at=time.time(),
                    n_voices=idx["n_voices"], n_shards_expected=idx["n_shards"],
                    n_samples=idx["n_samples"],
                    n_tars_present=len(tars), n_cand_parquets_present=len(cands),
                    bytes_present=int(sum(os.path.getsize(t) for t in tars)))
    json.dump(manifest, open(f"{PROD}/release/manifest.json", "w"), indent=2)
    print(json.dumps(manifest, indent=2))

    if a.dry_run:
        print("dry run: card and manifest written, nothing uploaded")
        return 0

    tok = None
    if os.path.exists(a.token_file):
        tok = open(a.token_file).read().strip()
    if not tok:
        print(f"no token at {a.token_file}: nothing pushed (this is the expected "
              f"state -- the token was rotated; commit and wait)", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi
    api = HfApi(token=tok)
    api.create_repo(a.repo, repo_type="dataset", exist_ok=True, private=False)
    api.upload_file(path_or_fileobj=f"{PROD}/release/README.md", path_in_repo="README.md",
                    repo_id=a.repo, repo_type="dataset")
    api.upload_file(path_or_fileobj=f"{PROD}/release/manifest.json",
                    path_in_repo="manifest.json", repo_id=a.repo, repo_type="dataset")
    api.upload_file(path_or_fileobj=f"{PROD}/index/norm_stats.json",
                    path_in_repo="norm_stats.json", repo_id=a.repo, repo_type="dataset")
    print("card + manifest + normalisation constants uploaded")
    if a.card_only:
        return 0
    api.upload_large_folder(folder_path=VC, repo_id=a.repo, repo_type="dataset",
                            num_workers=16)
    print("audio + scores uploaded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
