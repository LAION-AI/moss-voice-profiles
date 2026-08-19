# INDEX — every voice-conversion artefact, where it lives, what it is for

`NB = /e/data1/datasets/playground/mmlaion/schuhmann1/dramabox`
`REPO = $NB/gh/moss-voice-profiles` → https://projects.laion.ai/moss-voice-profiles/

Read [`README.md`](README.md) first. This file is the map.

**Copy policy.** Code and small result files (< 100 kB each) were **copied** into this
folder so the repo is self-contained and browsable on the web — total 418 kB, 47 files.
Audio, tars, npz caches, parquet candidate tables and logs were **not** copied and are
referenced by absolute path instead: they run from 62 MB to 11 TB and do not belong in a
GitHub Pages repo. Nothing was moved or deleted; every original is still in place.

---

## In this folder

| path | what it is |
|---|---|
| `README.md` | the main document — what VC is for, both variants side by side with measurements, defects, reproduction |
| `INDEX.md` | this file |

### `code/engine/` — shared by both variants

| file | what it is |
|---|---|
| `vcengine.py` | the whole engine: Chatterbox VC load/target/tokenize/batched generate, SIDON wrapper, ECAPA + WavLM speaker sim, BUD-E scorer, LUFS, mp3 encode/decode. Copied from `$NB/vcbon/code/vcengine.py` |

### `code/collage/` — the four-arm comparison, i.e. **both collage variants**

Copied from `$NB/vc4arm/code/`.

| file | what it is |
|---|---|
| `select20.py` | picks the 20 collages: 6 mirror emotion pairs in both directions + 8 region-stratified fills → `$NB/vc4arm/work/selection.json` |
| `prep.py` | source preparation. **Holds `collapse_internal`, the internal-silence fix** (>400 ms → 150 ms, 40 ms equal-power crossfade). Writes the silence audit |
| `arms.py` | builds all five arms. `vc_whole` = variant B (one pass over the finished soundscape). `vc_clips` + `Sidon.restore` = variant A (per clip, then restore, then concatenate) |
| `measure.py` | per-segment measurement: ECAPA/WavLM against **both** anchors, target emotion z, quality, speaker spread, seam jump with an interior control, gap levels |
| `run.sbatch` | 1 node / 1 GPU / 72 CPU / 400 GB / 4 h, `booster`, account `reformo`; does the mandatory serial warm-up import first |

### `code/corpus/` — the corpus-scale best-of-N pipeline

Copied from `$NB/vcbon/code/`.

| file | what it is |
|---|---|
| `prep_refs.py` | builds the 500 prepared references (`decode → 16 k → SIDON → −23 LUFS → 0.95 ceiling`) |
| `build_sample.py` | assembles the 354-take pilot sample and its three target arms (`self` / `nn` / `far`) |
| `pilot.py`, `pilot.sbatch` | the one-GPU pilot: 8 candidates per take, all three arms |
| `analyse.py` | pilot analysis → `data/vcbon_pilot_analysis.json` |
| `ref_ablate.py` | the five-arm reference-preparation ablation + the TF32-on-generation check |
| `bench_gen.py`, `bench_sidon.py`, `bench_pipeline.py` | throughput sweeps: N, pack size M, length bucketing, dtype; SIDON alone; end-to-end |
| `costmodel.py` | the cost model the production plan was built from |
| `smoke.py`, `export_audio.py` | smoke test; audio export for the published page |
| `prod.py`, `prod.sbatch` | **the voice-claiming production implementation** — this is what produced `run1`. See README §5.4 before using it |

### `code/corpus/shard/` — the shard-based production implementation (**use this one**)

Copied from `$NB/vcbon/prod/code/`.

| file | what it is |
|---|---|
| `vcindex.py` | builds the shard manifest from parquet metadata only → 2,000 shards, 20,125,736 samples |
| `vcprod.py` | one shard end to end: decode → generate 4 → SIDON 4 → score 4 → reward → store. Five verification gates, per-pack fault isolation, the tokenizer and short-clip fixes |
| `vcrun.py` | per-GPU claim loop (`O_CREAT\|O_EXCL` claims on the shared filesystem) |
| `vcsweep.py`, `vcsweepd.sh` | the sweeper and its restart wrapper: holds the repair gate, refuses to submit without frozen normalisation constants, reaps dead claims, keeps 48 nodes alive |
| `vcgate.py` | polls the identity-repair state every 300 s and releases the run |
| `vcnorm.py` | freezes the reward normalisation constants, with a robustness check |
| `vcanalyse.py` | the ordering A/B, emotion recovery and throughput, all as parquet queries |
| `vcsidonbench.py` | SIDON TF32 on/off on identical clips, including the output difference in dB |
| `vcreport.py` | builds `../vc_scenario1.html` |
| `vchfpublish.py` | Hugging Face release packaging (**not run by this task — that Space is a mirror**) |
| `vcworker.sbatch`, `vcsmoke.sbatch`, `vcprodsmoke.sbatch`, `vcnormpass.sbatch`, `vcbench.sbatch` | 4 GPUs/node, `--cpus-per-task=64`, warm-up import first |

### `docs/`

| file | what it is |
|---|---|
| `FOURARM_STATE.md` | **authoritative** for the four-arm study's decisions: arm 1 rebuilt not reused, the 20 collages, the silence fix, the clip recipe, each arm's reference, the TF32 policy. Also records the two stages that were never done |
| `PROTOCOL.md` | the corpus run's append-only record, 758 lines: every decision, every measurement, every bug, and one correction appended rather than edited away |

### `data/` — the evidence behind the README's numbers

| file | what it is |
|---|---|
| `fourarm_arm_means.csv` | mean of every metric by arm (n = 20 collages each) |
| `fourarm_arm_sd.csv` | the same, standard deviations |
| `fourarm_paired_vs_arm1.csv` | paired difference of each arm against the raw concatenation: mean, sd, wins/20, t |
| `fourarm_arm3_vs_arm2.csv` | **the head-to-head**: variant A minus variant B, per metric |
| `fourarm_seed_noise_arm2b_vs_arm2.csv` | the run-to-run noise floor — arm 2 re-run with a different seed |
| `fourarm_silence_audit.json` | the internal-silence defect quantified: 43/96 clips, 56 stretches, 23.62 s, 0 surviving the fix |
| `fourarm_metrics.parquet` | 100 rows (20 collages × 5 arms) × 23 columns — the raw table all the above derive from |
| `fourarm_segments.parquet` | 480 per-segment rows, for anyone who wants to re-cut the aggregation |
| `fourarm_selection.json` | which 20 collages were used |
| `vcbon_pilot_analysis.json` | the pilot: identity gains, best-of-k curves, SIDON effect, target-dimension loss, per-block breakdown, target-match quality |
| `vcbon_ref_ablate.json` | the five-arm reference-preparation ablation and the TF32-on-generation check |
| `vcbon_sidon_tf32.json` | SIDON with TF32 on vs off: ms/clip and output difference in dB |
| `vcbon_normpass_analysis.json` | the 17-shard normalisation pass: reward comparison over 34,000 sources, the ordering A/B |
| `vcbon_prod_analysis.json` | one complete full-size production shard: reward recovery, stage timings, projected cost |
| `vcbon_done_voices.json` | the 346 voices that had a verified identity repair when `run1` was launched |

---

## Not in this folder — referenced by absolute path

### The four-arm study (`$NB/vc4arm/`)

| path | what it is | size |
|---|---|---|
| `out/arm1/*.mp3` | 20 × raw concatenation of the silence-fixed clips — the control | |
| `out/arm2/*.mp3` | 20 × **variant B**: one whole-file VC pass, target = the collage's first snippet | |
| `out/arm2b/*.mp3` | 20 × variant B again with a different seed — **the noise floor**, keep it | |
| `out/arm3/*.mp3` | 20 × **variant A**: per-clip VC → SIDON → concatenate, target = the voice's own prepared reference | |
| `out/arm4/*.mp3` | 20 × variant A followed by variant B | |
| `out/wav/*.npz` | all five arms plus the reference snippet at 48 kHz float — what `measure.py` reads | |
| `out/arms.json` | per-collage record: durations, segment bounds, length ratios, every loudness decision | 56 kB |
| `work/clips/*.npz` | the silence-fixed source clips every arm was built from | 150 MB |
| `work/prep.json` | the full per-clip preparation record including the raw silence audit | 96 kB |
| `logs/*` | the three job logs; `vc4arms-1399769.out` carries the stage timings used for the cost table | |

**These 100 mp3s have never been published.** There is no `vc4arm.html`. See README §5.2.

### The corpus pipeline (`$NB/vcbon/`)

| path | what it is | size |
|---|---|---|
| `refs500/refs_prepared.tar` | the 500 prepared conversion targets — **the input variant A depends on** | 88 MB |
| `refs500/ref_prep.parquet`, `prep_config.json` | how they were made, and the audit | |
| `pilot/sources.tar`, `pilot/sources.parquet` | the 354-take pilot sample | 69 MB |
| `pilot/vc_v1/converted-{self,nn,far}.tar` | the pilot's converted audio, three target arms | 3 × 70 MB |
| `pilot/vc_v1/candidates.parquet` | every score for every one of the 8,496 pilot candidates | 683 kB |
| `prod/index/shards.parquet` | the 2,000-shard manifest, verified to the unit against the spec's 20,125,736 | 67 kB |
| `prod/index/norm_stats.json` | **the frozen reward normalisation constants** — 158 `(kind, name, sign)` groups. Without this the run silently falls back to within-set ranking | 62 kB |
| `prod/state/sweep.json` | live run state. At time of writing: `ready 2000, done 3, todo 1997, repair_gate_open false` | |
| `prod/state/sweeper.conf.json` | the two controls: `enabled`, `wait_for_repair` | |
| `prod/smoke*/`, `prod/out/` | the smoke shards, the normalisation pass output and its analyses | 350 MB |
| `prod/release/README.md`, `manifest.json` | the drafted Hugging Face dataset card for `laion/moss-voice-conversion-bestof4`. Nothing was uploaded | |
| `prod/run1/voices/*/cands.tar` | **11.15 TB of unverified partial output from the abandoned voice-claiming run.** 152 non-empty tars, 0 completed voices. Uncompressed WAV. See README §5.4 | 11.15 TB |
| `prod/run1/state/*.claim` | 192 claims, 0 `.DONE` | |
| `logs/`, `prod/logs/` | worker logs and the gate poller | 73 MB |
| `ecapa_ckpt/` | symlinks to the ECAPA checkpoint in the shared HF cache | |
| `done_voices.json` | 346 voices — also copied to `data/` | |

### The trajectory collages

| path | what it is |
|---|---|
| `$NB/collage_traj/build.py` | the original collage builder — **head/tail trim only, no internal-silence handling. This is the defect's origin.** README §5.1 |
| `$NB/collage_traj/mp3/*.mp3` | 100 collages, 62 MB. **Carry the internal-silence defect. Retained deliberately as the diagnostic — do not delete, do not use as current output** |
| `$NB/collage_traj/collages.json` | the collage plan the four-arm study reads |
| `$NB/collage_traj/emotion_norm_constants.json` | z constants for the target-emotion metric |
| `$REPO/traj_audio/C*.mp3` | byte-identical copies of the above, served by `../emotion_trajectories.html` (verified: `C000.mp3` md5 `0126249d2b04c310b6b3712b06ff4fca` in both places) |
| `$NB/collage_traj2/build2.py` | the successor builder — **carries the fix** (400 ms → 150 ms, 20 ms crossfade) |

### The predecessor VC work (`$NB/vprof/code/`)

| path | what it is |
|---|---|
| `vcprep.py` | stages the top-1 take of every published condition group of one voice; matches on `(source_dir, audio_key)` because `audio_key` alone is not unique across runs → `vprof/work/vc_winners.parquet`, `vprof/work/vc_src/` (832 clips) |
| `vcbest.py`, `vcbest.sbatch` | best-of-3 seeds, one voice, no SIDON; winner = highest EmoNet-vector cosine to the original take, against a `cos_self` noise floor; MD5s every seed to prove the seeds actually differ → `$NB/gh/moss-voice-profiles-vc/vprof_audio_vc/` (3,328 files), `vprof/work/vc_results_shard*.parquet` |
| `vpvc.py`, `vcjob.sbatch` | post-hoc identity repair below `spk_sim < 0.45`; keep rule `sim_after > sim_before AND dnsmos_after ≥ dnsmos_before − 0.15`. Superseded; **known defect: no peak guard, so DNSMOS returned silent NaN for every converted clip** |
| `pvverify.py` | release verification — decodes every sampled mp3 and cross-checks duration, identity and every numeric field against the parquet; exits non-zero on failure |
| `vcbest_login.sh`, `vcbest_extra.sh` | login-node fallbacks for `vcbest.py` when `booster` was drained |
| `$NB/code/chatterbox-voice-conversion/expressive_bestofn/` | the method generalised and published, with `METHOD.md` and a live demo. Recommends N = 8 and rank-then-SIDON-the-winner |

### The identity-repair sweep (`$NB/vprof/repair/`) — the corpus run's gate

Not voice conversion, but the corpus VC waits on it and its outputs are VC's inputs.

| path | what it is |
|---|---|
| `code/MANUAL.md` | the narrative doc for the sweep |
| `state/sweep.json` | live state. At time of writing: `ready 499, done 426, todo 73, workers_running 35` |
| `report/aggregate.json`, `report/index.html` | the results: above-floor fraction 0.245 % → 74.99 % over 426 voices |
| `voices/<voice>/` | per voice: rank-4 identity LoRA, repaired takes, `report.json`, `VREPAIR.DONE` |

**Still running at time of writing — 37 jobs queued. Do not cancel.**

### Published pages in `$REPO`

| page | relation to this folder |
|---|---|
| `vc_bestofn.html` (+ `audio_vcbon/`, 53 files) | the pilot: best-of-N curves, both target scenarios, costs |
| `vc_scenario1.html` | the corpus run's design and measured throughput |
| `acting_concats.html` (+ `audio_concat/`, 297 files) | 9 scenes × 8 assemblies, four columns including `_vc.mp3` (variant A, per part) and `_vcwhole.mp3` (variant B) — **the listening comparison of the two variants**, and where the dispute originated |
| `emotion_trajectories.html` (+ `traj_audio/`, 276 files) | the trajectory collages — **audio carries the §5.1 defect** |
| `trajectories.html` (+ `audio_traj/`, 120 files) | VoiceNet ladders and emotion arcs |
| `$NB/gh/hfspace_vp/` | a Hugging Face Space mirror holding copies of several of these pages. **Mirror only — do not push to it** |

---

## The one-line version

- Want to **hear** the two variants: `$NB/vc4arm/out/arm2/` (whole-file) vs
  `$NB/vc4arm/out/arm3/` (per-clip), same 20 collage ids, or the `_vcwhole` / `_vc` columns
  of `../acting_concats.html` which are already online.
- Want the **numbers**: `data/fourarm_arm3_vs_arm2.csv`.
- Want to know **why there are two**: README §3.
- Want to know **what is broken**: README §5.
