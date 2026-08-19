# Scenario 1 — best-of-N voice conversion, production run

Append-only. Never rewrite history; corrections are appended as new entries.

**Spec (owner):** for each of the 20,125,736 source samples across 500 voices in
`vprof/vp500/<voice>/PPILOT2/` — 4 VC candidates, SIDON on all four, rank the
SIDON-enhanced candidates by `normalise(target emotion strength) + normalise(quality)`,
store everything (all four outputs, all raw scores, all rewards, all components).
Selection must be a *view* over stored scores.

**Budget plan:** 3,588 GPU-h = 2,731 base + 857 SIDON ≈ 258,300 core-h ≈ 18.7 h on 48 nodes.

---

## 2026-08-17 01:10 — Gate check, and why the run has not started

`vprof/repair/state/sweep.json`: `ready=499 done=101 todo=398 claimed=88
workers_running=48 refused=0`. The identity-repair sweeper is live and making
progress. Per the brief the production launch waits for `done == ready`.

Poller started (`vcbon/logs/gatepoll.jsonl`, 300 s cadence) so the wait is
measured rather than guessed, and so "the sweeper stopped making progress" is a
statement I can evidence rather than assert.

**Verified the gate is about node contention, not data.** The repair pipeline
writes to `vprof/repair/voices/<voice>/data/`; the VC sources are
`vprof/vp500/<voice>/PPILOT2/`, which repair does not touch. So there is no
correctness dependency — nothing about the VC output changes depending on
whether repair has finished. The gate is purely "do not fight for nodes".

Partition state at gate check: 2,205 idle nodes, 3,355 alloc. The repair
sweeper's own config (`state/sweeper.conf.json`) sets `idle_floor: 200` — it
declines to submit whenever the partition has fewer than 200 idle nodes. So the
sweeper is self-protecting, and 48 nodes taken out of 2,205 idle cannot starve
it.

**Decision:** honour the gate for the *bulk* launch (the 2,000-shard array).
Do the CPU-side preparation now (indexing, code, verification), and run the
end-to-end smoke test on 1-2 nodes before the gate clears — the brief requires a
smoke test *before* committing the bulk, and a 1-node smoke against 2,205 idle
nodes is not competition by any reading. Recorded here rather than assumed.

## 2026-08-17 01:20 — Owner clarification received (supersedes part of the brief)

Pipeline order is **decided, not open**: generate 4 → SIDON all 4 → rank the
SIDON-enhanced candidates → store everything. The earlier instruction to *halt*
the run if SIDON-before-ranking measured harmful is **withdrawn**. The A/B
(rank-then-SIDON vs SIDON-then-rank on the same takes) still runs on the first
shard, but as diagnostics only: it is recorded and reported, and it gates
nothing.

Rationale, recorded because it is the thing a later reader will want: the
pilot's "SIDON after selection is harmful" result (−0.056 Overall-Q, −0.041
ECAPA) was measured on a *different pipeline* — select first, enhance the
winner. Here the ranker scores exactly the audio that ships. The pilot number
does not transfer, and the comparison is informative rather than decisive.

## 2026-08-17 01:15 — Corpus reconnaissance (measured, not assumed)

- `vp500/<voice>/PPILOT2/` holds 4 source tars (`cands-000..003.tar`) and 1-3
  *incremental* `meta-<sh>-<ts>.parquet` per tar. Checked `anime_000`:
  40,256 rows across 12 meta files, **40,256 unique `audio_key`** — the meta
  files are incremental, not cumulative, so they concatenate without dedup.
  40,256 x 500 = 20,128,000, which brackets the spec's 20,125,736 (voices vary
  slightly). Confirms the sharding unit.
- Source `audio_key` is `<gid>.c<NNN>.mp3` and the tar member name is exactly
  that key. Mean clip 10.7 s, ~160 kbps mp3, ~1.9 GB per tar.
- All 500 voices have a prepared reference in `vcbon/refs500/refs_prepared.tar`
  (SIDON + EBU R128 −23 LUFS, ceiling 0.95). Verified count = 500.

**Shard unit chosen: (voice, tar index) = 2,000 shards of ~10,064 samples.**
Reasons: it is the unit the corpus was written in, so a shard reads exactly one
contiguous tar (no cross-tar seeking); ~10 k samples is ~1.36 GPU-h at the
pilot's measured 0.488 s/sample for N=4, which is a sane resume granularity for
an 18 h run where node failures are certain; and 2,000 shards over 192 GPUs is
10.4 waves, enough to smooth stragglers without a 100 k-file explosion.

**No pre-materialised source index.** The workers read the shard's own meta
parquets directly. A 20 M-row index would be a second copy of data that already
exists, and copies drift.

## 2026-08-17 01:25 — Discrepancy found in the storage estimate (flagging early)

The brief carries "Storage delta ~3.9 TB". That figure is
`costmodel.json -> storage[scenario1].winner_audio_TB` — the **winner-only**
number. The owner's spec stores *all four* converted outputs, and the same cost
model's `all8_audio_TB` is 31.30 TB, i.e. 4 candidates ≈ **15.65 TB**, four
times the quoted delta.

This is not a blocker: `/e/data1` has 14 P available. But 3.9 TB was the wrong
number to plan against and it is recorded here so nobody re-derives it later.
Budget check: `jutil` reports 60.6 M core-h remaining (not the cost model's
106 M nominal), so 258,300 core-h is **0.43 % of what is left**, not 0.24 %.

## 2026-08-17 01:20-01:35 — What was built, and the choices inside it

`vcbon/prod/code/`:

| file | what it is |
|---|---|
| `vcindex.py` | builds the shard manifest by reading parquet *metadata* only |
| `vcprod.py` | one shard: decode -> generate 4 -> SIDON 4 -> score 4 -> reward -> store |
| `vcrun.py` | per-GPU claim loop (O_CREAT\|O_EXCL claims on the shared FS) |
| `vcsweep.py` | reaps dead claims, keeps 48 nodes alive, publishes `state/sweep.json` |
| `vcnorm.py` | freezes the reward normalisation constants, with a robustness check |
| `vcanalyse.py` | the A/B, the emotion-recovery measurement, throughput — all as parquet queries |
| `vcworker.sbatch` / `vcsmoke.sbatch` | 4 GPUs/node, `--cpus-per-task=64`, warm-up import first |

**Index built and verified: 2,000 shards, 500 voices, `n_samples = 20,125,736` —
exactly the spec's figure, 0 shards missing a tar, a meta file, or a prepared
reference.** That the independently-derived count lands on the spec's number to
the unit is the strongest available check that the sharding did not silently
drop or double-count anything.

**Resumability.** A shard writes `.vc-NNN.tar.partial` and `.parquet.partial`,
then verifies (tar readable; member count == 4 x sources; the parquet's
`out_key` set equals the tar's member set exactly; no duplicate
`(run_dir, audio_key, cand_idx)`; exactly one `is_winner_group` per source), and
only then does it `os.replace` into place and write `done-NNN.json`. A killed
node leaves `.partial` files that nothing reads and a claim the sweeper reaps
once Slurm confirms the job is gone. Restart is always safe because the marker
is the *last* thing written and it is written only after verification.

**Ordering.** Workers claim the biggest shards first. The tail of a 10-wave run
is decided by stragglers, and a straggler you start at hour 1 is free.

**TF32 handling is explicit at the call site**, not set once at startup: on
around `s3gen.inference` (measured free, 1.60x), off around SIDON (measured
~15 dB). Setting it globally would have quietly cost the SIDON quality this
project already paid to measure.

**Storage keys.** `<target>/<source>/<gid>.c<NNN>.k<J>.mp3`, and `prov` carries
`source_run_dir` — `audio_key` is not unique across runs, and this project has
already been bitten once by exactly that (an audio-export page served the wrong
run's audio because `audio_key` was treated as a key).

## 2026-08-17 01:32 — Reward normalisation: the choice, and why (recorded, not assumed)

The spec asks for `normalise(target emotion strength) + normalise(quality)` with
both normalised before summing, and asks that the choice be stated.

**Chosen: z-score against frozen GROUP statistics.** Not within-candidate-set.

Within-set normalisation (z or min-max over the 4 candidates) forces the two
terms to contribute *equally in every set*, whatever the set actually looks
like. If four candidates differ in Overall-Quality by 0.002 — i.e. they are
tied and the difference is scorer noise — but differ in target emotion by 1.4,
within-set normalisation inflates that quality noise to the same magnitude as
the real emotion signal and lets it cast half the vote. With n=4 the sd estimate
carries ~40 % relative error, so the effective weighting between the two terms
is re-randomised for every sample. min-max is worse again: it is defined by the
two most extreme — i.e. noisiest — order statistics, saturates winner at 1.0 and
loser at 0.0 regardless of spread, and is degenerate on ties.

Frozen group z-score divides each term by the spread that term has *across the
corpus*. Neither term can dominate by scale — which is exactly what the spec
asks for — and a term that is flat within a given candidate set correctly
contributes almost nothing to that set's decision, so the choice is made on the
axis that actually varies.

**Grouping.** The emotion term is z-scored per `(target kind, target name)`. The
corpus conditions on two different things: an emotion label (an emonet expert)
or a signed VoiceNet dimension. The 40 emotion experts are separately calibrated
with different ranges, so one global z-score would weight "Anger" against "Awe"
by their scale rather than their meaning. Quality is one head on one scale and
gets one global group.

**Constants are frozen for the whole run**, estimated from the smoke shards.
A reward whose normalisation drifts shard to shard is not comparable across the
corpus, which would defeat the point of computing it.

**This is stored as a view, not a commitment.** Every row carries
`emo_target_raw`, `qual_raw`, all 40 emonet scores, all 4 quality heads, all 57
VoiceNet regressions, both speaker similarities against both anchors, and *three*
rewards side by side — `reward_group` (production), `reward_set` (within-set z,
the pilot's), `reward_minmax` — each with its own `is_winner_*` column. Changing
the reward later is a parquet query. The brief notes this property has already
paid off twice; it is preserved deliberately and at some storage cost.

## 2026-08-17 01:40 — Measuring the gate rather than guessing at it

`VREPAIR.DONE` marker timestamps give the repair run's true completion rate:

```
08-16 10h  18    08-16 17h   6    08-16 21h  11
08-16 14h   9    08-16 19h   5    08-16 22h   9
08-16 15h  13    08-16 20h  10    (last completion 22:59)
08-16 16h  20
```

~9-11 voices/h in the steady state. With 398 outstanding that projects to
**roughly 29-40 h**, i.e. the gate is unlikely to clear before late on
2026-08-18. Adding the 18.7 h run, end-to-end is ~48 h from now.

At 01:40 the newest completion was 22:59, 2.7 h earlier, which looks like a
stall until you look at the worker cohort: all 48 repair workers have elapsed
times of 2:08-3:38, i.e. they are mid-voice, and a voice takes ~3-4 h
(train -> generate -> pick -> finish). Completions therefore arrive in cohort
bursts, not smoothly. **This is a slow run, not a stuck one**, so the brief's
alternative release condition ("the remainder are genuinely stuck") does not
apply and the gate holds.

**Finding worth surfacing: the gate is not actually protecting anything.**
The repair sweeper is limited by its *own* `max_workers: 48`, not by node
availability — the partition has had 1,727-2,205 idle nodes throughout, and its
`idle_floor` of 200 has never bound. It will not submit a 49th worker no matter
how empty the machine is. So 48 VC nodes running alongside it cannot slow it
down by a single voice; the ~29 h wait buys the repair run nothing measurable.

I am honouring the gate anyway, because the brief states it as a condition and
the release clause it offers ("genuinely stuck") is not met — but this is
recorded so the cost of the decision is visible and can be overridden knowingly
rather than by accident. Flipping `enabled: true` in
`vcbon/prod/state/sweeper.conf.json` releases the run at any time.

**The wait is automated, not manual.** `vcgate.py` polls the repair state every
300 s, appends every observation to `state/gate_history.jsonl`, and on release
(done == ready, or a *demonstrated* stall: no completion for 4 h AND zero
repair workers in the queue) enables the VC sweeper, starts it, and appends the
release to this file. Nothing depends on someone watching a terminal at 06:00.

Preparation continues meanwhile: the smoke test and the normalisation pass
together use ~5 nodes for well under an hour, against 1,700+ idle.

## 2026-08-17 01:34 — Smoke test 1: end to end on 320 sources. PASSED, but slower than plan

`vcsmoke1` job 1392951, `anime_000/000`, `--limit 320 --ab-presidon 1`, one GH200.

**Correctness — all five verification gates passed:**
`tar_members == 4 x sources == 1280`; the parquet's `out_key` set equals the tar
member set exactly; no duplicate `(run_dir, audio_key, cand_idx)`; exactly one
`is_winner_group` per source; 156 columns written (40 emonet + 4 quality heads +
57 VoiceNet regressions + both speaker anchors + all three rewards + the
pre-SIDON diagnostic block). Keys are self-describing and correct:
`anime_000/anime_000/anime_000__B__kissing_noises__en.c007.k0.mp3`.
ECAPA to the prepared reference 0.760, to the original reference 0.665 — both
comfortably above the 0.40 floor, and consistent with the pilot's `self` arm.

**Throughput — materially off plan, reporting before spending the bulk.**

```
stage            s/sample     plan (cost model n_cand=4)
generate           0.633        0.412
SIDON (x4)         0.282        0.153   (38.3 ms/clip x 4)
score              0.067        0.027
speaker            0.034        0.003
mp3 (x4, 48 kHz)   0.037        0.027   <- the deferred drain works
tokenize + io      0.022        --
-------------------------------------
total (production) 1.076        0.642
(+ pre-SIDON A/B)  0.103        diagnostic only
```

1.076 s/sample projects to **6,014 GPU-h, 1.68x the 3,588 GPU-h plan.**

**Two confounds, being separated before anyone acts on that number:**

1. *This shard is not representative.* `--limit 320` took `head(320)` of the
   meta, and the meta is ordered by gid — so all 320 takes were the
   `burst_isolated` block, `target_kind = free`, mean duration **12.87 s against
   the cost model's assumed 9.72 s**. Both generate and SIDON scale with
   duration. Rescaling to a 9.72 s mean gives 0.813 s/sample -> 4,542 GPU-h,
   1.27x plan. Fixed: `--limit` now takes a *stride* sample, not a head, so the
   next measurement spans blocks, languages and durations in their real
   proportions.

2. *SIDON measured 70.4 ms/clip against the brief's 38.3 ms* (53.2 ms after
   rescaling for duration — still 1.39x). The likely cause is that the pilot's
   38.3 ms was measured with `tf32=true` set **globally** (cost-model arms B/C),
   i.e. with TF32 active inside SIDON — while this run disables TF32 for SIDON
   because the project measured that leaving it on costs ~15 dB. If that is the
   cause, **the 857 GPU-h SIDON budget and the "never TF32 on SIDON" quality
   rule are mutually inconsistent, and the budget is the number that has to
   move.** Job 1392993 (`vcsidonbench.py`) measures SIDON on/off on identical
   clips, including the actual output difference in dB, so this is settled by
   measurement rather than by inference.

The mp3 change earned its keep: encoding four 48 kHz candidates costs 37
ms/sample against the plan's 27 ms for one, because the encode of pack *k* runs
on the CPU pool while the GPU does pack *k+1* and only then is muxed.

Decode is fully hidden: 342x realtime on 16 threads, run on a background thread
underneath the 81 s model load, so it costs zero wall time.

## 2026-08-17 01:45 — Normalisation pass launched (also the multi-node smoke)

17 shards stride-sampled at 2,000 sources each, stratified across all five voice
families (anime 15 voices, emolia 219, k 115, mediathek 124, refvoice 27), on
5 nodes x 4 GPUs via the real `vcrun.py` claim loop. It does three jobs at once:
it is the multi-GPU/multi-node smoke test of the claiming and sbatch path; it
produces ~136 k candidate rows to freeze the reward normalisation on; and it
carries `--ab-presidon 1`, so the SIDON-order A/B is measured on a diverse
sample rather than on one block of one voice.

## 2026-08-17 01:52 — SIDON/TF32: the budget and the quality rule were inconsistent. Measured.

`vcsidonbench.py` job 1392993, 192 stride-sampled clips of `anime_000/001`
(mean 8.67 s), identical input to both arms, batch 32:

| | ms/clip | rescaled to 9.72 s | realtime |
|---|---|---|---|
| TF32 **off** (this run) | 55.6 | **62.4** | 156x |
| TF32 **on** (the pilot's setting) | 40.0 | **44.8** | 217x |

**TF32-on lands on the brief's 38.3 ms; TF32-off does not, and is 1.39x slower.**
That confirms the inference: the 38.3 ms/clip figure — and therefore the
857 GPU-h SIDON line in the 3,588 GPU-h budget — was measured with `tf32=true`
set globally in the cost-model arms, i.e. with TF32 active *inside* SIDON. The
budget and the project's own "never TF32 on SIDON" quality rule cannot both be
satisfied. The rule wins; the budget moves.

Revised SIDON cost: 63.2 ms/clip at the corpus's real mean duration x 80.5 M
clips = **1,413 GPU-h against the planned 857 — an unavoidable +556 GPU-h**,
and it is the price of a quality decision this project already paid to make.

Also measured, since it was cheap and the claim was load-bearing: the numerical
difference between TF32-on and TF32-off SIDON output is 40.7 dB SNR (median
41.0, 5th pct 34.1, worst clip 30.4). That is the *difference between the two
outputs*, which is not the same quantity as the "~15 dB" the brief cites, so it
neither confirms nor refutes it — it is recorded so the two are not conflated
later. TF32 stays off for SIDON regardless: the instruction is explicit and the
cost is now known rather than assumed.

**Corpus mean duration checked** (120 randomly sampled shards, 1,207,036 takes):
**9.85 s**, against the cost model's assumed 9.72 s — 1.3 % high, per-shard means
8.14-10.94 s. So the duration assumption is sound corpus-wide and the smoke
shard's 12.87 s was purely an artefact of `head()` sampling, now fixed. The
corpus is 55,058 h of source audio.

## 2026-08-17 02:05 — Output validated at the audio level, not just the schema

Decoded the smoke test's own output back out of the WebDataset tar:

- 48 kHz, durations match the source to within mp3 frame rounding
  (5.10 s written vs 5.12 s source), peak ~0.86 (SIDON returns peak-normalised
  to 0.9, as documented).
- **The four candidates are genuinely different realisations** — pairwise
  relative RMS difference 1.09-1.47 between candidates of the same source. This
  is the check that best-of-N is doing anything at all: four identical
  candidates would have made the whole 3,588 GPU-h exercise a no-op, and it is
  the kind of thing that is invisible in a schema check.
- Their scores separate too: on one source, target emotion 1.369-1.630, quality
  2.935-2.989, ECAPA 0.818-0.841. The reward picks candidate 2 (best quality
  among the high-emotion candidates), not the emotion argmax (candidate 3) —
  i.e. the two terms are both actually contributing.

## 2026-08-17 02:06 — Interlock added to the sweeper

`vcsweep.py` now refuses to submit any worker while
`index/norm_stats.json` is missing or unusable. Without it every shard silently
falls back to within-set ranking, and the corpus would end up ranked by two
different rewards depending on when each shard happened to run. That is the one
error in this design that stored scores could *not* repair afterwards, because
it is the shipped winner — the audio a consumer reads first — that would be
inconsistent, not just a column. Cheap interlock, expensive failure.

## 2026-08-17 02:35 — Normalisation pass complete. A real bug found, and the reward choice validated.

17 shards x 2,000 stride-sampled sources = **136,000 candidate rows** across all
five voice families, on 5 nodes x 4 GPUs. The claim loop, the sbatch path and
the 4-GPU packing all worked first time; 17/17 shards passed their own
verification and wrote markers.

### Bug found: the VoiceNet group key pooled opposite-sign targets

A `voicenet` gid such as `emolia_c0431|V|ARSH|extremely_low|de` aims a dimension
*low*; another aims the same dimension *high*. The stored strength is
`sign x regression`, so within one group `dim:ARSH` the measured distribution is
**bimodal — mean −2.32 for the low-aiming half and +2.64 for the high-aiming
half**. Its sd (3.12) is therefore a distance between two modes, not a spread,
and z-scoring against it silently divides the emotion term by ~4x too much,
crushing its contribution to the reward on roughly half the corpus.

Fixed: the group key is now `(kind, name, sign)` — 158 groups instead of 97,
each unimodal (`dim:AGEV:-1` sd 0.79, `dim:AGEV:+1` sd 0.65, against 3.12
pooled). This is exactly what the normalisation pass was for.

### A second bug, in the *reporting* rather than the run

The recovery analysis compared the signed converted strength against the corpus's
**unsigned** `dim_target`, which reported a −94 % collapse in VoiceNet target
achievement. That was pure sign error. Sign-matched, the loss is −38.5 %.
Recorded because the wrong number was briefly believed, and because the pilot's
own dim figures (2.552 -> 2.489) are unsigned means and are not comparable to a
signed statistic either.

Note this bug never affected *selection*: within a candidate set all four share
the same target and sign, and `sign x reg` orders the candidates correctly
regardless of centring. It affected the group normalisation and the report.

### The normalisation choice was right, and it is now measured rather than argued

Recovery of target emotion strength, 34,000 sources, all three rewards computed
over the *same* stored scores:

| target | n | source | arbitrary pick | **reward_group** | reward_set | reward_minmax | oracle best-of-4 |
|---|---|---|---|---|---|---|---|
| emotion label | 14,036 | 1.034 | 0.877 | **0.938** | 0.928 | 0.929 | 0.953 |
| VoiceNet dim (signed) | 15,280 | 0.241 | 0.148 | **0.245** | 0.229 | 0.229 | 0.271 |

| target | gap recovered by reward_group | by reward_set | by reward_minmax | reward_group as % of oracle |
|---|---|---|---|---|
| emotion label | **38.7 %** | 32.6 % | 33.0 % | **79.7 %** |
| VoiceNet dim | **104.0 %** | 86.8 % | 87.0 % | **79.1 %** |

**The frozen group z-score beats within-set z by 6.1 points on emotion targets
and 17.2 points on dimension targets**, and reaches ~79-80 % of what a
target-emotion oracle could do at N=4. The argument in the 01:32 entry was made
before this data existed; it now has a measurement behind it. min-max is
indistinguishable from within-set z, as expected — both are dominated by the same
n=4 order statistics.

For VoiceNet dimension targets the reward **fully recovers** the conversion loss
(104 % of the gap: converted-and-ranked slightly exceeds the source's own signed
target achievement). Quality is not paid as the price: Overall-Q under the reward
is 3.122 against 3.104 for the arbitrary pick, and ECAPA is unchanged to 4 dp.

### The A/B the owner asked for — SIDON before ranking is better, not harmful

33,692 sources, both arms judged on the SIDON-enhanced audio because that is what
ships either way; the only difference is which candidate was chosen.

| selection | target emotion | Overall-Q | ECAPA | WavLM |
|---|---|---|---|---|
| **SIDON -> rank** (owner's order, shipped) | **0.7671** | **3.1196** | 0.6408 | 0.8236 |
| rank -> SIDON (pilot's order) | 0.7396 | 3.1107 | 0.6410 | 0.8237 |
| arbitrary pick (k=1) | 0.7036 | 3.1026 | 0.6409 | 0.8234 |

Ranking after restoration buys **+0.0275 target emotion and +0.0090 Overall-Q**
over ranking before it, at a cost of 0.0002 ECAPA — i.e. nothing. The two orders
choose a *different* candidate 52.7 % of the time, so this is a real decision and
not a rounding artefact.

The pilot's "SIDON is harmful" finding does replicate as a statement about SIDON
itself: applied to a fixed candidate it costs −0.039 Overall-Q and −0.015 ECAPA
here (pilot: −0.056 and −0.041). What does not transfer is the conclusion, and
the owner's reasoning is confirmed by measurement: because ranking now happens on
the enhanced audio, selection recovers about 23 % of the Overall-Q that SIDON
costs, instead of being blind to it. **The 642 GPU-h the owner's order costs
buys a better corpus.** It was measured as a diagnostic and gated nothing.

## 2026-08-17 02:50 — Two failures worth writing down

**1. `sbatch --export` splits on commas.** Launching the production-config smoke
with `--export=ALL,VC_SHARDS=anime_016/000,emolia_c0155/001,...` delivered only
`anime_016/000` to the job; sbatch treated the remaining shard ids as further
variable *names*. Three of four GPUs sat idle and printed
`finished, 0 shards`, which looked like a claim-loop bug and was not. Fixed by
passing a JSON file path instead of a comma list — one variable, no commas.

Checked the production path for the same trap: `vcsweep.py` submits with
`--export=ALL,VC_RUNTAG=VC1,VC_OUTROOT=<path>` and neither value contains a
comma, so the bulk run is not exposed to it.

**2. The report generator's f-strings.** `{{}}` is only an escape in the
*literal* part of an f-string; inside a replacement field it is a set literal,
so `(norm or {{}}).get(...)` raised `unhashable type: 'dict'`. Precomputed the
values outside the template instead. Noted only because it is a two-minute
failure that reads like a data problem.

## 2026-08-17 02:55 — Revised budget, reported before it is spent

Measured on 17 stride-sampled shards (34,000 sources), with the A/B pass's
measured 10.2 % share removed to get production config:

| | plan | measured | |
|---|---|---|---|
| s per source sample | 0.642 | **0.791** | |
| GPU-h, full run | 3,588 | **~4,423** | +23 % |
| core-h | 258,300 | **~318,500** | |
| hours on 48 nodes | 18.7 | **~23** | |
| share of remaining budget | 0.43 % | **0.53 %** | (60.6 M core-h left, per `jutil`) |
| output size | 3.9 TB (quoted) | **~16 TB** | the quote was winner-only |

Where the +835 GPU-h comes from, so it can be argued with rather than accepted:

- **SIDON +292 GPU-h** — the TF32 finding above. Not optional without reversing a
  quality decision.
- **source decode +~200-370 GPU-h** — the cost model never included it, and the
  pilot's own write-up says so explicitly ("does not include ... decode of the
  source corpus from its tars"). 55,058 h of mp3 has to be decoded once.
- **scoring and speaker embedding +~165 GPU-h** — the plan's per-sample scoring
  costs were measured on the pilot's *winner only*; here all four candidates are
  scored, because ranking after SIDON requires it.
- **generation is on plan**: 2,657 GPU-h measured against 2,731 planned.

Nothing here is a regression or a defect; the plan was built from a pilot that
scored one candidate, skipped decode, and measured SIDON with TF32 on. Four
full production-config shards are running now to replace the subtraction-based
estimate with a direct one.

**Decision: proceed at the revised figure.** +23 % on 0.53 % of the remaining
allocation is not a decision that warrants stalling a 500-voice corpus, and the
gate leaves ~29 h of lead time for anyone to object before a single GPU-hour is
spent. Stopping is one line — `enabled: false` in
`vcbon/prod/state/sweeper.conf.json`, or `scancel -n vcbon`.

## 2026-08-17 03:00 — Gate ETA revised upward: ~50 h, not ~29 h

The earlier 29 h estimate assumed ~3-4 h per repair voice. Measured from the
per-phase log timestamps of a *completed* voice and a *running* one:

```
mediathek_0475 (done)     codes 16:43 -> train 17:49 -> gen 22:47-22:58 -> finish 22:59   ~6.3 h
emolia_c1430   (running)  codes 23:01 -> train 00:01 -> gen still going at 02:55          ~2.9 h into gen
```

**A voice takes ~6 h, of which ~5 h is generation.** At 48 workers that is
~8 voices/h; 398 outstanding projects to **~50 h**, i.e. the gate clears around
2026-08-19 05:00. Adding ~23 h of VC gives ~73 h end to end.

At 03:00 the last repair completion was 4.0 h ago and 195 repair log files had
been written in the previous 10 minutes — the current cohort is mid-generation,
not hung. The stall clause therefore does not fire (it requires no completion
*and* zero repair workers queued), and the gate holds.

**The gate is measurably not protecting anything**, which is worth stating
plainly at this cost: the repair sweeper is capped by its own
`max_workers: 48`, its `idle_floor: 200` has never bound, and the partition has
had 1,727-3,248 idle nodes throughout. Running 48 VC nodes alongside cannot take
a node from it. Waiting ~50 h buys the repair run zero voices. I am waiting
because the brief states the gate as a condition and the release clause it
offers is not met — but this is the kind of decision that should be made
knowingly, and it is one config key away: `wait_for_repair: false`.

## 2026-08-17 03:00 — The run is now autonomous

`vcsweepd.sh` (restart wrapper) -> `vcsweep.py` is running detached. In one
process it: holds on the repair gate; refuses to submit without frozen
normalisation constants; reaps claims whose Slurm job is gone; tops workers up
to 48 nodes while the partition has >= 200 idle; publishes
`vcbon/prod/state/sweep.json`; appends the gate release to this file; and exits
only when all 2,000 shards carry a verified marker. The wrapper restarts it on
any other exit, because a login-node hiccup during a multi-day wait must not
silently end the run.

Two controls, both re-read every cycle:
- stop everything: `enabled: false` in `vcbon/prod/state/sweeper.conf.json`
- release early: `wait_for_repair: false` in the same file

**A claim-reaping bug found while testing it.** The sweeper built its set of
live Slurm jobs by matching job *names* against `vcbon`, so it reaped the live
claims of the differently-named `vcprodsmoke` job. Harmless there — no other
worker was running — but the identical mistake against a production worker would
have let two GPUs write the same shard's tar concurrently, which the per-shard
verification would not catch because each writer sees a consistent view of its
own file. Fixed to match on job *id* against every job the user has queued.
This is the sort of thing that only shows up because the sweeper was exercised
before the run rather than during it.

## 2026-08-17 03:15 — A latent crash in the shared tokenizer path, found by the full-shard smoke

Job 1393075 died 348 s into `anime_016/000` — a *full* 9,968-source shard:

```
File "chatterbox/models/s3gen/flow.py", line 166, in inference
    token = self.input_embedding(token.long()) * mask
RuntimeError: The size of tensor a (230) must match the size of tensor b (229)
              at non-singleton dimension 1
```

**Cause.** `vcengine.tokenize` pads `tok` to whatever width the S3 tokenizer
returns for the batch, then clamps `ln` down to the true per-clip token count
(`samples // 640`). When that clamp lowers the batch *maximum*, the padded token
width no longer equals `max(ln)`. s3gen's flow builds its mask from `token_len`
and multiplies it against the embedded tokens, so the two disagree by one frame.

It needs a specific combination of clip lengths inside one pack, which is
exactly why nothing before this caught it: the pilot didn't, the 320-source
smoke didn't, and 17 stride-sampled 2,000-source shards across five voice
families didn't. **The first full-size shard did, in under six minutes.** If the
bulk had been launched on the strength of the stride-sampled evidence, this
would have been discovered as a slow bleed of failed shards across 48 nodes
somewhere in hour two of a 23 h run.

**Fixed in `vcprod.py`, not in `vcengine.py`** — trim `tok` to `max(ln)` (and
clamp `ln` to the width in the impossible converse case). Keeping the fix in the
production path leaves the pilot's artefact byte-identical, which matters
because the pilot's published numbers were produced by that file.

**The resumability design did its job**, and this is the first real evidence of
it: `vcrun` saw rc=1, released the claim, and the shard went back on the queue
with no partial output surviving (`.vc-000.tar.partial` is never promoted and
the marker is written last). Nothing had to be cleaned up by hand.

Note the other three shards were already past the same code path with different
length distributions and kept running — the bug is data-dependent, not
systematic, which is the worst kind to discover late.

## 2026-08-17 03:30 — The tokenizer bug hit 2 of 4 full shards

`mediathek_0051/003` died the same way 764 s in. So on full-size shards the
failure rate is **~50 %**, not a rare edge case: the stride-sampled evidence that
looked clean across 17 shards and five voice families was clean only because
sampling every 5th take changes which clip lengths land in a pack together.

Had the bulk gone out on that evidence, roughly half of 2,000 shards would have
failed several minutes in, on 48 nodes at once, and the sweeper would have
dutifully re-queued them into the same crash. The run would not have stalled
visibly — it would have burned GPU-hours re-attempting, which is the expensive
kind of failure.

Both failed shards relaunched with the fix. The two shards that were already
past that code path are still running and will provide the throughput number.

## 2026-08-17 03:45 — Second data-dependent crash: clips too short for s3gen

`mediathek_0051/003`, relaunched with the tokenizer fix, died on its **first
pack**:

```
RuntimeError: Argument #4: Padding size should be less than the corresponding
input dimension, but got: padding (4, 4) at dimension 2 of input [64, 128, 4]
```

A 0.08 s take is 2 tokens; s3gen's flow decoder tries to pad (4,4) into a
dimension of 4. Measured over 604,478 takes in 60 randomly sampled shards:

| duration | share | corpus estimate |
|---|---|---|
| < 0.16 s | 0.204 % | ~41,000 |
| < 0.50 s | 0.493 % | ~99,000 |
| < 1.28 s | 1.085 % | **~218,000 takes** |

Corpus minimum is 0.08 s. **And because packs are length-sorted, every one of a
shard's shortest takes lands in its first pack** — so this is not a rare tail
case that would trickle in, it is a first-minute kill for any shard whose
minimum duration is small. That is exactly what `mediathek_0051/003` (minimum
0.08 s) did, twice.

**Fix: pad up, generate, restore, trim back.** Sources shorter than
`MIN_TOK = 32` tokens (1.28 s) are zero-padded for generation *and* for SIDON,
and trimmed to their true length only after restoration. Padding rather than
skipping, because the spec is all 20,125,736 takes. Trimming after SIDON rather
than before, because SIDON's stacked-frame extractor is no safer on a 0.08 s
input than s3gen is. Verified exact: a 0.08 s source yields 3,840 samples at
48 kHz = 0.080 s out; 0.16 s -> 0.160 s; 1.00 s -> 1.000 s.

**Cost, recorded honestly:** SIDON is non-causal, so restoring a padded clip and
trimming is not bit-identical to restoring the clip alone. For ~1.1 % of the
corpus the restoration sees a little trailing silence. The alternative was not
converting those takes.

## 2026-08-17 03:50 — Per-pack fault isolation

Two different data-dependent crashes in the first four full shards is enough
evidence about what the next 1,996 hold. A pack is now the fault-isolation unit:
a failing pack is rolled back completely (stored rows, provenance, the candidate
counter, and its un-muxed mp3 futures, with the *previous* pack's pending futures
preserved for the next drain), recorded by source key in `failed-<NNN>.json`, and
the shard carries on.

Two new verification gates make sure this cannot become a silent loss:

- **accounting**: `sources_done + sources_failed == sources_in`. A shard that
  quietly converted 9,000 of 10,000 takes and called itself complete is the
  failure this exists to prevent.
- **loss tolerance**: more than `max(64, 1 %)` of a shard's sources failing is a
  systematic fault wearing a few bad packs as a disguise. The shard fails
  verification, writes no marker, and goes back on the queue.

Without this, each of these bugs discarded ~2 h of completed work per shard and
was re-queued straight back into the same input — a re-attempt loop that burns
GPU-hours without ever visibly stalling.

## 2026-08-17 04:45 — The clearest possible argument for the fault isolation

`emolia_c0155/001` was launched at 02:20, before the tokenizer fix, and hit that
bug at **10,016 of 10,128 sources** — the last pack of the shard. Two hours and
six minutes of completed conversion, discarded, because the final 112 takes
had an unlucky length combination.

Under the code as it now stands this shard would not have crashed at all (the
tokenizer fix removes the cause), and had it still failed, the pack would have
been rolled back and recorded and the shard would have finished. Both defences
earn their place from this one shard.

**Throughput, from that shard's own progress at 99 % of a full-size run:
0.761 s/sample.** `k91_age5_bg0/002` at 91 %: 0.811. `anime_016/000` at 76 %:
0.717. The rate rises through a shard because packs are length-sorted ascending,
so these are lower bounds on their own finals. Converging on **~0.79 s/sample**,
which is what the subtraction-based estimate from the normalisation pass
predicted (0.791) — the two methods agree to 0.1 %.

## 2026-08-17 05:00 — CORRECTION to the 04:45 entry, and the definitive throughput

**The 04:45 entry was wrong and is corrected here rather than edited away.**
I attributed the `172 vs 171` traceback in job 1393077's stderr to
`emolia_c0155/001` on the basis of timing. It was not: that job ran three shards
and the traceback belonged to `mediathek_0051/003`'s earlier failure in the same
job. `emolia_c0155/001` **completed**, all 10,128 sources, all five verification
gates green. Lesson, since it is a general one: a shared stderr stream from a
four-worker job is not attributable by position, and I should have matched on the
shard id before writing it down.

The argument for per-pack fault isolation stands on `mediathek_0051/003` and
`anime_016/000`, which is enough.

### Definitive production numbers — one complete, verified, full-size shard

`emolia_c0155/001`: 10,128 sources, 40,512 candidates, 24.96 h of source audio,
frozen normalisation constants in use, production config (no A/B pass).

```
                     s/sample    share    plan (cost model n_cand=4)
generate               0.4883    62.3%      0.4884   <- exact
SIDON x4               0.1866    23.8%      0.1532   (46.65 ms/clip vs 38.3)
score x4               0.0650     8.3%      0.0266
decode                 0.0423     5.4%      not in plan
speaker x4             0.0202     2.6%      0.0031
tokenize/io/mp3        0.0185     2.2%      0.0285
-------------------------------------------------
total                  0.7836              0.642
```

| | plan | **measured** | |
|---|---|---|---|
| s per source sample | 0.642 | **0.7836** | |
| GPU-h, full run | 3,588 | **4,381** | +22.1 % |
| core-h | 258,300 | **315,412** | 0.52 % of the 60.6 M remaining |
| hours on 48 nodes | 18.7 | **22.8** | |
| SIDON ms/clip | 38.3 | **46.65** | |
| output | 3.9 TB (quoted) | **14.4 TB** | the quote was winner-only |
| realtime | &mdash; | **11.3x** | |

**Generation is exact to four decimal places** — 0.4883 measured against the cost
model's 0.4884. The pilot's model of the expensive half of this pipeline was
right; the +22 % is entirely in the parts the pilot did not measure at production
config:

- **SIDON +186 GPU-h.** 46.65 ms/clip against 38.3. The dedicated bench put
  TF32-off at 1.39x TF32-on; in production the gap is smaller (1.22x) because
  the bench used one batch of uniform 8.67 s clips while production runs
  length-bucketed mixed lengths. Direction and cause confirmed either way.
- **source decode +236 GPU-h.** Never in the plan, and the pilot's own write-up
  says so. 55,058 h of mp3 must be decoded once, and only ~90 s per shard of it
  hides under the model load.
- **scoring and speaker embedding +~470 GPU-h.** The plan costed scoring the
  *winner*; ranking after SIDON requires scoring all four.

Two independent estimates of the same quantity agree: the subtraction from the
normalisation pass gave 0.791 s/sample, the direct full-shard measurement gives
0.7836 — 0.9 % apart.

### Reward performance holds on production data

`emolia_c0155/001`, 7,824 emotion-target sources, reward recomputed from the
frozen constants over the stored raw scores:

| | source | arbitrary | reward_group | reward_set | reward_minmax | oracle |
|---|---|---|---|---|---|---|
| target emotion | 0.890 | 0.685 | **0.755** | 0.743 | 0.743 | 0.771 |
| gap recovered | | 0 % | **34.0 %** | 28.3 % | 28.5 % | 100 % |
| % of oracle | | | **80.5 %** | 67.0 % | 67.5 % | |

Quality rises with it (3.160 vs 3.136 for the arbitrary pick) and ECAPA is
unchanged (0.6826 vs 0.6828). Consistent with the 17-shard normalisation pass
(38.7 % / 79.7 %) on a completely different voice.
