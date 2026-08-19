# Voice conversion in the MOSS voice-profile project

This folder consolidates everything the project built around **voice conversion (VC)** —
what it is for, the two competing ways it is applied to assembled multi-clip audio, the
measurements behind each, the known defects, and how to reproduce any of it.

Nothing here has been deleted or replaced. Both competing variants are kept, and the
defective early artefacts are kept too, deliberately, as diagnostics. Where something was
never finished, this document says so rather than rounding it up.

Everything below is measured unless it is explicitly labelled as an argument or an
assumption. Numbers carry the file they came from.

Written 2026-08-19. Base path used throughout:
`NB = /e/data1/datasets/playground/mmlaion/schuhmann1/dramabox`

---

## 1. What voice conversion is doing here, and why it was wanted

The project builds a per-voice conditioned-speech corpus: for each of 500 voices, a TTS
model generates millions of takes under controlled conditions (40 emotion labels × 2
intensity conditions, 57 VoiceNet dimensions × 4 levels, vocal bursts, edge cases, two
languages). Voice conversion was brought in to solve two distinct problems that the
generator does not solve on its own.

**Problem 1 — identity drift.** A generated take frequently does not sound like the voice
it was supposed to be. Measured over the whole corpus by the identity-repair census
(`$NB/vprof/repair/report/census_summary.json`): **8,177,013 takes across 499 voices fall
below the ECAPA speaker-similarity floor of 0.40**, a mean below-floor fraction of 40.9 %
per voice. Regenerating those takes is expensive and throws away the performance. Voice
conversion instead re-imposes the target speaker's timbre on the audio that already
exists, keeping the acting.

It works, and the effect is large. On the 354-take pilot
(`data/vcbon_pilot_analysis.json`, arm `self` = convert to the voice's own reference):

| | source takes | after VC to own reference |
|---|---|---|
| ECAPA to reference | 0.492 | 0.649 (0.678 against the prepared reference) |
| fraction above the 0.40 floor | 63.0 % | **98.3 %** |
| of the takes that were *below* floor, fraction rescued | — | **95.4 %** |

**Problem 2 — seams in assembled audio.** Several downstream artefacts splice 3–6
separately generated takes into one continuous piece: the "assembled voice-acting scenes"
(`../acting_concats.html`) and the emotion-trajectory collages
(`../emotion_trajectories.html`, `../trajectories.html`). Each part is an independent draw,
so the assembly can wander in timbre and the joins can be audible. The question of *how*
to apply VC to such an assembly is where the two competing variants come from — see §3.

**What VC costs.** Emotion. On the pilot, takes conditioned on an emotion label lost
roughly half their measured target-emotion strength through conversion (1.246 → 0.613,
**−50.8 %**; `data/vcbon_pilot_analysis.json → target_dimension`). VoiceNet-dimension
targets survived far better (2.552 → 2.480 unsigned). Recovering that emotion loss is the
entire reason best-of-N exists in this project: generate several candidates, keep the one
that is expressive *and* clean.

---

## 2. The shared machinery

Both variants run on the same engine, `code/engine/vcengine.py` (copied from
`$NB/vcbon/code/vcengine.py`).

| role | model | notes |
|---|---|---|
| voice conversion | **Chatterbox VC** (`ResembleAI/chatterbox`) | S3 tokenizer @16 kHz → 25 Hz content tokens → S3Gen conditional flow-matching decoder conditioned on the target x-vector → HiFi-GAN @ **24 kHz** → PerTH watermark |
| speech restoration | **SIDON** (`sarulab-speech/sidon-v0.1`) | w2v-BERT encoder + DAC decoder, **48 kHz** out, returns peak-normalised audio at 0.9 |
| speaker identity | **ECAPA-TDNN** (`speechbrain/spkrec-ecapa-voxceleb`, 192-d) | the 0.40 floor is defined on this |
| timbre, second opinion | **Orange/Speaker-wavLM-tbr** (512-d) | deliberately independent of ECAPA |
| emotion + quality | **BUD-E-Whisper** encoder + 40 EmoNet heads + 4 quality heads (`laion/Empathic-Insight-Voice-Small` / `-Plus`) | plus 57 VoiceNet regressions, genuineness, vocal-burst blend |

**Chatterbox VC is stochastic** — the flow-matching decoder starts from Gaussian noise, so
each seed is a different realisation. This is verified rather than assumed: on production
output the four candidates of one source differ by a pairwise **relative RMS difference of
1.09–1.47**, and their scores separate (target emotion 1.369–1.630, quality 2.935–2.989,
ECAPA 0.818–0.841 on one source; `docs/PROTOCOL.md`, 02:05 entry). Four identical
candidates would have made best-of-N a no-op.

**TF32 policy — set at the call site, never globally.** ON around `s3gen.inference`,
OFF around SIDON.

- TF32 on generation is free: identical scores to 4 decimal places (ECAPA 0.62606 vs
  0.62609, Overall-Q 3.0114 vs 3.0118) at **1.60× the speed** (21.46 s → 13.44 s)
  — `data/vcbon_ref_ablate.json → tf32`.
- TF32 inside SIDON is *not* free and is disabled. It costs throughput: 55.6 ms/clip off
  vs 40.0 ms/clip on, i.e. **1.39× slower** (`data/vcbon_sidon_tf32.json`). The numerical
  difference between the two SIDON outputs is 40.7 dB SNR (median 41.0, 5th pct 34.1,
  worst clip 30.4). That is the difference between two outputs, which is *not* the same
  quantity as the "~15 dB" quality claim the original instruction cited — the two are
  recorded separately so they are not conflated.
- Consequence, recorded because it changed a budget: the project's original SIDON cost
  line (38.3 ms/clip, 857 GPU-h) had been measured with TF32 globally on. The quality
  rule and that budget cannot both hold. The rule won; the budget moved to 1,413 GPU-h.

**The conversion target ("prepared reference").** Every voice has one canonical reference
in `$NB/vcbon/refs500/refs_prepared.tar` (500/500 present, verified), built by
`decode → 16 kHz → SIDON(48 kHz) → EBU R128 −23 LUFS → 0.95 peak ceiling`
(`refs500/prep_config.json`). That chain was chosen from a five-arm ablation over 768
conversions, all scored against the *untouched original* reference so the arms are
comparable (`data/vcbon_ref_ablate.json`):

| reference preparation | ECAPA to original ref | Overall-Q | burst blend | above floor |
|---|---|---|---|---|
| raw, −23 LUFS | **0.682** | 2.973 | 2.707 | 96.1 % |
| raw, peak-normalised | 0.681 | 2.950 | 2.593 | 96.1 % |
| SIDON, −18 LUFS | 0.625 | 3.004 | 2.869 | 95.1 % |
| **SIDON, −23 LUFS (chosen)** | 0.626 | **3.012** | **2.972** | 94.8 % |
| SIDON, peak-normalised | 0.624 | 3.003 | 2.861 | 95.2 % |

Read honestly: preparing the reference with SIDON **costs 0.056 ECAPA against the raw
original** and buys +0.039 Overall-Q and +0.265 burst blend. It is a trade, not a free
win, and the run therefore stores speaker similarity against **both** anchors — the
prepared reference and the original — so the asymmetry stays visible instead of being
hidden by whichever anchor flatters the arm.

---

## 3. The two competing variants

The competition is about **assembled multi-clip audio**, not about single takes. For a
single take there is only one thing to do. For a spliced scene or a trajectory collage
there are two, and the project kept both.

The dispute has a concrete origin, recorded on the published scenes page
(`../acting_concats.html`):

> *"+ VC, per part — what the production pipeline does: each part converted toward the
> anchor separately, then spliced. + VC, whole clip — the assembled scene converted in
> one pass. **A listener reported the per-part version still sounds fragmented**, and the
> code confirms why: each part is converted with no knowledge of its neighbours, so the
> converter cannot smooth a seam it never sees."*

That is a hypothesis from one listener plus a plausible mechanism. The **four-arm study**
(`$NB/vc4arm`) was built to measure it. Its design decisions are recorded in
`docs/FOURARM_STATE.md` and are authoritative.

### 3.1 Variant A — per-clip conversion, then restore, then concatenate

*This is the corpus pipeline's recipe, applied at collage level. It is arm 3 of the study.*

```
clip_1 ─┐
clip_2 ─┼─► Chatterbox VC per clip, target = the voice's own prepared reference
clip_n ─┘        ↓
            SIDON restoration per clip
                 ↓
            re-level each clip to its designed −23 LUFS ± deviation
                 ↓
            concatenate: 15 ms sqrt fades, fixed 150 ms pause at every seam,
                         −1 dBFS peak ceiling on the whole thing
```

Code: `code/collage/arms.py::vc_clips` + `E.Sidon.restore` + `concat`.

**What it is good at**

- **It is the only arm that makes the audio sound more like the voice it claims to be.**
  ECAPA against the voice's canonical prepared reference: **+0.153 vs the raw
  concatenation, better on 20/20 collages, t = +15.4**. WavLM agrees independently:
  **+0.121, 19/20, t = +7.0**. Both whole-file arms are statistically indistinguishable
  from raw on this measure (arm 2: −0.021, t = −1.3; arm 4: −0.003, t = −0.1).
- **It preserves the collage's designed structure.** The 150 ms inter-clip pauses stay at
  digital silence (−200 dB, i.e. exact zeros). Durations, seam positions and the per-clip
  loudness design survive intact.
- It fixes the "one person all the way through" problem nearly as well as a whole-file
  pass: within-collage speaker spread 0.348 → **0.174** (0/20 worse, t = −10.6).
- It reduces seam excess as much as a whole-file pass does — see the finding in §3.3.

**What it is bad at**

- **It drifts away from what the listener hears first.** ECAPA to the collage's own
  opening snippet: **−0.176, 0/20 better, t = −9.1**. WavLM: −0.133, 3/20. If the listener's
  reference point is the beginning of the clip rather than an external voice profile, this
  arm changes the voice out from under them.
- **Quality does not improve and may fall slightly**: Overall-Q −0.018 vs raw, better on
  only 3/20, **t = −1.83 — not significant**. Against the whole-file pass it is clearly
  behind: −0.066, 3/20, t = −6.2.
- **Largest emotion loss of the single-pass arms**: target-emotion z −0.389 vs raw
  (1/20 better, t = −5.9).
- **Most internally unstable frame to frame**: interior MFCC jump +0.618 vs raw, worse on
  20/20, t = +12.1 — the largest of any arm. Per-clip conversion makes each clip more
  variable inside itself.
- **Most expensive.** ~1.5× the GPU cost of a whole-file pass (§3.4).

### 3.2 Variant B — concatenate first, convert the finished soundscape in one pass

*Arm 2 of the study. Arm 4 = variant A followed by variant B.*

```
clip_1..n ─► concatenate (same recipe: 15 ms fades, 150 ms gaps, −1 dBFS)
                 ↓
            ONE Chatterbox VC pass over the entire file,
            target = the collage's FIRST SNIPPET (−23 LUFS, ceiling 0.95)
```

Code: `code/collage/arms.py::vc_whole`.

**What it is good at**

- **Best quality.** Overall-Q **+0.048 vs raw, better on 19/20, t = +5.9**. Speech quality
  +0.012, 16/20, t = +3.2.
- **Best internal consistency with what the listener actually hears.** ECAPA to the
  collage's first snippet **+0.121, 20/20, t = +8.7**; WavLM +0.065, 17/20, t = +3.6.
- **Lowest speaker spread of the single-pass arms**: 0.348 → **0.151** (t = −11.9),
  slightly better than variant A (0.174; head-to-head t = +3.8 in variant A's disfavour).
- **Cheapest** — one forward pass over the whole file, no SIDON (§3.4).
- Smallest emotion loss of any converting arm: −0.266 z vs raw, against −0.389 for
  variant A (head-to-head +0.123 in variant B's favour, t = +2.2).

**What it is bad at**

- **It fills the silence.** The 150 ms designed pauses stop being silent. Mean gap level
  goes from digital silence to **−71.8 dBFS**, which is 49.1 dB below the speech in the
  same file — median 52.8 dB below, but **worst case only 29.0 dB below** on one collage,
  and arm 4 is far worse (mean 29.5 dB below, worst case **10.8 dB**, i.e. clearly
  audible). The generator invents room tone, breath and artefacts where the design put
  nothing. This is the single defect that makes variant B unusable for anything where the
  pause structure is part of the content.
- **It does not restore the voice's identity.** ECAPA and WavLM against the canonical
  reference are indistinguishable from doing nothing at all. If the artefact's job is to
  demonstrate a voice, this arm does not do that job.
- **Length drift.** The output is not the same length as the input: −0.030 s per collage
  (−0.086 %) for arm 2, −0.128 s (−0.337 %) for arm 4, on 0/20 collages preserved. Segment
  boundaries have to be rescaled by the length ratio, which the code does
  (`arms.py`, `len_ratio`), but any externally stored timing goes stale.
- **Genuineness drops**: −0.124 vs raw, t = −2.2 (arm 4: −0.146, t = −2.8).
- The whole file is conditioned on **one snippet** — its own opening — so a collage whose
  first clip is atypical propagates that atypicality across the entire piece.

### 3.3 The head-to-head numbers

20 trajectory collages, 833 s, 20 distinct voices, 96 clips, 480 measured segments.
All arms built from **identical, silence-fixed source clips** so the comparison measures
the conversion and not the fix. Every finished soundscape levelled to −23 LUFS / 0.95
ceiling so no arm wins on loudness. Full tables: `data/fourarm_*.csv`,
raw rows in `data/fourarm_metrics.parquet` and `data/fourarm_segments.parquet`.

Arm means (n = 20 each):

| metric | arm1 raw | **arm2 whole-file** | arm2b (seed twin) | **arm3 per-clip** | arm4 both |
|---|---|---|---|---|---|
| ECAPA → voice's reference | 0.4914 | 0.4704 | 0.4743 | **0.6441** | 0.4889 |
| WavLM → voice's reference | 0.7376 | 0.7466 | 0.7431 | **0.8586** | 0.7578 |
| ECAPA → collage's first snippet | 0.7226 | **0.8433** | 0.8439 | 0.5465 | 0.8313 |
| WavLM → collage's first snippet | 0.8249 | **0.8895** | 0.8920 | 0.6920 | 0.8873 |
| target emotion (z) | **2.1367** | 1.8706 | 1.8555 | 1.7479 | 1.6951 |
| Overall-Q | 3.0920 | 3.1399 | 3.1412 | 3.0739 | **3.1421** |
| genuineness | **1.2423** | 1.1187 | 1.0903 | 1.1499 | 1.0960 |
| within-collage speaker spread (lower better) | 0.3479 | 0.1508 | 0.1523 | 0.1744 | **0.1335** |
| seam excess (lower better) | 1.5332 | 1.0526 | 1.0899 | 1.0472 | **0.9466** |
| interior MFCC jump (lower better) | **3.4221** | 3.7009 | 3.7268 | 4.0402 | 3.8099 |
| gap level below speech, dB (higher = quieter pauses) | **177.4** | 49.1 | 48.1 | 177.1 | 29.5 |

**arm2b is the noise floor and it matters.** It is arm 2 re-run with a different seed —
same recipe, same input, same target. Every difference reported above is far larger than
the seed-to-seed noise it measures: ECAPA-first ±0.0006, Overall-Q ±0.0012, emotion z
±0.015, speaker spread ±0.0015, seam excess ±0.037
(`data/fourarm_seed_noise_arm2b_vs_arm2.csv`). Only one comparison is inside the noise —
see the next paragraph.

**Finding: the seam hypothesis is not confirmed.** The whole-file pass was adopted on the
theory that a converter which sees the seam can smooth it. It does reduce seam excess
(1.533 → 1.053, t = −5.9). But so does per-clip conversion (1.533 → 1.047, t = −3.4), and
the difference between the two is **−0.0055 with t = −0.04** — indistinguishable, and
smaller than the seed noise of 0.037. Both arms fix the seam by the same amount, because
both converge every clip onto a single common target; seeing the join contributes nothing
measurable. If listeners still prefer the whole-file version — and one did — the cause is
not timbre discontinuity at the join, which is what these metrics capture. The most likely
remaining candidates are prosodic discontinuity across clips and the presence or absence
of room tone in the pauses; neither was measured, and this document does not claim to
have explained the listener report.

**There is no overall winner, and the measurements say so plainly.** The two variants
disagree on the *reference question itself*: variant A moves the audio toward the voice's
canonical identity and away from the collage's own opening; variant B does exactly the
reverse. Both moves are large (0.15–0.30 ECAPA), both are consistent (19–20 of 20), and
which one is "right" depends entirely on what the artefact is for. This asymmetry was
anticipated in the design — `docs/FOURARM_STATE.md` records that both anchors are measured
"so the reference asymmetry is visible rather than hidden".

### 3.4 Cost

From the study's own stage timings (`$NB/vc4arm/logs/vc4arms-1399769.out`), one GH200,
20 collages, 833 s of source audio:

| variant | GPU seconds per collage | realtime factor |
|---|---|---|
| B, whole-file pass | 0.87 (52.0 s / 60 passes) | ~47× |
| A, per-clip VC + SIDON | 1.32 (15.0 s VC + 11.3 s SIDON, / 20) | ~31× |

Variant A costs about **1.5×** variant B. At this scale neither is a budget item; the
choice should be made on the quality axes, not on cost.

### 3.5 Which to use

| if the artefact's job is… | use | because |
|---|---|---|
| demonstrating **a voice** — profile pages, speaker-identity datasets, anything where "is this the same speaker as the reference?" is the question | **A — per-clip + SIDON** | it is the only arm that improves ECAPA (+0.153) and WavLM (+0.121) against the canonical reference, on 20/20 and 19/20 collages |
| a **coherent listening experience** — demo pages, scenes heard end to end, where the listener's only reference is the clip itself | **B — one whole-file pass** | best quality (+0.048, 19/20), best consistency with the opening (+0.121, 20/20), lowest speaker spread, least emotion loss, cheapest |
| audio where the **pause structure is content** (designed silences, timing-sensitive assemblies, anything downstream that segments on the gaps) | **A only** | B fills the pauses to 49 dB below speech on average and 29 dB in the worst case; arm 4 reaches 10.8 dB, which is audible |
| seam artefacts are the dominant complaint and nothing else matters | arm 4 (A then B) | lowest seam excess (0.947) and lowest speaker spread (0.134) — but the largest emotion loss (−0.442 z) and by far the worst gap fill |
| corpus-scale conversion of **individual takes** | A's recipe (this is what the corpus pipeline is) | there is no assembly to convert whole; see §4 |

Anything that needs both properties needs a variant that does not exist yet: per-clip
conversion for identity, followed by something that re-imposes silence on the pauses of a
whole-file pass. Nobody built that, and this document does not pretend it exists.

---

## 4. The corpus-scale pipeline (best-of-N), which is where variant A's recipe comes from

`$NB/vcbon` — code copied to `code/corpus/`, full append-only record in `docs/PROTOCOL.md`.

**Spec.** For each of **20,125,736** source takes across 500 voices: generate **4 VC
candidates**, run SIDON on **all four**, rank the SIDON-enhanced candidates by
`normalise(target emotion strength) + normalise(quality)`, and **store everything** — all
four outputs, all raw scores, all rewards, all components. Selection is a *view* over
stored scores, never a filter applied at write time, so a different reward can be applied
later without regenerating anything.

### 4.1 Best-of-N works, and its size was chosen by measurement

Pilot, 354 takes × 8 candidates, converting to the voice's own reference
(`data/vcbon_pilot_analysis.json → best_of_k`):

| k | ECAPA | above floor | emotion strength | Overall-Q |
|---|---|---|---|---|
| 1 | 0.6805 | 97.74 % | 0.5930 | 3.1340 |
| 2 | 0.6850 | 97.74 % | 0.6131 | 3.1415 |
| 4 | 0.6883 | 98.02 % | 0.6138 | 3.1492 |
| 8 | 0.6901 | 98.31 % | 0.6297 | 3.1553 |

Gains are real but sublinear; the public write-up of the same method
(`$NB/code/chatterbox-voice-conversion/expressive_bestofn/README.md`) puts k = 8 at ~71 %
of the 1→32 reward gain and recommends 8. The corpus run uses **N = 4**, a cost decision:
generation is 62 % of pipeline time and all four candidates must additionally be
SIDON-restored and scored.

### 4.2 The two orderings — a second competing pair, and here the measurement does pick one

There are two ways to combine restoration with selection, and the project has shipped both
at different times:

- **rank → SIDON the winner** (the pilot's design, and what the public
  `expressive_bestofn` repo recommends): cheapest, roughly half the SIDON cost.
- **SIDON all → rank** (the corpus run's design): the ranker scores exactly the audio that
  ships.

Measured on 33,692 sources, both arms judged on SIDON-enhanced audio because that is what
ships either way — the only difference is which candidate was chosen
(`docs/PROTOCOL.md`, 02:35 entry):

| selection order | target emotion | Overall-Q | ECAPA | WavLM |
|---|---|---|---|---|
| **SIDON → rank** (shipped) | **0.7671** | **3.1196** | 0.6408 | 0.8236 |
| rank → SIDON (pilot's) | 0.7396 | 3.1107 | 0.6410 | 0.8237 |
| arbitrary pick (k = 1) | 0.7036 | 3.1026 | 0.6409 | 0.8234 |

Ranking after restoration buys **+0.0275 target emotion and +0.0090 Overall-Q** for
**−0.0002 ECAPA**, and the two orders pick a *different* candidate **52.7 %** of the time,
so this is a real decision and not a rounding artefact. The cost is +642 GPU-h.

The pilot's "SIDON is harmful" finding replicates as a statement about SIDON itself —
applied to a fixed candidate it costs −0.039 Overall-Q and −0.015 ECAPA (pilot: −0.056 and
−0.041) — but the *conclusion* does not transfer, because ranking on the enhanced audio
recovers about 23 % of the quality SIDON costs instead of being blind to it.

### 4.3 The reward, and why it is normalised the way it is

`reward = z(target emotion strength) + z(quality)`, z-scored against **frozen corpus-wide
group statistics**, not within the 4-candidate set. Groups: `(target kind, target name,
sign)` for emotion — 158 groups — and one global group for quality.

Within-set normalisation forces the two terms to contribute equally in *every* set,
whatever the set looks like; with n = 4 the sd estimate carries ~40 % relative error, so
the effective weighting is re-randomised per sample. Frozen group z-scoring divides each
term by the spread that term has across the corpus, so a term that is flat within a given
set correctly contributes almost nothing to that set's decision.

That was an argument. It was then measured over 34,000 sources, all three candidate rewards
recomputed over the *same* stored scores
(`data/vcbon_normpass_analysis.json`, `data/vcbon_prod_analysis.json`):

| target | source | arbitrary pick | **reward_group** | reward_set | reward_minmax | oracle best-of-4 |
|---|---|---|---|---|---|---|
| emotion label (n = 14,036) | 1.034 | 0.877 | **0.938** | 0.928 | 0.929 | 0.953 |
| VoiceNet dim, signed (n = 15,280) | 0.241 | 0.148 | **0.245** | 0.229 | 0.229 | 0.271 |

| target | gap recovered, reward_group | reward_set | reward_minmax | % of oracle |
|---|---|---|---|---|
| emotion label | **38.7 %** | 32.6 % | 33.0 % | **79.7 %** |
| VoiceNet dim | **104.0 %** | 86.8 % | 87.0 % | **79.1 %** |

Frozen group z beats within-set z by **6.1 points** on emotion targets and **17.2 points**
on dimension targets, and reaches ~80 % of what a target-emotion oracle could do at N = 4.
Quality is not paid as the price (3.122 vs 3.104 for the arbitrary pick) and ECAPA is
unchanged to four decimal places. The result replicates on a complete production shard of
a different voice: 34.0 % gap recovered, 80.5 % of oracle, quality 3.160 vs 3.136, ECAPA
0.6826 vs 0.6828.

The `sign` component of the group key was a **bug fix that this pass existed to find**. A
VoiceNet group such as `dim:ARSH` pools takes aiming the dimension *low* (mean −2.32) with
takes aiming it *high* (mean +2.64); its pooled sd of 3.12 is a distance between two modes,
not a spread, and z-scoring against it divided the emotion term by ~4× too much on roughly
half the corpus. Split by sign, the groups are unimodal (`dim:AGEV:−1` sd 0.79,
`dim:AGEV:+1` sd 0.65). Selection was never affected — all four candidates in a set share
the same target and sign — but the normalisation and the report were.

### 4.4 Measured throughput and cost

From one complete, verified, full-size shard (`emolia_c0155/001`: 10,128 sources, 40,512
candidates, 24.96 h of source audio, production config, frozen constants;
`data/vcbon_prod_analysis.json`):

```
                     s/sample   share    cost-model plan
generate               0.4883   62.3%      0.4884   <- exact to 4 dp
SIDON ×4               0.1866   23.8%      0.1532   (46.65 ms/clip vs 38.3 planned)
score ×4               0.0650    8.3%      0.0266
decode                 0.0423    5.4%      not in plan
speaker ×4             0.0202    2.6%      0.0031
tokenize / io / mp3    0.0185    2.2%      0.0285
--------------------------------------------------
total                  0.7836              0.642
```

| | plan | measured |
|---|---|---|
| GPU-h, full run | 3,588 | **4,381** (+22.1 %) |
| core-h | 258,300 | **315,412** |
| hours on 48 nodes | 18.7 | **22.8** |
| output size | 3.9 TB (quoted) | **14.4 TB** |
| realtime | — | **11.3×** |

The quoted 3.9 TB was the *winner-only* figure; the spec stores all four candidates. The
+22 % is entirely in parts the pilot never measured at production config: SIDON with TF32
off (+186 GPU-h), source decode which was never in the plan (+236 GPU-h, 55,058 h of mp3
must be decoded once), and scoring all four candidates rather than the winner (+470 GPU-h).
Generation itself came in exact to four decimal places. Two independent estimates of the
same quantity agree to 0.9 % (0.791 by subtraction from the normalisation pass, 0.7836
direct).

---

## 5. Known defects, stated plainly

### 5.1 `collage_traj` mp3s carry an internal-silence defect — and are kept on purpose

`$NB/collage_traj/build.py` trims silence only from the **head and tail** of each clip. It
has no internal-silence handling. Long quiet stretches *inside* a clip therefore survive
into the concatenation, on top of the 150 ms pauses the design puts at each seam.

Measured on the 96 clips of the 20 selected collages
(`data/fourarm_silence_audit.json`):

- **43 of 96 clips (44.8 %)**, in **19 of 20 collages**, contained an internal quiet
  stretch longer than 400 ms.
- 56 such stretches: min 0.42 s, median 0.50 s, p75 0.64 s, **max 1.24 s**.
- 23.62 s of unintended silence across 821.7 s of clip audio — **2.9 %**.

The affected files are `$NB/collage_traj/mp3/*.mp3` (100 files, 62 MB) and their
byte-identical copies in `../traj_audio/`, which is what
`../emotion_trajectories.html` serves. (Verified: `C000.mp3` md5
`0126249d2b04c310b6b3712b06ff4fca` in both locations.)

**They are deliberately retained, not deleted.** They are the diagnostic — the "before"
recording of what the defect sounds like, and the reference against which the fix was
judged. `docs/FOURARM_STATE.md` records this explicitly: *"The existing
`collage_traj/mp3/*.mp3` still carry the internal-silence defect and are kept only as a
diagnostic."* Do not treat them as current output, and do not use them for any
listening comparison that is not about this bug.

**The fix** lives in `code/collage/prep.py::collapse_internal`: any internal quiet stretch
longer than 400 ms is collapsed to exactly 150 ms with a 40 ms equal-power crossfade,
built from the stretch's own first and last material so the room tone is preserved rather
than replaced by digital silence. The silence detector is byte-for-byte the same rule as
the head/tail trim (20 ms RMS frames, p95 − 35 dB, clamped to −60 dBFS). After the fix,
**zero** stretches over 400 ms survive. The same fix propagated into the successor
collage builder `$NB/collage_traj2/build2.py` (same 400 ms / 150 ms thresholds, 20 ms
crossfade).

The consequence for the four-arm study is that arm 1 had to be **rebuilt** rather than
reused — the brief's "reuse arm 1" and "all four arms see identical silence-fixed source"
are incompatible. Rebuilding was the recorded decision.

### 5.2 The four-arm page was never built

`docs/FOURARM_STATE.md` lists two unfinished stages: `$NB/vc4arm/code/page.py → vc4arm.html` and
publication. Neither happened. The 100 arm mp3s exist only at
`$NB/vc4arm/out/arm{1,2,2b,3,4}/*.mp3` and have never been published. **This README is the
first published account of the four-arm numbers.** Anyone who wants the listening
comparison has to build the page, or copy those mp3s into the repo.

### 5.3 The 2,000-shard corpus production run never started

`$NB/vcbon/prod/state/sweep.json` (2026-08-19 23:19): `ready 2000, done 3, todo 1997,
workers_running 0`. The three "done" shards are smoke shards run before the gate. The
sweeper is alive and configured `enabled: true`, but it holds on `wait_for_repair: true`
and `repair_gate_open: false` — the identity-repair sweep has not reported
`done == ready`, so the gate has never opened.

`docs/PROTOCOL.md` records, at some length, that **this gate is measurably not protecting
anything**: the repair sweeper is capped by its own `max_workers: 48`, its `idle_floor:
200` has never bound, and the partition has had 1,727–3,248 idle nodes throughout, so 48
VC nodes running alongside it cannot take a node from it. The gate was honoured anyway
because the brief stated it as a condition. It is one config key away from release
(`wait_for_repair: false` in `$NB/vcbon/prod/state/sweeper.conf.json`).

### 5.4 The incremental restart (`run1`) produced no completed voice, and 11 TB of orphans

A second, simpler implementation (`code/corpus/prod.py`, claims one whole *voice* rather
than a shard) was launched on 2026-08-18 on a 48-node array to start with the voices whose
repair had already landed — an explicit, documented deviation from the protocol's gate.

Its outcome, measured on disk:

- 192 voices claimed (`prod/run1/state/*.claim`), **0 `.DONE` markers, 0 `summary.json`**.
- 152 non-empty `cands.tar`, **11.15 TB total**, largest single file 95 GB.
- The array hit its 6 h walltime at 2026-08-18 21:39; one worker's log ends at
  25,208 of 40,195 takes for its voice.

So **11.15 TB of unverified partial output is sitting under
`$NB/vcbon/prod/run1/voices/`**, belonging to no completed voice. It has not been deleted
(nothing here is), but nothing reads it either. Two design differences explain why this
implementation is the weaker of the two: a voice (~40,000 takes) is far too large a unit
of work for a 6 h walltime — the shard implementation deliberately chose ~10,000 takes for
exactly this reason — and `prod.py` writes **uncompressed 48 kHz WAV** into the tar, where
`vcprod.py` writes 128–160 kbps mp3.

If the corpus run is resumed, use `code/corpus/shard/` (the 2,000-shard implementation),
not `code/corpus/prod.py`. The shard implementation has resumable claims, five per-shard
verification gates, per-pack fault isolation, accounting (`sources_done + sources_failed
== sources_in`) and a loss tolerance that fails a shard rather than silently shipping a
partial one. `prod.py` has none of these.

### 5.5 The identity-repair sweep was still running when this was written

At 2026-08-19 23:20: **426 of 499 voices complete** (85.4 %), **37 jobs in the queue**
(35 running, 2 pending behind a cluster maintenance reservation). It was **not**
cancelled and must not be. Anything in this document that depends on "all voices repaired"
— principally the corpus VC gate in §5.3 — is therefore still open.

A separate problem in that sweep, noted because it blocks release rather than the run:
only 10 of 428 verified voices have been published to Hugging Face; every attempt since
2026-08-16 10:49 fails with `403 Forbidden` on the xet-write-token endpoint. That looks
like a credential scope problem, not a pipeline bug.

### 5.6 Data-dependent crashes found late, and the defences added for them

Recorded because they are reproducibility-critical, and because both were invisible to
every smaller test:

- **Tokenizer off-by-one.** `vcengine.tokenize` pads tokens to the batch's returned width
  but clamps per-clip lengths to `samples // 640`; when that clamp lowers the batch
  *maximum*, s3gen's flow mask and the padded token width disagree by one frame and it
  raises. It needs a specific combination of clip lengths in one pack. The pilot, a
  320-source smoke and 17 stride-sampled 2,000-source shards across five voice families
  all missed it; **the first full-size shard hit it in under six minutes, and 2 of the
  first 4 full shards died of it**. Fixed in `vcprod.py` (trim tokens to `max(ln)`), not
  in `vcengine.py`, so the pilot's published artefacts stay byte-identical.
- **Clips too short for s3gen.** A 0.08 s take is 2 tokens; the flow decoder tries to pad
  (4,4) into a dimension of 4 and raises. Measured over 604,478 takes: **1.085 % of the
  corpus is under 1.28 s** (~218,000 takes), 0.204 % under 0.16 s, corpus minimum 0.08 s.
  Because packs are length-sorted, every shard's shortest takes land in its *first* pack —
  a first-minute kill, not a rare tail. Fixed by padding to 32 tokens for generation *and*
  SIDON and trimming back after restoration. Honest cost: SIDON is non-causal, so for
  ~1.1 % of the corpus the restoration sees a little trailing silence. The alternative was
  not converting those takes at all.
- **`sbatch --export` splits on commas**, so a comma-separated shard list silently
  delivered only its first element and three of four GPUs idled. Fixed by passing a JSON
  file path.
- **A claim-reaping bug** in the sweeper matched Slurm *job names* rather than job ids and
  reaped a live differently-named job's claims. Harmless when found; against a production
  worker it would have let two GPUs write the same shard concurrently, which per-shard
  verification could not have caught.

### 5.7 A correction that was made rather than edited away

`docs/PROTOCOL.md` contains one entry (04:45) that was **wrong** and one (05:00) that
corrects it: a traceback in a four-worker job's shared stderr was attributed to the wrong
shard on the basis of timing. The shard it was blamed on had in fact completed. The
correction is appended rather than substituted, which is the record's stated policy. Read
both.

---

## 6. How to reproduce each variant

Environment for everything: `$NB/env_mossaudio/bin/python`, with `source $NB/env.sh` in
every sbatch. `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` — all weights are already in
`$NB/hfcache`. On this filesystem a serial warm-up import before the timed section is
mandatory, and multi-GPU launches must be staggered by 150 s (concurrent
`import transformers` off `/e` puts every worker into uninterruptible sleep).

### 6.1 The four-arm comparison (both collage variants, side by side)

```bash
cd $NB/vc4arm
sbatch code/run.sbatch select20.py    # -> work/selection.json          (CPU)
sbatch code/run.sbatch prep.py        # -> work/clips/<cid>.npz, work/prep.json
sbatch code/run.sbatch arms.py        # -> out/arm{1,2,2b,3,4}/<cid>.mp3, out/wav/, out/arms.json
sbatch code/run.sbatch measure.py     # -> out/metrics.parquet, out/segments.parquet
```

`run.sbatch` requests 1 node, 1 GPU, 72 CPUs, 400 GB, 4 h, partition `booster`, account
`reformo`. The whole four-arm run took **~4 minutes of GPU work** for 20 collages.

Inputs: `$NB/collage_traj/collages.json` (the collage plan),
`$NB/collage_traj/emotion_norm_constants.json` (z constants for the emotion metric),
`$NB/vprof/vp500/<voice>/PPILOT2/cands-<shard>.tar` (source clips),
`$NB/vcbon/refs500/refs_prepared.tar` (variant A's conversion target).

Settings that define the comparison, all in `code/collage/`:

| setting | value | where |
|---|---|---|
| sample rate | 48 kHz throughout, VC internally at 16 kHz in / 24 kHz out | `arms.py` |
| internal-silence collapse | >400 ms → exactly 150 ms, 40 ms equal-power crossfade | `prep.py` |
| silence detector | 20 ms RMS frames, threshold `max(p95 − 35 dB, −60 dBFS)`, 30 ms pad | `prep.py` |
| per-clip level | −23 LUFS + own deviation from collage mean, clipped ±3 dB | `prep.py` |
| concatenation | 15 ms sqrt fades, 150 ms gap at every seam, −1 dBFS ceiling | `arms.py::concat` |
| final level, every arm | EBU R128 −23 LUFS, 0.95 ceiling | `arms.py::level` |
| variant A target | the voice's own prepared reference (`refs500`) | `arms.py`, arm 3 |
| variant B target | the collage's first snippet, −23 LUFS / 0.95 | `arms.py`, arms 2, 2b, 4 |
| seeds | `20260817 + n*1000`; arm 2b uses `+777`, arm 4 uses `+13` | `arms.py` |
| pack size for per-clip VC | 6, length-bucketed | `arms.py --pack` |
| TF32 | on inside `vc_whole`/`vc_clips`, asserted off before SIDON | `arms.py` |
| mp3 | 128 kbps | `arms.py` |

Measurement is **per segment**, then aggregated — the BUD-E whisper encoder truncates at
30 s and the collages run 21–65 s, so a whole-file score would silently be a score of the
first half. Seam timbre jump is measured against an **interior control** taken at
non-seam positions of the same segments; `seam − interior` is the reported excess.

### 6.2 The corpus best-of-N pipeline

Two implementations exist. **Use the shard one.**

```bash
# preparation (once)
$PY code/corpus/prep_refs.py                     # -> vcbon/refs500/refs_prepared.tar (500 voices)
$PY code/corpus/shard/vcindex.py                 # -> prod/index/shards.parquet (2,000 shards)
sbatch code/corpus/shard/vcnormpass.sbatch       # -> prod/index/norm_stats.json (frozen constants)
sbatch code/corpus/shard/vcsmoke.sbatch          # end-to-end on one shard, 5 verification gates

# the run
./code/corpus/shard/vcsweepd.sh                  # detached; tops workers up to 48 nodes
```

`vcsweep.py` holds on the repair gate, refuses to submit while `norm_stats.json` is missing
or unusable (without it every shard would silently fall back to within-set ranking, which
is the one error stored scores could not repair afterwards), reaps claims whose Slurm job
is gone, and publishes `prod/state/sweep.json`. Two controls, re-read every cycle in
`prod/state/sweeper.conf.json`: `enabled: false` stops everything, `wait_for_repair: false`
releases the gate.

Key settings: `--n-cand 4`, pack 8 length-sorted sources per forward pass, shard =
(voice, tar index) ≈ 10,064 samples, 4 GPUs/node, `--cpus-per-task=64`, 12 h walltime,
48 nodes, mp3 at 48 kHz. Storage key
`<target>/<source>/<gid>.c<NNN>.k<J>.mp3`, with `prov` carrying `source_run_dir` —
`audio_key` is **not** unique across runs and treating it as a key has already served the
wrong audio once on a published page.

Every row stores 156 columns: 40 EmoNet scores, 4 quality heads, 57 VoiceNet regressions,
both speaker similarities against both anchors, and **three** rewards side by side
(`reward_group` production, `reward_set`, `reward_minmax`) each with its own
`is_winner_*` column. Changing the reward later is a parquet query, not a rerun.

The alternative implementation `code/corpus/prod.py` + `prod.sbatch` claims one voice at a
time and is what produced `run1`. It is kept because it ran and because its output exists,
but see §5.4 before using it.

### 6.3 The pilot, if you want to re-derive the k-curves or the ablations

```bash
$PY code/corpus/build_sample.py     # -> vcbon/pilot/sources.tar (354 takes, 3 arms)
sbatch code/corpus/pilot.sbatch     # -> vcbon/pilot/vc_v1/{converted-*.tar, candidates.parquet}
$PY code/corpus/analyse.py          # -> vcbon/out/pilot_analysis.json
$PY code/corpus/ref_ablate.py       # -> vcbon/out/ref_ablate.{json,parquet}
$PY code/corpus/bench_gen.py        # -> vcbon/out/bench_gen.json  (N, M, bucketing, dtype sweeps)
```

The pilot's three arms are conversion targets, not pipelines: `self` (the take's own
voice), `nn` (the nearest other voice in the profile space), `far` (a distant one). They
exist to show how much of the ECAPA gain is "conversion works" versus "the target was
already close": the correlation between profile distance and post-conversion ECAPA is
**−0.755**.

---

## 7. Lineage — how this arrived at two variants

Reading order for anyone reconstructing the history:

1. **`$NB/vprof/code/vcprep.py` → `vcbest.py`** — the first VC work, one voice
   (`k325_age3_bg1`), 832 published top-1 takes, **best-of-3 seeds**, one clip at a time,
   **no SIDON**. Selection was by cosine of the 40-d EmoNet vector to the *original take's*
   stored vector — an emotion-*preservation* criterion, not a reward. It carried its own
   noise floor (`cos_self`: the original re-scored through the identical decode path
   against its own stored vector) so a best-of-3 gain smaller than `1 − cos_self` could be
   rejected. It also MD5s every seed's output and counts groups whose three seeds are not
   bit-distinct, rather than assuming `torch.manual_seed` did anything.
2. **`$NB/vprof/code/vpvc.py`** — post-hoc identity repair on one voice: convert the rank-0
   take of every group whose `spk_sim < 0.45`, keep it only if
   `sim_after > sim_before AND dnsmos_after ≥ dnsmos_before − 0.15`. Superseded, and it has
   a known defect its successor documents: no peak guard, so DNSMOS (which raises outside
   [−1, 1]) returned a silent `NaN` for every converted clip, because S3Gen output routinely
   peaks at 1.02–1.10.
3. **`$NB/code/chatterbox-voice-conversion/expressive_bestofn/`** — the method generalised
   and published, with a live demo and `METHOD.md`. Recommends **N = 8** and
   **rank-then-SIDON-the-winner**.
4. **`$NB/vcbon`** — the same method at corpus scale: 500 voices, N = 4, batched generation,
   SIDON on all four, a frozen-group z-score reward, everything stored. This reversed
   step 3's ordering, and §4.2 is the measurement that justifies the reversal.
5. **`$NB/collage_traj` → `$NB/vc4arm`** — VC applied to assembled trajectory collages.
   The listener report on `../acting_concats.html` raised the whole-file alternative;
   the four-arm study measured it; §3 is the result.
6. **`$NB/collage_traj2`** — the successor collage build, carrying the internal-silence fix
   from step 5.

`$NB/vprof/code/pvverify.py` is not a VC script but is worth knowing about: it reads a
packed release back, **decodes** every sampled mp3 and cross-checks duration, identity and
every numeric field against the parquet, exiting non-zero on any failure. It was written
after a 0.24 s clip was served for a row claiming 5.76 s — the kind of failure only
decoding catches.

---

## 8. Published pages this relates to

| page | what it shows |
|---|---|
| [`../vc_bestofn.html`](../vc_bestofn.html) | the one-GPU pilot: best-of-N curves, both target scenarios, costs |
| [`../vc_scenario1.html`](../vc_scenario1.html) | the corpus run's design and measured throughput |
| [`../acting_concats.html`](../acting_concats.html) | 9 scenes × 8 assemblies: raw / per-part VC / **whole-clip VC** / SIDON — where the two variants first appear side by side, for listening |
| [`../emotion_trajectories.html`](../emotion_trajectories.html) | the trajectory collages — **its audio carries the §5.1 internal-silence defect** |
| [`../trajectories.html`](../trajectories.html) | VoiceNet ladders and emotion arcs |

There is no four-arm page. See §5.2.
