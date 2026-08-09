# MOSS voice profiles — conditioned speech corpus

**[→ Browse the site](https://laion-ai.github.io/moss-voice-profiles/)**

For one reference voice: every emotion at four intensity/containment settings, every VoiceNet
dimension at four levels, edge cases and style variants — in **English and German from the same
sentence**, so the pairs can train a speech-to-speech model that preserves speaker and emotion.

- **[Design manual](https://laion-ai.github.io/moss-voice-profiles/manual.html)** — the full
  condition matrix, which adapter at which merge strength for which emotion/dimension, prompt
  templates, ranking formulas, metadata schema, safety gate, compute estimates.
- **[Demo grid](https://laion-ai.github.io/moss-voice-profiles/demo_grid.html)** — 808 conditions
  for one voice, generated twice (standard vs 25 % Mediathek merge), side by side with the
  reference. 1,617 clips.
- **[Execution protocol](https://laion-ai.github.io/moss-voice-profiles/protocol.html)** — every
  step, adapter, prompt and setting actually used, with per-dimension results.

## Scale

808 groups/voice · 323,200 generations per profile · 646,917 candidates stored · 152 GPU-hours ·
2,126 generations/GPU-hour.

Extrapolated: **100 voices ≈ 1,226 GPU-h · 1,000 ≈ 12,260 · 6,000 ≈ 73,600 GPU-h and ~17 TB.**
6,000 at the full matrix is not feasible as one run; see the manual for the proposed subsets.

## Headline findings

- **German conditioning holds the speaker far less well than English** — mean speaker similarity
  **0.578 (EN) vs 0.376 (DE)** over 323,200 candidates. This is the central obstacle for the
  DE↔EN goal.
- **Intensity, not containment, breaks the voice clone**: intense conditions 0.088 lower, with
  19.4 % below the usable floor vs 0.0 % for moderate.
- **The 25 % Mediathek merge is near-null**: +0.008 speaker similarity, no change to composite,
  WER, genuineness, blend or emotion strength.
- **Voice conversion is not free where it is most needed**: on 271 badly drifted clips it raised
  similarity 0.381 → 0.623 on all of them but cost −0.109 DNSMOS, and only 12 % passed the
  published keep rule.

## Limitations

No human listening study — all numbers come from an automatic sensor stack that agrees with a
listener at only about ρ ≈ +0.21 on spoken material. The four-level VoiceNet ladder, the A/B/C/D
stage directions and the domain labels are this project's own constructions, documented as such in
the manual.

Base model: [`laion/moss-tts-local-transformer-4.55b-voice-acting-v2`](https://huggingface.co/laion/moss-tts-local-transformer-4.55b-voice-acting-v2).
