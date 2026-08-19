# Four-arm before/after comparison — resume state

Working dir: `$NB/vc4arm/` (`code/`, `work/`, `out/`, `logs/`)
Env: `$NB/env_mossaudio/bin/python`. Source `$NB/env.sh` in every sbatch.

## Decisions already made (do not re-derive)

- **Arm 1 is REBUILT, not reused.** The brief's "reuse arm 1" and "all four arms
  see identical silence-fixed source" are incompatible; coordinator confirmed the
  rebuild. The existing `collage_traj/mp3/*.mp3` still carry the internal-silence
  defect and are kept only as a diagnostic.
- **20 collages selected** -> `work/selection.json`. 6 mirror emotion pairs in
  both directions (C000/C001 Anger<->Thankfulness, C002/C003 Sadness<->Elation,
  C006/C007 Fear<->Pride, C010/C011 Amusement<->Distress, C054/C055
  Contempt<->Affection, C058/C059 Shame<->Triumph) + 8 region-stratified fills
  (C075 C095 C074 C066 C037 C085 C039 C067). 833 s, 20 distinct voices, both
  ops, 11 en / 9 de, path lengths 3-6, 18 distinct (start,end) region cells.
- **Silence fix**: >400 ms internal quiet -> exactly 150 ms, 40 ms equal-power
  crossfade. Same detector as head/tail trim (20 ms RMS frames, p95-35 dB
  clamped to -60 dBFS). Implemented in `code/prep.py::collapse_internal`.
- **Clip recipe otherwise identical to `collage_traj/build.py`**: head/tail trim
  with 30 ms pad, per-clip EBU R128 to -23 LUFS + own deviation from the collage
  mean clipped to +-3 dB, 15 ms sqrt fades, 150 ms gap at every seam, -1 dBFS
  peak ceiling on the concatenation, 128 kbps mp3.
- **Reference for arms 2 and 4**: the collage's first snippet, EBU R128 -23 LUFS
  ceiling 0.95 (`vcengine.loudness_normalize`). Arm 3's per-clip target is the
  voice's own prepared reference from `vcbon/refs500/refs_prepared.tar` (the
  production pipeline's target). ECAPA/WavLM are measured against BOTH anchors
  so the reference asymmetry is visible rather than hidden.
- **TF32**: on around `s3gen.inference`, off around SIDON. Set at the call site.
- SIDON via `vcengine.Sidon` -> `mediathek_sidon/code/sidon_batch.py`.

## Stage status

- [x] `code/select20.py` -> `work/selection.json`
- [ ] `code/prep.py` -> `work/clips/<cid>.npz`, `work/prep.json` (silence audit)
- [ ] `code/arms.py` (GPU, 1 node 1 GPU) -> `out/arm{1,2,3,4}/<cid>.mp3` + wavs
- [ ] `code/measure.py` -> `out/metrics.parquet`
- [ ] `code/page.py` -> `vc4arm.html` + `vc4arm_audio/` on the gh mirror
- [ ] publish with `$NB/.gh_token`

## Cluster

48 vrepair running; repair 210/499. Take 1 node. Partition had ~1,180 idle.
