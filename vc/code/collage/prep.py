#!/usr/bin/env python
"""Source preparation for the four-arm comparison.

Reproduces `collage_traj/build.py`'s clip recipe exactly, with ONE addition:
internal silent stretches longer than 400 ms are collapsed to exactly 150 ms
with a 40 ms equal-power crossfade across the join.

The fix is applied here, BEFORE any arm is built, so all four arms see identical
source material and the comparison measures the conversion rather than the fix.

Outputs (per collage id):
  work/clips/<cid>.npz   clips_0..n at 48 kHz, post trim / collapse / loudness
  work/prep.json         every measurement, including the silence audit
"""
import os, sys, json, glob, tarfile, subprocess, time
import numpy as np
from multiprocessing import Pool

NB = '/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox'
CT = NB + '/collage_traj'
VP = NB + '/vprof/vp500'
FFMPEG = NB + '/bin/ffmpeg'
W = NB + '/vc4arm'
SR = 48000

# --- silence definition: identical to build.py's trim_silence -----------------
FRAME_MS, REL_DB, FLOOR_DB, PAD_MS = 20, 35.0, -60.0, 30
# --- internal-silence collapse ------------------------------------------------
MAX_INTERNAL_MS = 400.0     # stretches at or below this are left alone
TARGET_GAP_MS = 150.0       # collapsed stretches become exactly this
XFADE_MS = 40.0             # equal-power crossfade across the join


def decode(mp3bytes):
    p = subprocess.run([FFMPEG, '-v', 'quiet', '-i', 'pipe:0', '-f', 'f32le',
                        '-ac', '1', '-ar', str(SR), 'pipe:1'],
                       input=mp3bytes, capture_output=True)
    return np.frombuffer(p.stdout, dtype=np.float32).astype(np.float64)


def frame_db(x):
    fl = int(SR * FRAME_MS / 1000)
    nf = len(x) // fl
    if nf < 2:
        return None, fl, nf
    fr = x[:nf * fl].reshape(nf, fl)
    rms = np.sqrt((fr ** 2).mean(1) + 1e-20)
    return 20 * np.log10(rms), fl, nf


def trim_silence(x):
    """build.py's head-and-tail energy trim, byte-for-byte the same rule."""
    db, fl, nf = frame_db(x)
    if db is None:
        return x, 0.0, 0.0, False
    thr = max(np.percentile(db, 95) - REL_DB, FLOOR_DB)
    above = np.flatnonzero(db > thr)
    if len(above) == 0:
        return x, 0.0, 0.0, True
    pad = int(SR * PAD_MS / 1000)
    a = max(0, above[0] * fl - pad)
    b = min(len(x), (above[-1] + 1) * fl + pad)
    return x[a:b], a / SR, (len(x) - b) / SR, False


def find_internal_silences(x):
    """Runs of frames at/below the same threshold, bounded by speech on BOTH sides.

    Returns list of (start_sample, end_sample, duration_s) in ascending order.
    """
    db, fl, nf = frame_db(x)
    if db is None:
        return []
    thr = max(np.percentile(db, 95) - REL_DB, FLOOR_DB)
    quiet = db <= thr
    runs = []
    i = 0
    while i < nf:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < nf and quiet[j]:
            j += 1
        if i > 0 and j < nf:                      # bounded by speech on both sides
            runs.append((i * fl, j * fl, (j - i) * fl / SR))
        i = j
    return runs


def collapse_internal(x):
    """Collapse every internal quiet stretch > MAX_INTERNAL_MS to TARGET_GAP_MS.

    The retained gap is built from the stretch's own first and last material so
    the room tone is preserved, joined with a XFADE_MS equal-power crossfade.
    Net retained silence is exactly TARGET_GAP_MS.
    """
    runs = find_internal_silences(x)
    long_runs = [r for r in runs if r[2] * 1000.0 > MAX_INTERNAL_MS]
    if not long_runs:
        return x, [], [r[2] for r in runs]
    xf = int(SR * XFADE_MS / 1000)
    g = int(SR * (TARGET_GAP_MS + XFADE_MS) / 1000)     # so the result is exactly 150 ms
    gl, gr = g // 2, g - g // 2
    fo = np.sqrt(np.linspace(1, 0, xf))
    fi = np.sqrt(np.linspace(0, 1, xf))
    out, collapsed, prev = [], [], 0
    for (a, b, dur) in long_runs:
        if b - a <= g:                                   # nothing to gain
            continue
        left = x[prev:a + gl]
        right_head = x[b - gr:b - gr + xf]
        out.append(left[:-xf] if xf < len(left) else left[:0])
        n = min(xf, len(left), len(right_head))
        out.append(left[-n:] * fo[-n:] + right_head[:n] * fi[:n])
        prev = b - gr + n
        collapsed.append(dict(t_start_s=round(a / SR, 3), dur_before_s=round(dur, 3),
                              dur_after_s=round(TARGET_GAP_MS / 1000.0, 3),
                              removed_s=round(dur - TARGET_GAP_MS / 1000.0, 3)))
    out.append(x[prev:])
    return np.concatenate(out), collapsed, [r[2] for r in runs]


import pyloudnorm as pyln
METER = pyln.Meter(SR)


def prep_one(job):
    """Everything up to (but not including) concatenation, for one collage."""
    cid, comps = job
    clips, info = [], []
    for c in comps:
        x0 = decode(c['_bytes'])
        n0 = len(x0) / SR
        x, lead, tail, dead = trim_silence(x0)
        if dead or len(x) < SR * 1.0:
            return cid, None, 'silent-clip'
        dur_trim = len(x) / SR
        runs_before = find_internal_silences(x)
        x, collapsed, _ = collapse_internal(x)
        runs_after = find_internal_silences(x)
        try:
            lufs = float(METER.integrated_loudness(x))
        except Exception:
            lufs = -23.0
        if not np.isfinite(lufs):
            lufs = -23.0
        clips.append(x)
        info.append(dict(
            audio_key=c['audio_key'], state_name=c['state_name'], emotion=c['emotion'],
            condition=c['condition'], dur_raw=round(n0, 3), dur_trim=round(dur_trim, 3),
            dur_fixed=round(len(x) / SR, 3), lead_trim=round(lead, 3), tail_trim=round(tail, 3),
            lufs_before=round(lufs, 2),
            sil_before=[round(r[2], 3) for r in runs_before],
            sil_after=[round(r[2], 3) for r in runs_after],
            n_long_before=sum(1 for r in runs_before if r[2] * 1000 > MAX_INTERNAL_MS),
            n_long_after=sum(1 for r in runs_after if r[2] * 1000 > MAX_INTERNAL_MS),
            max_sil_before=round(max([r[2] for r in runs_before], default=0.0), 3),
            max_sil_after=round(max([r[2] for r in runs_after], default=0.0), 3),
            collapsed=collapsed, removed_s=round(sum(d['removed_s'] for d in collapsed), 3)))
    # build.py's level design: each clip to -23 LUFS + its own deviation from the
    # collage mean, clipped to +-3 dB.  Kept identical so arm 1 stays the same
    # soundscape it was, minus the silence defect.
    mean_l = float(np.mean([i['lufs_before'] for i in info]))
    for x, i in zip(clips, info):
        dev = float(np.clip(i['lufs_before'] - mean_l, -3.0, 3.0))
        target = -23.0 + dev
        gain = 10 ** ((target - i['lufs_before']) / 20)
        x *= gain
        i['lufs_after'] = round(target, 2)
        i['gain_db'] = round(float(20 * np.log10(gain)), 2)
    np.savez(f'{W}/work/clips/{cid}.npz',
             **{f'c{k}': x.astype(np.float32) for k, x in enumerate(clips)})
    return cid, info, 'ok'


if __name__ == '__main__':
    t0 = time.time()
    os.makedirs(W + '/work/clips', exist_ok=True)
    sel = json.load(open(W + '/work/selection.json'))
    C = {x['id']: x for x in json.load(open(CT + '/collages.json'))}
    plan = [C[i] for i in sel]

    needed = {}
    for p in plan:
        for c in p['comps']:
            needed.setdefault((p['voice'], c['shard']), set()).add(c['audio_key'])

    def fetch(args):
        (voice, shard), keys = args
        got = {}
        with tarfile.open(f'{VP}/{voice}/PPILOT2/cands-{shard:03d}.tar') as tf:
            for m in tf:
                if m.name in keys:
                    got[m.name] = tf.extractfile(m).read()
                    if len(got) == len(keys):
                        break
        return (voice, shard), got

    print(f'extracting {sum(len(v) for v in needed.values())} clips from '
          f'{len(needed)} tars...', flush=True)
    blobs = {}
    with Pool(12) as pool:
        for k, got in pool.imap_unordered(fetch, list(needed.items())):
            blobs[k] = got
    print(f'  extracted in {time.time()-t0:.0f}s', flush=True)

    jobs = []
    for p in plan:
        comps = []
        for c in p['comps']:
            cc = dict(c)
            cc['_bytes'] = blobs[(p['voice'], c['shard'])][c['audio_key']]
            comps.append(cc)
        jobs.append((p['id'], comps))

    out = {}
    with Pool(10) as pool:
        for cid, info, st in pool.imap_unordered(prep_one, jobs):
            out[cid] = dict(status=st, clips=info)
            print(f'  {cid} {st}', flush=True)

    meta = dict(params=dict(frame_ms=FRAME_MS, rel_db=REL_DB, floor_db=FLOOR_DB,
                            pad_ms=PAD_MS, max_internal_ms=MAX_INTERNAL_MS,
                            target_gap_ms=TARGET_GAP_MS, xfade_ms=XFADE_MS, sr=SR),
                collages=out)
    json.dump(meta, open(W + '/work/prep.json', 'w'), indent=1)

    # ------------------------------------------------------- silence audit ---
    allc = [d for v in out.values() for c in v['clips'] for d in c['collapsed']]
    survive = sum(c['n_long_after'] for v in out.values() for c in v['clips'])
    nclips = sum(len(v['clips']) for v in out.values())
    affected = sum(1 for v in out.values() for c in v['clips'] if c['n_long_before'])
    colls_aff = sum(1 for v in out.values() if any(c['n_long_before'] for c in v['clips']))
    print(f'\n=== internal-silence audit ===')
    print(f'clips {nclips} in {len(out)} collages; {affected} clips '
          f'({100*affected/nclips:.1f} %) had a >400 ms internal stretch, '
          f'in {colls_aff}/{len(out)} collages')
    print(f'stretches collapsed: {len(allc)}')
    if allc:
        b = np.array([d['dur_before_s'] for d in allc])
        print(f'  original length s: min {b.min():.3f} p25 {np.percentile(b,25):.3f} '
              f'med {np.median(b):.3f} p75 {np.percentile(b,75):.3f} max {b.max():.3f}')
        print(f'  total removed    : {sum(d["removed_s"] for d in allc):.2f} s')
    print(f'>400 ms internal stretches surviving the fix: {survive}')
    print(f'\ndone in {time.time()-t0:.0f}s')
