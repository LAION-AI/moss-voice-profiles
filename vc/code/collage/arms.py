#!/usr/bin/env python
"""Build the four arms for the 20 selected trajectories.

  arm1  raw          concatenation of the silence-fixed clips
  arm2  concat->VC   arm1 run through Chatterbox VC as ONE file
  arm3  VC->SIDON->concat   each clip converted and restored, then concatenated
  arm4  arm3->VC     arm3 run through Chatterbox VC as ONE file

  arm2b a second seed of arm2, purely to give every metric a run-to-run noise
        floor (the s3gen flow decoder is stochastic).

Conversion targets
  arms 2 and 4 : the collage's FIRST SNIPPET, EBU R128 -23 LUFS / ceiling 0.95
  arm 3        : the voice's own prepared reference (vcbon/refs500), i.e. the
                 target the production pipeline uses

Every finished soundscape is levelled to -23 LUFS / ceiling 0.95 so the four
players on the page are directly A/B-able and no arm wins on loudness.
"""
import os, sys, json, time, tarfile, argparse
import numpy as np
import torch, torchaudio

NB = '/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox'
W = NB + '/vc4arm'
sys.path.insert(0, NB + '/vcbon/code')
sys.path.insert(0, NB)
import vcengine as E

SR = 48000
DEV = 'cuda:0'
FADE_MS, GAP_MS, PEAK_DBFS = 15.0, 150.0, -1.0
TARGET_LUFS, CEIL = -23.0, 0.95

ap = argparse.ArgumentParser()
ap.add_argument('--ids', default='')
ap.add_argument('--out', default=W + '/out')
ap.add_argument('--pack', type=int, default=6)
a = ap.parse_args()

sel = json.load(open(W + '/work/selection.json'))
if a.ids:
    sel = [i for i in sel if i in a.ids.split(',')]
prep = json.load(open(W + '/work/prep.json'))
COLL = {x['id']: x for x in json.load(open(NB + '/collage_traj/collages.json'))}
for d in ('arm1', 'arm2', 'arm3', 'arm4', 'arm2b', 'wav'):
    os.makedirs(f'{a.out}/{d}', exist_ok=True)


# ------------------------------------------------------------------- utils ---
def concat(clips):
    """build.py's join: 15 ms sqrt fades, a fixed 150 ms pause at every seam."""
    fade = int(SR * FADE_MS / 1000)
    gap = int(SR * GAP_MS / 1000)
    f = np.sqrt(np.linspace(0, 1, fade))
    out, bounds, t = [], [], 0
    for k, x in enumerate(clips):
        y = np.asarray(x, np.float64).copy()
        y[:fade] *= f
        y[-fade:] *= f[::-1]
        out.append(y)
        bounds.append((t, t + len(y)))
        t += len(y)
        if k < len(clips) - 1:
            out.append(np.zeros(gap))
            t += gap
    sig = np.concatenate(out)
    pk = np.abs(sig).max()
    if pk > 10 ** (PEAK_DBFS / 20):
        sig *= (10 ** (PEAK_DBFS / 20)) / pk
    return sig.astype(np.float32), bounds


def level(x, tag):
    """EBU R128 to -23 LUFS with a 0.95 ceiling, and a record of what it did.

    Also records what -18 LUFS WOULD have done, because the project's older -18
    target is a silent no-op on peak-normalised (SIDON) material and that claim
    is worth carrying evidence for rather than repeating.
    """
    y, pre, gain, post, clipped = E.loudness_normalize(x, SR, TARGET_LUFS, CEIL)
    pk = float(np.abs(np.asarray(x, np.float32)).max())
    g18 = (-18.0 - pre) if np.isfinite(pre) else 0.0
    headroom_db = 20 * np.log10(CEIL / max(pk, 1e-9))
    return y, dict(tag=tag, lufs_pre=round(float(pre), 2), gain_db=round(float(gain), 2),
                   lufs_post=round(float(post), 2), ceiling_clipped=bool(clipped),
                   peak_pre=round(pk, 4),
                   gain_needed_18=round(float(g18), 2),
                   headroom_to_ceiling_db=round(float(headroom_db), 2),
                   noop_at_18=bool(g18 > headroom_db))


def to16(x):
    return torchaudio.functional.resample(torch.as_tensor(np.asarray(x, np.float32)),
                                          SR, 16000).numpy()


@torch.inference_mode()
def vc_whole(vc, wav48, ref_dict, seed):
    """One Chatterbox VC pass over a whole soundscape.  TF32 ON (measured free)."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        w16 = to16(wav48)
        tok, ln = E.tokenize(vc, [w16], DEV)
        tok = tok[:, :int(ln.max())]
        y = E.generate_batch(vc, tok, ln, 1, ref_dict=ref_dict, seed=seed)
        y = y.reshape(y.shape[0], -1)[0].float().cpu()
        y48 = torchaudio.functional.resample(y, E.OUT_SR, SR).numpy()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return y48.astype(np.float32), len(w16) / 16000.0


@torch.inference_mode()
def vc_clips(vc, clips48, ref_dict, seed, pack):
    """Per-clip VC, length-bucketed.  Returns 24 kHz outputs in input order."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        w16 = [to16(c) for c in clips48]
        order = sorted(range(len(w16)), key=lambda i: len(w16[i]))
        out = [None] * len(w16)
        for s in range(0, len(order), pack):
            idx = order[s: s + pack]
            tok, ln = E.tokenize(vc, [w16[i] for i in idx], DEV)
            tok = tok[:, :int(ln.max())]
            y = E.generate_batch(vc, tok, ln, 1, ref_dict=ref_dict, seed=seed + s)
            y = y.reshape(y.shape[0], -1).float().cpu().numpy()
            for k, i in enumerate(idx):
                n = int(round(len(w16[i]) / 16000 * E.OUT_SR))
                out[i] = y[k][:n]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return out


def relevel_to(x, sr, target_lufs):
    pre = E.measure_lufs(x, sr)
    if not np.isfinite(pre):
        return np.asarray(x, np.float32)
    return (np.asarray(x, np.float32) * 10 ** ((target_lufs - pre) / 20)).astype(np.float32)


# ------------------------------------------------------------------ models ---
t0 = time.time()
vc = E.load_vc(DEV)
print(f'[arms] VC loaded {time.time()-t0:.0f}s', flush=True)
t0 = time.time()
sd = E.Sidon(DEV, threads=16)
print(f'[arms] SIDON loaded {time.time()-t0:.0f}s', flush=True)

refs = {}
with tarfile.open(NB + '/vcbon/refs500/refs_prepared.tar') as tf:
    want = {COLL[i]['voice'] for i in sel}
    for m in tf:
        c = os.path.basename(m.name).replace('.prep.mp3', '')
        if c in want:
            refs[c] = E.decode_audio_bytes(tf.extractfile(m).read())
print(f'[arms] {len(refs)}/{len(want)} prepared references', flush=True)

# ------------------------------------------------------------------- run -----
report, T = {}, dict(vc_whole=0.0, vc_clips=0.0, sidon=0.0, io=0.0)
for n, cid in enumerate(sel):
    tc = time.time()
    z = np.load(f'{W}/work/clips/{cid}.npz')
    clips = [z[f'c{k}'].astype(np.float64) for k in range(len(z.files))]
    info = prep['collages'][cid]['clips']
    voice = COLL[cid]['voice']
    seed = 20260817 + n * 1000

    rec = dict(cid=cid, voice=voice, n_clips=len(clips), level={}, arms={})

    # ---- arm 1 : raw concatenation of the silence-fixed clips
    a1, b1 = concat(clips)
    a1, lv = level(a1, 'arm1')
    rec['level']['arm1'] = lv
    rec['arms']['arm1'] = dict(dur_s=round(len(a1) / SR, 3),
                               bounds=[[int(x), int(y)] for x, y in b1])

    # ---- the conversion target for the whole-soundscape passes
    ref_first, lvr = level(clips[0], 'ref_first_snippet')
    rec['level']['ref_first_snippet'] = lvr
    rec['ref_first_snippet_s'] = round(len(ref_first) / SR, 3)
    rd_first = E.set_target_from_wav(vc, ref_first, SR, peak_norm=None)
    rd_first = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in rd_first.items()}

    # ---- arm 2 : one VC pass over the finished soundscape
    t = time.time()
    a2, _ = vc_whole(vc, a1, rd_first, seed)
    T['vc_whole'] += time.time() - t
    sc = len(a2) / len(a1)
    a2, lv = level(a2, 'arm2')
    rec['level']['arm2'] = lv
    rec['arms']['arm2'] = dict(dur_s=round(len(a2) / SR, 3), len_ratio=round(sc, 4),
                               bounds=[[int(x * sc), int(y * sc)] for x, y in b1])

    # ---- arm 2b : the same thing again, different seed -> noise floor
    t = time.time()
    a2b, _ = vc_whole(vc, a1, rd_first, seed + 777)
    T['vc_whole'] += time.time() - t
    scb = len(a2b) / len(a1)
    a2b, _ = level(a2b, 'arm2b')
    rec['arms']['arm2b'] = dict(dur_s=round(len(a2b) / SR, 3), len_ratio=round(scb, 4),
                                bounds=[[int(x * scb), int(y * scb)] for x, y in b1])

    # ---- arm 3 : per-clip VC -> SIDON -> concat, target = the voice's own ref
    rw, rsr = refs[voice]
    rd_self = E.set_target_from_wav(vc, rw, rsr, peak_norm=None)
    rd_self = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in rd_self.items()}
    t = time.time()
    conv24 = vc_clips(vc, clips, rd_self, seed, a.pack)
    T['vc_clips'] += time.time() - t
    t = time.time()
    # SIDON: TF32 must stay OFF here (~15 dB); it is off by default and vc_* restore it
    assert not torch.backends.cuda.matmul.allow_tf32
    rest48 = sd.restore(conv24, E.OUT_SR, max_items=16)
    T['sidon'] += time.time() - t
    # SIDON hands back peak-normalised audio, which throws away the collage's
    # level design; put each clip back on its intended -23+dev target.
    fixed = [relevel_to(x, SR, info[k]['lufs_after']) for k, x in enumerate(rest48)]
    a3, b3 = concat(fixed)
    a3, lv = level(a3, 'arm3')
    rec['level']['arm3'] = lv
    rec['arms']['arm3'] = dict(dur_s=round(len(a3) / SR, 3),
                               bounds=[[int(x), int(y)] for x, y in b3])

    # ---- arm 4 : a second VC pass over arm 3's finished soundscape
    t = time.time()
    a4, _ = vc_whole(vc, a3, rd_first, seed + 13)
    T['vc_whole'] += time.time() - t
    sc4 = len(a4) / len(a3)
    a4, lv = level(a4, 'arm4')
    rec['level']['arm4'] = lv
    rec['arms']['arm4'] = dict(dur_s=round(len(a4) / SR, 3), len_ratio=round(sc4, 4),
                               bounds=[[int(x * sc4), int(y * sc4)] for x, y in b3])

    t = time.time()
    np.savez(f'{a.out}/wav/{cid}.npz', arm1=a1, arm2=a2, arm3=a3, arm4=a4, arm2b=a2b,
             ref_first=ref_first.astype(np.float32))
    for tag, y in (('arm1', a1), ('arm2', a2), ('arm3', a3), ('arm4', a4), ('arm2b', a2b)):
        E.encode_mp3(y, SR, f'{a.out}/{tag}/{cid}.mp3', bitrate=128000)
    T['io'] += time.time() - t

    report[cid] = rec
    print(f'[{n+1}/{len(sel)}] {cid} {voice:18s} {len(clips)} clips  '
          f'a1 {rec["arms"]["arm1"]["dur_s"]:5.1f}s  a2 {rec["arms"]["arm2"]["dur_s"]:5.1f}s '
          f'a3 {rec["arms"]["arm3"]["dur_s"]:5.1f}s  a4 {rec["arms"]["arm4"]["dur_s"]:5.1f}s  '
          f'{time.time()-tc:.0f}s', flush=True)
    json.dump(report, open(f'{a.out}/arms.json', 'w'), indent=1)

print(f'\n[arms] stage seconds: {json.dumps({k: round(v,1) for k, v in T.items()})}')
print('=== ARMS DONE ===')
