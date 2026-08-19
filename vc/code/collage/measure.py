#!/usr/bin/env python
"""Measure every arm: speaker similarity, target emotion strength, quality, seams.

Everything is measured PER SEGMENT and then aggregated, because the BUD-E
whisper encoder truncates at 30 s and the collages run 21-65 s: a whole-file
score would silently be a score of the first half.  Per-segment is also the only
level at which "target emotion strength" is defined, since each segment has its
own target state.

Seams get two measures and a real noise floor:
  * within-collage speaker spread - mean pairwise ECAPA cosine DISTANCE between
    the collage's own segments.  This is "does it sound like one person all the
    way through", which is the thing a single pass over the finished audio is
    supposed to fix.
  * seam timbre jump - distance between the MFCC profile of the 200 ms before a
    seam and the 200 ms after it, in units of the corpus-wide MFCC sd.  Its
    control is the identical measure taken at interior positions of the same
    segments, where no seam exists.  seam minus interior is the seam EXCESS,
    and that is the number that means something.
"""
import os, sys, json, time, tarfile, argparse
import numpy as np
import torch, torchaudio

NB = '/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox'
W = NB + '/vc4arm'
sys.path.insert(0, NB + '/vcbon/code')
sys.path.insert(0, NB)
import vcengine as E

SR, DEV = 48000, 'cuda:0'
ARMS = ['arm1', 'arm2', 'arm3', 'arm4', 'arm2b']
WIN_MS = 200.0
FADE_MS = 15.0

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=W + '/out')
a = ap.parse_args()

arms = json.load(open(f'{a.out}/arms.json'))
COLL = {x['id']: x for x in json.load(open(NB + '/collage_traj/collages.json'))}
CONST = json.load(open(NB + '/collage_traj/emotion_norm_constants.json'))
E2D = lambda e: 'Jealousy_&_Envy' if e == 'Jealousy_and_Envy' else e

# ------------------------------------------------------------------ models ---
t0 = time.time()
sp = E.SpeakerSim(DEV, savedir=f'{NB}/vcbon/ecapa_ckpt', spk_emb_path=f'{NB}/vprof/idloop/code')
print(f'[measure] speaker models {time.time()-t0:.0f}s  wavlm={"yes" if sp.orange else "NO"}', flush=True)
t0 = time.time()
sys.path.insert(0, NB)
from pp_scores_fast import FastScorer
fs = FastScorer('cuda')
print(f'[measure] scorer {time.time()-t0:.0f}s', flush=True)

refs = {}
with tarfile.open(NB + '/vcbon/refs500/refs_prepared.tar') as tf:
    want = {COLL[i]['voice'] for i in arms}
    for m in tf:
        c = os.path.basename(m.name).replace('.prep.mp3', '')
        if c in want:
            refs[c] = E.decode_audio_bytes(tf.extractfile(m).read())

MFCC = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=13,
                                  melkwargs=dict(n_fft=400, hop_length=160, n_mels=40)).to(DEV)


def to16(x):
    return torchaudio.functional.resample(torch.as_tensor(np.asarray(x, np.float32)),
                                          SR, 16000).numpy()


def mfcc_profile(w16):
    """Mean MFCC vector of a short window (13-dim)."""
    if len(w16) < 400:
        return None
    t = torch.as_tensor(np.asarray(w16, np.float32), device=DEV)[None]
    with torch.inference_mode():
        m = MFCC(t)[0]                       # (13, frames)
    return m.mean(1).float().cpu().numpy()


# ------------------------------------------------------- gather every window --
seg_wavs, seg_key = [], []
prof_raw, prof_key = [], []
gapinfo = {}

for cid, rec in arms.items():
    z = np.load(f'{a.out}/wav/{cid}.npz')
    for arm in ARMS:
        if arm not in rec['arms']:
            continue
        x = z[arm].astype(np.float32)
        b = rec['arms'][arm]['bounds']
        b = [(max(0, int(s)), min(len(x), int(e))) for s, e in b]
        for k, (s, e) in enumerate(b):
            if e - s < SR * 0.5:
                continue
            seg_wavs.append(to16(x[s:e]))
            seg_key.append((cid, arm, k))
        # ---- seam / interior MFCC windows
        wn = int(SR * WIN_MS / 1000)
        fd = int(SR * FADE_MS / 1000)
        for k in range(len(b) - 1):
            pre = x[max(b[k][0], b[k][1] - fd - wn): b[k][1] - fd]
            post = x[b[k + 1][0] + fd: min(b[k + 1][1], b[k + 1][0] + fd + wn)]
            prof_raw.append(to16(pre)); prof_key.append((cid, arm, 'seam_pre', k))
            prof_raw.append(to16(post)); prof_key.append((cid, arm, 'seam_post', k))
        for k, (s, e) in enumerate(b):
            n = e - s
            if n < 4 * wn:
                continue
            for j, frac in enumerate((0.34, 0.62)):
                c = s + int(n * frac)
                prof_raw.append(to16(x[c - wn:c])); prof_key.append((cid, arm, f'int{j}_pre', k))
                prof_raw.append(to16(x[c:c + wn])); prof_key.append((cid, arm, f'int{j}_post', k))
        # ---- what the arm did to the 150 ms inter-clip pauses
        rms_gap, rms_seg = [], []
        for k in range(len(b) - 1):
            g = x[b[k][1]: b[k + 1][0]]
            if len(g):
                rms_gap.append(float(np.sqrt((g.astype(np.float64) ** 2).mean() + 1e-20)))
        for s, e in b:
            rms_seg.append(float(np.sqrt((x[s:e].astype(np.float64) ** 2).mean() + 1e-20)))
        if rms_gap:
            gapinfo[(cid, arm)] = dict(
                gap_db=round(float(20 * np.log10(np.mean(rms_gap))), 2),
                seg_db=round(float(20 * np.log10(np.mean(rms_seg))), 2),
                gap_below_seg_db=round(float(20 * np.log10(np.mean(rms_seg) / np.mean(rms_gap))), 2))

print(f'[measure] {len(seg_wavs)} segments, {len(prof_raw)} mfcc windows', flush=True)

# --------------------------------------------------------------- embeddings --
t0 = time.time()
EC = sp.ecapa_emb(seg_wavs, max_batch=32).cpu().numpy()
TB = sp.orange_emb(seg_wavs, max_batch=16)
TB = TB.cpu().numpy() if TB is not None else None
print(f'[measure] speaker embeddings {time.time()-t0:.0f}s', flush=True)

t0 = time.time()
SCO = []
for s in range(0, len(seg_wavs), 12):
    SCO.extend(fs.score_batch([torch.as_tensor(w) for w in seg_wavs[s:s + 12]]))
    if s % 120 == 0:
        print(f'   scored {s}/{len(seg_wavs)}', flush=True)
print(f'[measure] scoring {time.time()-t0:.0f}s', flush=True)

# reference embeddings: the voice's prepared reference AND the first snippet
ref_ec, ref_tb, first_ec, first_tb = {}, {}, {}, {}
rl, rk = [], []
for cid, rec in arms.items():
    v = rec['voice']
    if v not in ref_ec:
        w, sr = refs[v]
        rl.append(torchaudio.functional.resample(torch.as_tensor(w), sr, 16000).numpy())
        rk.append(('ref', v))
    z = np.load(f'{a.out}/wav/{cid}.npz')
    rl.append(to16(z['ref_first'])); rk.append(('first', cid))
RE = sp.ecapa_emb(rl, max_batch=32).cpu().numpy()
RT = sp.orange_emb(rl, max_batch=16)
RT = RT.cpu().numpy() if RT is not None else None
for i, (kind, key) in enumerate(rk):
    (ref_ec if kind == 'ref' else first_ec)[key] = RE[i]
    if RT is not None:
        (ref_tb if kind == 'ref' else first_tb)[key] = RT[i]

# ------------------------------------------------------------ mfcc profiles --
P = [mfcc_profile(w) for w in prof_raw]
ok = [i for i, p in enumerate(P) if p is not None]
Pm = np.stack([P[i] for i in ok])
psd = Pm.std(0) + 1e-6
prof = {}
for i in ok:
    prof[prof_key[i]] = P[i] / psd

# ------------------------------------------------------------------ tables ---
segidx = {k: i for i, k in enumerate(seg_key)}
rows = []
for cid, rec in arms.items():
    voice, comps = rec['voice'], COLL[cid]['comps']
    for arm in ARMS:
        if arm not in rec['arms']:
            continue
        ks = [k for (c, ar, k) in seg_key if c == cid and ar == arm]
        if not ks:
            continue
        ec = np.stack([EC[segidx[(cid, arm, k)]] for k in ks])
        tb = np.stack([TB[segidx[(cid, arm, k)]] for k in ks]) if TB is not None else None
        sc = [SCO[segidx[(cid, arm, k)]] for k in ks]

        ecapa_ref = float(np.mean(ec @ ref_ec[voice]))
        ecapa_first = float(np.mean(ec @ first_ec[cid]))
        wavlm_ref = float(np.mean(tb @ ref_tb[voice])) if tb is not None else float('nan')
        wavlm_first = float(np.mean(tb @ first_tb[cid])) if tb is not None else float('nan')

        # target emotion strength, z-scored on the corpus constants
        emo_z, emo_raw = [], []
        for k, s in zip(ks, sc):
            d = E2D(comps[k]['emotion'])
            v = float(s['emonet'][d])
            emo_raw.append(v)
            emo_z.append((v - CONST['mean'][d]) / CONST['sd'][d])
        qual = float(np.mean([s['quality']['overall_quality'] for s in sc]))
        sq = float(np.mean([s['quality']['speech_quality'] for s in sc]))
        genu = float(np.mean([s['genuineness_0_6'] for s in sc]))

        # seam consistency
        G = ec @ ec.T
        iu = np.triu_indices(len(ks), 1)
        spread = float(1.0 - G[iu].mean()) if len(ks) > 1 else float('nan')

        def jump(tagpre, tagpost):
            out = []
            for k in range(len(comps)):
                p, q = prof.get((cid, arm, tagpre, k)), prof.get((cid, arm, tagpost, k))
                if p is not None and q is not None:
                    out.append(float(np.linalg.norm(p - q)))
            return out

        seam = jump('seam_pre', 'seam_post')
        interior = jump('int0_pre', 'int0_post') + jump('int1_pre', 'int1_post')
        rows.append(dict(
            cid=cid, arm=arm, voice=voice, n_seg=len(ks),
            ecapa_ref=ecapa_ref, ecapa_first=ecapa_first,
            wavlm_ref=wavlm_ref, wavlm_first=wavlm_first,
            emo_z=float(np.mean(emo_z)), emo_raw=float(np.mean(emo_raw)),
            qual=qual, speech_q=sq, genuineness=genu,
            spk_spread=spread,
            seam_jump=float(np.mean(seam)) if seam else float('nan'),
            interior_jump=float(np.mean(interior)) if interior else float('nan'),
            seam_excess=(float(np.mean(seam)) - float(np.mean(interior)))
            if seam and interior else float('nan'),
            n_seam=len(seam), n_interior=len(interior),
            dur_s=rec['arms'][arm]['dur_s'],
            **(gapinfo.get((cid, arm), {}))))

import pandas as pd
df = pd.DataFrame(rows)
df.to_parquet(f'{a.out}/metrics.parquet')

# per-segment detail, for the page and for anyone who wants to re-cut the tables
seg = []
for i, (cid, arm, k) in enumerate(seg_key):
    comps = COLL[cid]['comps']
    d = E2D(comps[k]['emotion'])
    seg.append(dict(cid=cid, arm=arm, seg=k, state=comps[k]['state_name'],
                    emotion=comps[k]['emotion'],
                    emo_raw=float(SCO[i]['emonet'][d]),
                    emo_z=float((SCO[i]['emonet'][d] - CONST['mean'][d]) / CONST['sd'][d]),
                    qual=float(SCO[i]['quality']['overall_quality']),
                    ecapa_ref=float(EC[i] @ ref_ec[arms[cid]['voice']]),
                    ecapa_first=float(EC[i] @ first_ec[cid])))
pd.DataFrame(seg).to_parquet(f'{a.out}/segments.parquet')

print(f'\n[measure] wrote {len(df)} arm-rows, {len(seg)} segment-rows')
M = df.groupby('arm')[['ecapa_ref', 'ecapa_first', 'wavlm_ref', 'wavlm_first',
                       'emo_z', 'qual', 'spk_spread', 'seam_jump',
                       'interior_jump', 'seam_excess']].mean()
print(M.round(4).to_string())
print('=== MEASURE DONE ===')
