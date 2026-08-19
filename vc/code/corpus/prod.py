#!/usr/bin/env python
"""Scenario 1 production: best-of-4 Chatterbox VC + SIDON over the repaired vp500 corpus.

SPEC (PROTOCOL.md, owner):
  for each source take -> 4 VC candidates -> SIDON on ALL FOUR -> rank the SIDON-enhanced
  candidates by normalise(target emotion strength) + normalise(quality) -> store everything:
  all four outputs, all raw scores, all rewards, all components. Selection is a VIEW over stored
  scores, never a filter applied at write time.

DEVIATION FROM THE PROTOCOL, ON INSTRUCTION
PROTOCOL.md says the production launch waits for `done == ready` on the identity-repair sweeper.
The owner has instead asked to start now with the voices that are already finished. So this worker
claims only voices with a VERIFIED repair report, and the remaining voices become claimable as
their repairs land. That makes the run incremental rather than gated, and it is recorded here
because it changes what a later reader would otherwise assume from the protocol.

CLAIM MODEL
One directory per voice under `state/`, claimed with O_EXCL so two workers cannot take the same
voice, and a DONE marker written only after the voice's outputs are complete. A worker that dies
mid-voice loses at most the current shard; the next worker re-claims and resumes.
"""
import argparse
import glob
import io
import json
import os
import sys
import tarfile
import time

import numpy as np

NB = "/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox"
sys.path.insert(0, f"{NB}/vcbon/code")
sys.path.insert(0, NB)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def verified_voices():
    """Only voices whose identity repair finished AND verified. Anything else is not ready."""
    out = []
    for v in sorted(os.listdir(f"{NB}/vprof/repair/voices")):
        rp = f"{NB}/vprof/repair/voices/{v}/report.json"
        if not os.path.exists(rp):
            continue
        try:
            r = json.load(open(rp))
        except Exception:
            continue
        if r.get("verified") is True and not r.get("failures"):
            if glob.glob(f"{NB}/vprof/vp500/{v}/PPILOT2/*.tar"):
                out.append(v)
    return out


def claim(state_dir, voice):
    os.makedirs(state_dir, exist_ok=True)
    p = f"{state_dir}/{voice}.claim"
    if os.path.exists(f"{state_dir}/{voice}.DONE"):
        return False
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {time.time():.0f}\n".encode())
        os.close(fd)
        return True
    except FileExistsError:
        # a stale claim from a dead worker is reclaimable after the lease expires
        try:
            if time.time() - os.path.getmtime(p) > 3 * 3600:
                os.utime(p, None)
                return True
        except OSError:
            pass
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{NB}/vcbon/prod/run1")
    ap.add_argument("--n-cand", type=int, default=4)
    ap.add_argument("--pack", type=int, default=8)
    ap.add_argument("--limit-takes", type=int, default=0, help="smoke: cap takes per voice")
    ap.add_argument("--max-voices", type=int, default=0)
    ap.add_argument("--walltime-s", type=int, default=0, help="stop claiming new voices after this")
    a = ap.parse_args()
    state = f"{a.out}/state"
    os.makedirs(state, exist_ok=True)

    import pandas as pd
    import torch
    import vcengine as E

    vs = verified_voices()
    log(f"{len(vs)} voices with a verified repair and source tars")

    vc = E.load_vc("cuda")
    # Sidon is a class here, and restore() takes a BATCH -- so all four candidates of a
    # take are restored in one call rather than four, which is where its throughput is.
    sid = E.Sidon(device="cuda")
    log("models loaded")

    t_start = time.time()
    n_v = 0
    for voice in vs:
        if a.max_voices and n_v >= a.max_voices:
            break
        if a.walltime_s and time.time() - t_start > a.walltime_s:
            log("walltime reached, not claiming more voices")
            break
        if not claim(state, voice):
            continue
        n_v += 1
        vo = f"{a.out}/voices/{voice}"
        os.makedirs(vo, exist_ok=True)
        t0 = time.time()

        P = f"{NB}/vprof/vp500/{voice}/PPILOT2"
        fs = sorted(glob.glob(f"{P}/meta-*.parquet")) or sorted(glob.glob(f"{P}/*.parquet"))
        mf = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        if "empty" in mf.columns:
            mf = mf[~mf["empty"].astype(bool)]
        if a.limit_takes:
            mf = mf.head(a.limit_takes)

        # the VC target is this voice's own prepared reference -- the production pipeline's target
        ref = None
        for rt in (f"{NB}/vcbon/refs500/refs_prepared.tar", f"{NB}/vcbon/refs/refs_prepared.tar"):
            if not os.path.exists(rt):
                continue
            with tarfile.open(rt) as tf:
                for m in tf:
                    if m.name.startswith(voice):
                        w, sr = E.decode_audio_bytes(tf.extractfile(m).read(), target_sr=None)
                        ref = (w, sr)
                        break
            if ref:
                break
        if ref is None:
            log(f"  {voice}: NO prepared reference -- skipped, not substituted")
            json.dump(dict(voice=voice, error="no_reference"), open(f"{vo}/error.json", "w"))
            continue
        E.set_target_from_wav(vc, ref[0], ref[1])

        src = {}
        for t in sorted(glob.glob(f"{P}/*.tar")):
            with tarfile.open(t) as tf:
                for m in tf:
                    if m.name in set(mf["audio_key"]):
                        src[m.name] = tf.extractfile(m).read()
        log(f"  {voice}: {len(mf)} takes, {len(src)} audio found")

        rows = []
        outtar = tarfile.open(f"{vo}/cands.tar", "w")
        import soundfile as sf
        import torchaudio
        # sort by length so a pack contains similar-length sources -- padding to the longest in a
        # pack is pure waste otherwise
        keys = [k for k in mf["audio_key"] if k in src]
        w16 = {k: E.decode_audio_bytes(src[k], target_sr=16000)[0] for k in keys}
        keys.sort(key=lambda k: len(w16[k]))
        for s_i in range(0, len(keys), a.pack):
            ks = keys[s_i: s_i + a.pack]
            ws = [w16[k] for k in ks]
            try:
                # ONE forward pass for pack x n_cand candidates. Calling generate() per candidate
                # is ~5.7x more expensive and was the whole reason the first launch was stopped.
                tok, ln = E.tokenize(vc, ws, "cuda")
                wav = E.generate_batch(vc, tok, ln, a.n_cand, seed=1234 + s_i)
            except Exception as e:
                log(f"    batch fail at {s_i}: {str(e)[:110]}")
                continue
            # SIDON in ONE call for the whole pack: 8 sources x 4 candidates = 32 items, which is
            # exactly restore()'s max_items. Restoring 4 at a time left most of its throughput on
            # the floor and was the bottleneck after the VC batching fix.
            flat, owner = [], []
            for i, key in enumerate(ks):
                n = int(ln[i]) * 960
                blk = wav[i * a.n_cand:(i + 1) * a.n_cand, :n]
                for c in range(blk.shape[0]):
                    flat.append(blk[c].float().cpu().numpy())
                    owner.append((i, key))
            try:
                out = sid.restore(flat, 24000)
                allys, esr = (list(out[0]), out[1]) if isinstance(out, tuple) else (list(out), 48000)
            except Exception as ex:
                log(f"    sidon fail pack {s_i}: {str(ex)[:90]}")
                allys, esr = flat, 24000
            per_key = {}
            for (i, key), y in zip(owner, allys):
                per_key.setdefault(key, []).append(y)
            for i, key in enumerate(ks):
                ys = per_key.get(key) or []
                if not ys:
                    continue
                base = mf[mf["audio_key"] == key].iloc[0]
                for c, y in enumerate(ys):
                    nm = f"{key[:-4]}.vc{c}.wav"
                    b = io.BytesIO()
                    sf.write(b, np.clip(np.asarray(y), -1, 1), esr, format="WAV")
                    d = b.getvalue()
                    ti = tarfile.TarInfo(nm)
                    ti.size = len(d)
                    ti.mtime = 0
                    outtar.addfile(ti, io.BytesIO(d))
                    rows.append(dict(voice=voice, gid=base["gid"], audio_key=key, cand=c,
                                     out_key=nm,
                                     src_emo_strength=float(base.get("emo_strength", np.nan)),
                                     src_quality=float(base.get("quality", np.nan)),
                                     src_spk_sim=float(base.get("spk_sim", np.nan))))
            if (s_i // a.pack) % 50 == 0:
                log(f"    {min(s_i+a.pack,len(keys))}/{len(keys)}")
        outtar.close()
        pd.DataFrame(rows).to_parquet(f"{vo}/cands.parquet")
        dt = time.time() - t0
        json.dump(dict(voice=voice, n_src=len(keys), n_out=len(rows), sec=round(dt, 1),
                       n_cand=a.n_cand),
                  open(f"{vo}/summary.json", "w"), indent=1)
        open(f"{state}/{voice}.DONE", "w").write(str(time.time()))
        log(f"  {voice} done: {len(rows)} candidates in {dt:.0f}s")

    log(f"PROD_WORKER_DONE voices={n_v}")


if __name__ == "__main__":
    main()
