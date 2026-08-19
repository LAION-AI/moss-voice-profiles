"""Best-of-8 expressive voice conversion + SIDON, on real vp500 corpus takes.

Three arms, so the run answers both scenarios at once:

  self  target = the take's OWN prepared reference
        -> voice conversion as an identity-repair path. Does converting a
           below-floor take push it above the 0.40 ECAPA floor, and what does
           that cost in emotion strength and blend?

  nn    target = the acoustically NEAREST other voice's prepared reference
  far   target = a deliberately DISTANT voice's prepared reference
        -> scenario 2. Borrowing a generated corpus and converting it only
           works if the borrowed voice is near; these two arms bracket that.

Every candidate is scored with the corpus's OWN sensors (pp_scores_fast:
voiceclap blend/genuineness, BUD-E emonet, VoiceNet dims, DNSMOS-distilled
quality) so before/after is the same instrument, plus two independent speaker
embedders (ECAPA, which defines the 0.40 floor, and Orange WavLM-tbr, which
does not agree with it).

Output: WebDataset tar + parquet, new paths, nothing overwritten.
"""
import os, sys, io, json, glob, time, tarfile, argparse
import numpy as np, torch, torchaudio, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox")
import vcengine as E

NB = "/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox"
DEV = "cuda:0"
SPK_FLOOR = 0.40          # vpspec.SPK_FLOOR — an ECAPA-scale convention
TBR_THR = 0.472           # Orange's published VoxCeleb1-clean EER threshold

ap = argparse.ArgumentParser()
ap.add_argument("--n-cand", type=int, default=8)
ap.add_argument("--pack", type=int, default=8, help="sources per forward pass")
ap.add_argument("--arms", default="self,nn,far")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--sidon-all", action="store_true", help="restore all N, not just the winner")
ap.add_argument("--tf32", type=int, default=0)
ap.add_argument("--out", default=f"{NB}/vcbon/pilot/vc_v1")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = bool(a.tf32)
torch.backends.cudnn.allow_tf32 = bool(a.tf32)

# ------------------------------------------------------------------ inputs ---
mf = pd.read_parquet(f"{NB}/vcbon/pilot/sources.parquet")
if a.limit:
    mf = mf.groupby("src_voice", group_keys=False).head(a.limit).reset_index(drop=True)
src16 = {}
with tarfile.open(f"{NB}/vcbon/pilot/sources.tar") as tf:
    for m in tf:
        src16[m.name] = E.decode_audio_bytes(tf.extractfile(m).read(), target_sr=16000)[0]

refs = {}
for rt in (f"{NB}/vcbon/refs500/refs_prepared.tar", f"{NB}/vcbon/refs/refs_prepared.tar"):
    if not os.path.exists(rt):
        continue
    with tarfile.open(rt) as tf:
        for m in tf:
            c = os.path.basename(m.name).replace(".prep.mp3", "")
            if c not in refs:
                refs[c] = E.decode_audio_bytes(tf.extractfile(m).read())
    if refs:
        break
print(f"[pilot] {len(mf)} takes, {len(refs)} prepared references", flush=True)

# ------------------------------------------------- nearest / farthest match ---
CONS = sorted(glob.glob(f"{NB}/hfcache/.cache/dramabox/"
                        f"datasets--TTS-AGI--moss-reference-voices-consolidated/snapshots/*"))[-1]
dims = np.load(f"{CONS}/annotations/dims_enh.npy")
meta = pd.read_parquet(f"{CONS}/metadata.parquet")
pool = json.load(open(f"{NB}/vprof/work/voices500.json"))["voices"]
pool = [c for c in pool if c in refs]
idx = {c: i for i, c in enumerate(meta["cid"])}
Z = (dims - dims.mean(0)) / (dims.std(0) + 1e-9)
P = np.array([idx[c] for c in pool])
match = {}
for v in mf.src_voice.unique():
    d = np.linalg.norm(Z[P] - Z[idx[v]], axis=1)
    o = np.argsort(d)
    nn = pool[int(o[1])]                       # o[0] is v itself
    far = pool[int(o[int(len(o) * 0.95)])]     # p95 distance, a bad-but-real match
    match[v] = dict(nn=nn, nn_d=float(d[o[1]]), far=far, far_d=float(d[o[int(len(o)*0.95)]]))
    print(f"[match] {v:16s} nn={nn:16s} d={d[o[1]]:6.2f}   far={far:16s} d={d[o[int(len(o)*0.95)]]:6.2f}",
          flush=True)
json.dump(match, open(f"{a.out}/match.json", "w"), indent=2)

# ------------------------------------------------------------------ models ---
T = {}
t0 = time.time(); vc = E.load_vc(DEV); T["load_vc"] = time.time() - t0
t0 = time.time(); sd = E.Sidon(DEV, threads=16); T["load_sidon"] = time.time() - t0
t0 = time.time()
sp = E.SpeakerSim(DEV, savedir=f"{NB}/vcbon/ecapa_ckpt", spk_emb_path=f"{NB}/vprof/idloop/code")
T["load_spk"] = time.time() - t0
t0 = time.time()
from pp_scores_fast import FastScorer
fs = FastScorer("cuda")
T["load_scorer"] = time.time() - t0
print(f"[pilot] models loaded: {json.dumps({k: round(v,1) for k,v in T.items()})}", flush=True)


def score16(wavs16):
    return fs.score_batch([torch.as_tensor(np.asarray(w, np.float32)) for w in wavs16])


def gid_target(gid, block, emotion, cond):
    """Which single number is 'the target dimension / emotion strength' here."""
    p = gid.split("|")
    if block == "voicenet" and len(p) >= 5:
        return ("dim", p[2], 1.0 if "high" in p[3] else -1.0)
    if emotion:
        return ("emo", emotion, 1.0)
    return (None, None, 1.0)


# reference embeddings + reference-side scores, once per target voice
tgt_cache = {}


def target_of(cid):
    """Two anchors on purpose.

    `ec`/`tb` are the PREPARED reference — the thing the conversion was actually
    aimed at. `ec0`/`tb0` are the reference file the corpus generator itself used
    (`manifest_<cid>.json["ref_wav"]`, 16 kHz mono, untouched). Only the second
    one is comparable to the corpus's stored `spk_sim` and therefore to the 0.40
    floor; the first is the fair measure of "did the conversion hit its target".
    Reporting one without the other would be sleight of hand.
    """
    if cid in tgt_cache:
        return tgt_cache[cid]
    w, sr = refs[cid]
    w16 = torchaudio.functional.resample(torch.as_tensor(w), sr, 16000).numpy()
    man = json.load(open(f"{NB}/vprof/work/manifest_{cid}.json"))
    import librosa
    o16, _ = librosa.load(man["ref_wav"], sr=16000, mono=True)
    tgt_cache[cid] = dict(wav=w, sr=sr,
                          ec=sp.ecapa_emb([w16])[0], tb=sp.orange_emb([w16])[0],
                          ec0=sp.ecapa_emb([o16])[0], tb0=sp.orange_emb([o16])[0])
    return tgt_cache[cid]


# ------------------------------------------------------------------ the run ---
arms = a.arms.split(",")
rows, tars, stage = [], {}, dict(tokenize=0.0, generate=0.0, score=0.0, spk=0.0,
                                 sidon=0.0, rescore=0.0, encode=0.0, io=0.0)
n_cand_total = 0
audio_s_in = 0.0
t_run0 = time.time()

for arm in arms:
    tarp = f"{a.out}/converted-{arm}.tar"
    tars[arm] = tarfile.open(tarp, "w")
    for v, g0 in mf.groupby("src_voice"):
        tcid = v if arm == "self" else match[v][arm]
        tg = target_of(tcid)
        E.set_target_from_wav(vc, tg["wav"], tg["sr"], peak_norm=None)
        g0 = g0.assign(_n=g0["key"].map(lambda k: len(src16[k]))).sort_values("_n")
        keys = list(g0["key"])
        for s in range(0, len(keys), a.pack):
            ks = keys[s: s + a.pack]
            sub = g0[g0.key.isin(ks)].set_index("key").loc[ks].reset_index()
            ws = [src16[k] for k in ks]
            audio_s_in += sum(len(w) / 16000 for w in ws)

            torch.cuda.synchronize(); t0 = time.time()
            tok, ln = E.tokenize(vc, ws, DEV)
            torch.cuda.synchronize(); stage["tokenize"] += time.time() - t0

            t0 = time.time()
            wav = E.generate_batch(vc, tok, ln, a.n_cand, seed=1234 + s)
            torch.cuda.synchronize(); stage["generate"] += time.time() - t0
            n_cand_total += len(ks) * a.n_cand

            # trim each candidate to its own source length, resample to 16k on GPU
            t0 = time.time()
            cands, c16 = [], []
            for i in range(len(ks)):
                n = int(ln[i]) * 960
                blk = wav[i * a.n_cand: (i + 1) * a.n_cand, :n]
                cands.append(blk)
                c16.append(torchaudio.functional.resample(blk.float(), 24000, 16000).cpu().numpy())
            torch.cuda.synchronize(); stage["io"] += time.time() - t0

            flat16 = [c16[i][j] for i in range(len(ks)) for j in range(a.n_cand)]
            t0 = time.time(); sc = score16(flat16); stage["score"] += time.time() - t0
            t0 = time.time()
            ec = sp.ecapa_emb(flat16); tb = sp.orange_emb(flat16)
            ecs = (ec @ tg["ec"]).cpu().numpy(); tbs = (tb @ tg["tb"]).cpu().numpy()
            ecs0 = (ec @ tg["ec0"]).cpu().numpy(); tbs0 = (tb @ tg["tb0"]).cpu().numpy()
            stage["spk"] += time.time() - t0

            # ---- rank ----
            def z(x):
                x = np.asarray(x, float); s = x.std()
                return (x - x.mean()) / s if s > 1e-6 else x * 0.0

            winners, per = [], []
            for i, k in enumerate(ks):
                r = sub.iloc[i]
                sl = slice(i * a.n_cand, (i + 1) * a.n_cand)
                S = sc[sl]
                kind, name, sgn = gid_target(r["gid"], r["block"], r["emotion"], r["condition"])
                if kind == "dim":
                    strength = np.array([sgn * float(x["voicenet"].get(name, {}).get("reg", np.nan))
                                         if isinstance(x["voicenet"].get(name), dict)
                                         else np.nan for x in S])
                elif kind == "emo":
                    strength = np.array([float(x["emonet"].get(name, 0.0)) for x in S])
                else:
                    strength = np.array([max(x["emonet"].values()) for x in S])
                ovq = np.array([float(x["quality"].get("overall_quality", np.nan)) for x in S])
                bl = np.array([float(x["blend_0_10"]) for x in S])
                gn = np.array([float(x["genuineness_0_6"]) for x in S])
                e = ecs[sl]; t_ = tbs[sl]
                st = np.nan_to_num(strength, nan=float(np.nanmean(strength)) if np.isfinite(strength).any() else 0.0)
                rw_up = z(st) + z(ovq)                      # upstream reward
                rw_id = z(st) + z(ovq) + z(e)               # identity-aware
                order = np.argsort(-rw_id)
                win = int(order[0])
                winners.append((i, win))
                for j in range(a.n_cand):
                    per.append(dict(
                        arm=arm, src_voice=v, target_voice=tcid,
                        run_dir=r["run_dir"], audio_key=r["audio_key"], src_key=r["key"],
                        gid=r["gid"], cand_src=int(r["cand"]), block=r["block"], lang=r["lang"],
                        emotion=r["emotion"], condition=r["condition"],
                        target_kind=kind, target_name=name,
                        cand_idx=j, rank=int(np.where(order == j)[0][0]), is_winner=(j == win),
                        dur=float(r["dur"]),
                        spk_sim_src=float(r["spk_sim_src"]), below_floor_src=bool(r["below_floor_src"]),
                        blend_src=float(r["blend_src"]), genuineness_src=float(r["genuineness_src"]),
                        emo_strength_src=float(r["emo_strength_src"]),
                        dim_target_src=(float(r["dim_target_src"]) if pd.notna(r["dim_target_src"]) else None),
                        ecapa_vc=float(e[j]), wavlm_vc=float(t_[j]),
                        ecapa_vc_origref=float(ecs0[sl][j]), wavlm_vc_origref=float(tbs0[sl][j]),
                        strength_vc=float(strength[j]) if np.isfinite(strength[j]) else None,
                        blend_vc=float(bl[j]), genuineness_vc=float(gn[j]),
                        overall_q_vc=float(ovq[j]),
                        speech_q_vc=float(S[j]["quality"].get("speech_quality", np.nan)),
                        reward_upstream=float(rw_up[j]), reward_identity=float(rw_id[j]),
                        sidon=False,
                    ))

            # ---- SIDON: the winner (or all) ----
            todo = ([(i, j) for i in range(len(ks)) for j in range(a.n_cand)]
                    if a.sidon_all else winners)
            t0 = time.time()
            sw = sd.restore([cands[i][j].float().cpu().numpy() for i, j in todo], 24000,
                            max_frames=8000, max_items=32)
            torch.cuda.synchronize(); stage["sidon"] += time.time() - t0
            sw16 = [torchaudio.functional.resample(torch.as_tensor(x), 48000, 16000).numpy() for x in sw]
            t0 = time.time()
            ssc = score16(sw16)
            se = sp.ecapa_emb(sw16); st_ = sp.orange_emb(sw16)
            ses = (se @ tg["ec"]).cpu().numpy(); sts = (st_ @ tg["tb"]).cpu().numpy()
            ses0 = (se @ tg["ec0"]).cpu().numpy(); sts0 = (st_ @ tg["tb0"]).cpu().numpy()
            stage["rescore"] += time.time() - t0

            t0 = time.time()
            for n, (i, j) in enumerate(todo):
                r = sub.iloc[i]
                kind, name, sgn = gid_target(r["gid"], r["block"], r["emotion"], r["condition"])
                X = ssc[n]
                if kind == "dim":
                    vnd = X["voicenet"].get(name)
                    strength = sgn * float(vnd["reg"]) if isinstance(vnd, dict) else None
                elif kind == "emo":
                    strength = float(X["emonet"].get(name, 0.0))
                else:
                    strength = float(max(X["emonet"].values()))
                base = dict(per[i * a.n_cand + j])
                base.update(sidon=True, cand_idx=j, is_winner=(j == winners[i][1]),
                            ecapa_vc=float(ses[n]), wavlm_vc=float(sts[n]),
                            ecapa_vc_origref=float(ses0[n]), wavlm_vc_origref=float(sts0[n]),
                            strength_vc=strength, blend_vc=float(X["blend_0_10"]),
                            genuineness_vc=float(X["genuineness_0_6"]),
                            overall_q_vc=float(X["quality"].get("overall_quality", np.nan)),
                            speech_q_vc=float(X["quality"].get("speech_quality", np.nan)))
                per.append(base)
                if j == winners[i][1]:
                    key = f"{arm}/{v}__{tcid}/{r['audio_key'][:-4]}.k{j}"
                    b = E.encode_mp3_bytes(sw[n], 48000, 160000)
                    ti = tarfile.TarInfo(key + ".mp3"); ti.size = len(b); ti.mtime = 0
                    tars[arm].addfile(ti, io.BytesIO(b))
                    prov = json.dumps(dict(
                        source_run_dir=r["run_dir"], source_audio_key=r["audio_key"],
                        source_gid=r["gid"], source_cand=int(r["cand"]),
                        target_voice=tcid, target_ref="prepared: SIDON + EBU R128 -23 LUFS",
                        arm=arm, candidate_index=j, n_candidates=a.n_cand,
                        rank=0, reward_identity=float(base["reward_identity"]),
                        reward_upstream=float(base["reward_upstream"]),
                        ecapa_to_target=float(ses[n]), wavlm_to_target=float(sts[n]),
                        ecapa_to_original_ref=float(ses0[n]), wavlm_to_original_ref=float(sts0[n]),
                        ecapa_src_to_own_ref=float(r["spk_sim_src"]),
                        sidon=True, model="ResembleAI/chatterbox s3gen VC",
                    )).encode()
                    ti = tarfile.TarInfo(key + ".json"); ti.size = len(prov); ti.mtime = 0
                    tars[arm].addfile(ti, io.BytesIO(prov))
            stage["encode"] += time.time() - t0
            rows.extend(per)
        print(f"[pilot] arm={arm} voice={v} -> {len(rows)} rows  "
              f"{time.time()-t_run0:.0f}s", flush=True)
    tars[arm].close()

wall = time.time() - t_run0
df = pd.DataFrame(rows)
df.to_parquet(f"{a.out}/candidates.parquet", index=False)
n_samples = len(mf) * len(arms)
summary = dict(n_takes=len(mf), arms=arms, n_cand=a.n_cand, pack=a.pack,
               sidon_all=a.sidon_all, tf32=bool(a.tf32),
               n_samples=n_samples, n_candidates=n_cand_total,
               audio_s_in=audio_s_in, wall_s=wall,
               stage_s=stage, load_s=T,
               s_per_sample=wall / n_samples,
               gpu_h_per_1k=wall / n_samples * 1000 / 3600,
               realtime_x=audio_s_in / wall)
json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=2, default=float)
print("\n=== pilot summary ===")
print(json.dumps(summary, indent=2, default=float))
