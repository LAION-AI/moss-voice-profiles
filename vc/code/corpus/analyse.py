"""Turn the pilot into the numbers the report needs."""
import os, sys, json, glob
import numpy as np, pandas as pd

NB = "/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox"
OUT = f"{NB}/vcbon/out"
RUN = sys.argv[1] if len(sys.argv) > 1 else f"{NB}/vcbon/pilot/vc_v1"
FLOOR = 0.40
TBR_THR = 0.472

df = pd.read_parquet(f"{RUN}/candidates.parquet")
summ = json.load(open(f"{RUN}/summary.json"))
R = {"summary": summ}
print(f"[analyse] {len(df)} candidate rows, arms={sorted(df.arm.unique())}")

win = df[df.is_winner & df.sidon].copy()          # the shipped take
winr = df[df.is_winner & ~df.sidon].copy()        # winner before SIDON
allr = df[~df.sidon].copy()                       # every raw candidate


# ------------------------------------------------------- identity: the floor --
def idblock(g, tag):
    """Two ECAPA columns on purpose.

    `_o` is measured against the SAME reference file the corpus generator used,
    so it is the only column comparable to `spk_sim_src` and to the 0.40 floor.
    The unsuffixed column is against the prepared (SIDON + R128) reference the
    conversion actually aimed at — a fairer measure of "did it hit its target",
    but not comparable to the corpus's own numbers.
    """
    return dict(
        tag=tag, n=len(g),
        ecapa_src=float(g.spk_sim_src.mean()),
        ecapa_vc_o=float(g.ecapa_vc_origref.mean()),
        ecapa_vc_prep=float(g.ecapa_vc.mean()),
        above_floor_src=float((g.spk_sim_src >= FLOOR).mean()),
        above_floor_vc_o=float((g.ecapa_vc_origref >= FLOOR).mean()),
        above_floor_vc_prep=float((g.ecapa_vc >= FLOOR).mean()),
        wavlm_vc_o=float(g.wavlm_vc_origref.mean()),
        above_tbr_vc_o=float((g.wavlm_vc_origref >= TBR_THR).mean()),
        blend_src=float(g.blend_src.mean()), blend_vc=float(g.blend_vc.mean()),
        blend_rel=float(g.blend_vc.mean() / max(g.blend_src.mean(), 1e-9) - 1),
        genuine_src=float(g.genuineness_src.mean()), genuine_vc=float(g.genuineness_vc.mean()),
        strength_src=float(g.emo_strength_src.mean()),
        strength_vc=float(g.strength_vc.mean()) if g.strength_vc.notna().any() else None,
        overall_q_vc=float(g.overall_q_vc.mean()),
    )


rows = []
for arm in sorted(win.arm.unique()):
    g = win[win.arm == arm]
    rows.append(idblock(g, arm))
    rows.append(idblock(g[g.below_floor_src], f"{arm}/below-floor"))
    rows.append(idblock(g[~g.below_floor_src], f"{arm}/above-floor"))
R["identity"] = rows
print("\n=== identity & expression, winner after SIDON ===")
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ---------------------------------------------- emotion-strength, E block only --
e = win[win.target_kind == "emo"].copy()
v = win[win.target_kind == "dim"].copy()
tgt = []
for arm in sorted(win.arm.unique()):
    ge, gv = e[e.arm == arm], v[v.arm == arm]
    tgt.append(dict(arm=arm,
                    n_emo=len(ge), emo_src=float(ge.emo_strength_src.mean()),
                    emo_vc=float(ge.strength_vc.mean()),
                    emo_rel=float(ge.strength_vc.mean() / max(ge.emo_strength_src.mean(), 1e-9) - 1),
                    n_dim=len(gv),
                    dim_src=float(gv.dim_target_src.abs().mean()) if gv.dim_target_src.notna().any() else None,
                    dim_vc=float(gv.strength_vc.abs().mean()) if gv.strength_vc.notna().any() else None))
R["target_dimension"] = tgt
print("\n=== target dimension / emotion strength ===")
print(pd.DataFrame(tgt).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# --------------------------------------------------------- best-of-k curve ----
bok = []
for arm in sorted(allr.arm.unique()):
    g = allr[allr.arm == arm]
    for k in (1, 2, 4, 8):
        sub = g[g.cand_idx < k]
        best = sub.loc[sub.groupby(["src_key"])["reward_identity"].idxmax()]
        be = best[best.target_kind == "emo"]
        bok.append(dict(arm=arm, k=k, n=len(best),
                        ecapa=float(best.ecapa_vc_origref.mean()),
                        above_floor=float((best.ecapa_vc_origref >= FLOOR).mean()),
                        emo_strength=float(be.strength_vc.mean()) if len(be) else None,
                        wavlm=float(best.wavlm_vc_origref.mean()),
                        overall_q=float(best.overall_q_vc.mean()),
                        blend=float(best.blend_vc.mean()),
                        strength=float(best.strength_vc.mean()) if best.strength_vc.notna().any() else None,
                        reward=float(best.reward_identity.mean())))
R["best_of_k"] = bok
print("\n=== best-of-k (raw candidates, identity-aware reward) ===")
print(pd.DataFrame(bok).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ------------------------------------------------------------- SIDON effect ---
m = winr.merge(win, on=["arm", "src_key", "cand_idx"], suffixes=("_raw", "_sid"))
sid = []
for arm in sorted(m.arm.unique()):
    g = m[m.arm == arm]
    sid.append(dict(arm=arm, n=len(g),
                    d_ecapa=float((g.ecapa_vc_origref_sid - g.ecapa_vc_origref_raw).mean()),
                    d_ecapa_prep=float((g.ecapa_vc_sid - g.ecapa_vc_raw).mean()),
                    d_wavlm=float((g.wavlm_vc_origref_sid - g.wavlm_vc_origref_raw).mean()),
                    d_overall=float((g.overall_q_vc_sid - g.overall_q_vc_raw).mean()),
                    d_speech=float((g.speech_q_vc_sid - g.speech_q_vc_raw).mean()),
                    d_blend=float((g.blend_vc_sid - g.blend_vc_raw).mean()),
                    d_strength=float((g.strength_vc_sid - g.strength_vc_raw).mean())))
R["sidon_effect"] = sid
print("\n=== SIDON on the winner (delta) ===")
print(pd.DataFrame(sid).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

# ---------------------------------------------------- the two floors disagree --
g = win[win.arm == "self"]
src_below = g.below_floor_src
R["floor_disagreement"] = dict(
    n=len(g),
    src_below_ecapa=float(src_below.mean()),
    src_below_but_vc_above=float((src_below & (g.ecapa_vc_origref >= FLOOR)).sum() / max(src_below.sum(), 1)),
    vc_above_ecapa=float((g.ecapa_vc_origref >= FLOOR).mean()),
    vc_above_wavlm=float((g.wavlm_vc_origref >= TBR_THR).mean()),
    vc_ecapa_below_but_wavlm_above=float(((g.ecapa_vc_origref < FLOOR) & (g.wavlm_vc_origref >= TBR_THR)).sum()
                                         / max((g.ecapa_vc_origref < FLOOR).sum(), 1)),
    src_ecapa_below_but_wavlm_src_na="corpus stores no WavLM for source takes",
)
print("\n=== floors ===");  print(json.dumps(R["floor_disagreement"], indent=2))

# ------------------------------------------------------------ per-block view --
blk = win[win.arm == "self"].groupby("block").agg(
    n=("ecapa_vc", "size"), ecapa_src=("spk_sim_src", "mean"), ecapa_vc=("ecapa_vc_origref", "mean"),
    af_src=("spk_sim_src", lambda x: (x >= FLOOR).mean()),
    af_vc=("ecapa_vc_origref", lambda x: (x >= FLOOR).mean()),
    blend_src=("blend_src", "mean"), blend_vc=("blend_vc", "mean"),
    q=("overall_q_vc", "mean")).reset_index()
R["per_block"] = blk.to_dict("records")
print("\n=== per block (arm=self) ===");  print(blk.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# --------------------------------------------------------- match quality vs Δ --
mm = json.load(open(f"{RUN}/match.json"))
R["match"] = mm
mq = []
for arm in ("nn", "far"):
    if arm not in set(win.arm):
        continue
    for v_, g in win[win.arm == arm].groupby("src_voice"):
        mq.append(dict(arm=arm, src_voice=v_, target=mm[v_][arm], dist=mm[v_][f"{arm}_d"],
                       ecapa=float(g.ecapa_vc.mean()),
                       above_floor=float((g.ecapa_vc >= FLOOR).mean()),
                       wavlm=float(g.wavlm_vc.mean()),
                       overall_q=float(g.overall_q_vc.mean()),
                       blend=float(g.blend_vc.mean()),
                       strength=float(g.strength_vc.mean())))
R["match_quality"] = mq
if mq:
    print("\n=== conversion quality vs how near the borrowed voice is ===")
    print(pd.DataFrame(mq).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    d = pd.DataFrame(mq)
    if len(d) > 2:
        R["match_corr_dist_ecapa"] = float(np.corrcoef(d.dist, d.ecapa)[0, 1])
        print(f"corr(distance, ECAPA-to-target) = {R['match_corr_dist_ecapa']:.3f}")

json.dump(R, open(f"{OUT}/pilot_analysis.json", "w"), indent=2, default=float)
print(f"\nwrote {OUT}/pilot_analysis.json")
