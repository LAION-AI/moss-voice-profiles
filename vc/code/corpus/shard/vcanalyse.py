"""Analysis over the stored candidate scores.

Everything here is a query over `cand-*.parquet`. That is the point of storing
all four candidates and every reward component: none of these questions cost a
GPU-hour, and a different reward can be evaluated the same way.

Three questions:

  ab        SIDON-then-rank (the owner's order, what ships) vs rank-then-SIDON
            (the pilot's order), scored on the SAME instrument and on the audio
            that would actually ship in each case. Diagnostics only -- the
            pipeline order is decided.
  recovery  conversion costs -50.8 % target emotion strength (pilot). How much
            of that does ranking on the reward recover, versus an arbitrary pick?
  through   measured throughput vs the 3,588 GPU-h plan.
"""
import os, sys, glob, json, argparse
import numpy as np, pandas as pd

NB = "/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox"
PROD = f"{NB}/vcbon/prod"


def pick(df, col):
    """Return one row per source sample, the candidate chosen by `col`."""
    i = df.groupby(["source_run_dir", "source_audio_key"])[col].idxmax()
    return df.loc[i]


def ab(df):
    """rank-then-SIDON vs SIDON-then-rank, both judged on post-SIDON audio."""
    if "pre_qual_raw" not in df.columns:
        return {"error": "no pre-SIDON columns; run a shard with --ab-presidon 1"}
    d = df.dropna(subset=["pre_qual_raw", "pre_emo_target_raw",
                          "qual_raw", "emo_target_raw"]).copy()
    g = ["source_run_dir", "source_audio_key"]

    # reward computed on PRE-SIDON scores, normalised the same way (within-set z,
    # since frozen group stats are calibrated on post-SIDON and would be a
    # different instrument on the pre-SIDON scale)
    def zs(s):
        m, sd = s.transform("mean"), s.transform("std")
        return (s.obj - m) / sd.replace(0, np.nan)
    gb = d.groupby(g)
    d["pre_reward"] = (zs(gb["pre_emo_target_raw"]) + zs(gb["pre_qual_raw"])).fillna(0.0)
    d["post_reward"] = (zs(gb["emo_target_raw"]) + zs(gb["qual_raw"])).fillna(0.0)

    w_post = pick(d, "post_reward")     # owner's order: rank the enhanced audio
    w_pre = pick(d, "pre_reward")       # pilot's order: rank first, then enhance
    rnd = d[d.cand_idx == 0]            # arbitrary pick, the k=1 baseline

    def summ(x, tag):
        return dict(arm=tag, n=int(len(x)),
                    emo_target=float(x.emo_target_raw.mean()),
                    overall_q=float(x.qual_raw.mean()),
                    ecapa=float(x.ecapa_to_prepared_ref.mean()),
                    wavlm=float(x.wavlm_to_prepared_ref.mean()),
                    blend=float(x.blend_vc.mean()),
                    genuineness=float(x.genuineness_vc.mean()))

    rows = [summ(w_post, "SIDON-then-rank (shipped)"),
            summ(w_pre, "rank-then-SIDON (pilot order)"),
            summ(rnd, "arbitrary pick (k=1)")]
    agree = float((w_post.reset_index(drop=True).cand_idx.values ==
                   w_pre.sort_values(g).reset_index(drop=True).cand_idx.values).mean()) \
        if len(w_post) == len(w_pre) else float("nan")

    # the pilot's separate claim: what does SIDON itself do to a fixed candidate?
    sid = dict(n=int(len(d)),
               d_overall_q=float((d.qual_raw - d.pre_qual_raw).mean()),
               d_emo_target=float((d.emo_target_raw - d.pre_emo_target_raw).mean()),
               d_ecapa=float((d.ecapa_to_prepared_ref - d.pre_ecapa).mean()),
               d_wavlm=float((d.wavlm_to_prepared_ref - d.pre_wavlm).mean()),
               d_blend=float((d.blend_vc - d.pre_blend).mean()),
               d_genuineness=float((d.genuineness_vc - d.pre_genuineness).mean()))
    return dict(arms=rows, winner_agreement=agree, sidon_effect_fixed_candidate=sid,
                delta_shipped=dict(
                    emo_target=rows[0]["emo_target"] - rows[1]["emo_target"],
                    overall_q=rows[0]["overall_q"] - rows[1]["overall_q"],
                    ecapa=rows[0]["ecapa"] - rows[1]["ecapa"]))


def apply_norm(df, norm_path):
    """Recompute `reward_group` from the frozen constants and the stored raw scores.

    This is the "selection is a view" property being used rather than described:
    the normalisation-pass shards were generated *before* the constants existed,
    so their stored `reward_group` column is degenerate — and it costs nothing to
    fix, because every raw score was kept. The same call re-scores the corpus
    under any future reward.
    """
    if not os.path.exists(norm_path):
        return df, False
    N = json.load(open(norm_path))
    if not (N.get("emo") and N.get("qual")):
        return df, False
    e, q = N["emo"], N["qual"]
    sgn = df.get("target_sign")
    sgn = sgn.fillna(1.0).astype(int) if sgn is not None else pd.Series(1, index=df.index)
    keys = df.target_kind.astype(str) + ":" + df.target_name.astype(str) + ":" + sgn.astype(str)

    def lookup(k, field):
        kk = str(k)
        st = e.get(kk)
        if st is None:
            p = kk.split(":")
            st = (e.get(f"{p[0]}:{p[1]}") or e.get(f"{p[0]}:*:{p[2]}")
                  or e.get(f"{p[0]}:*") or e["*"])
        return st[field]

    uniq = {k: (lookup(k, "mean"), lookup(k, "sd")) for k in keys.unique()}
    mu = keys.map(lambda k: uniq[k][0]).astype(float)
    sd = keys.map(lambda k: uniq[k][1]).astype(float).clip(lower=1e-9)
    df = df.copy()
    df["z_emo_group"] = (df.emo_target_raw - mu) / sd
    df["z_qual_group"] = (df.qual_raw - q["mean"]) / max(q["sd"], 1e-9)
    df["reward_group"] = df.z_emo_group.fillna(0.0) + df.z_qual_group.fillna(0.0)
    return df, True


def recovery(df):
    """How much of the target-emotion loss does the reward recover?

    The source-side comparison must be sign-matched. For `dim` targets the
    converted strength is `sign x regression` while the corpus stores
    `dim_target` unsigned, so comparing them directly reports a ~-94 % collapse
    that is pure sign error. The source column is signed here with the same rule.
    """
    out = []
    for kind in ("emo", "dim"):
        d = df[(df.target_kind == kind)].dropna(subset=["emo_target_raw"])
        if not len(d):
            continue
        src_col = "emo_strength_src" if kind == "emo" else "dim_target_src"
        if src_col not in d.columns:
            continue
        u = d.drop_duplicates(["source_run_dir", "source_audio_key"])
        src = u[src_col].astype(float)
        if kind == "dim" and "target_sign" in u.columns:
            src = src * u.target_sign.fillna(1.0).astype(float)
        src_mean = float(np.nanmean(src))
        arb = float(d[d.cand_idx == 0].emo_target_raw.mean())
        best4 = float(d.groupby(["source_run_dir", "source_audio_key"])
                      .emo_target_raw.max().mean())
        res = dict(target_kind=kind, n_sources=int(src.notna().sum()),
                   src_mean=src_mean, arbitrary_pick=arb, oracle_best_of_4=best4)
        for lab, col in (("reward_group", "reward_group"),
                         ("reward_set", "reward_set"),
                         ("reward_minmax", "reward_minmax")):
            if col not in d.columns:
                continue
            w = pick(d, col)
            v = float(w.emo_target_raw.mean())
            res[lab] = v
            res[lab + "_rel_to_src"] = (v - src_mean) / abs(src_mean) if src_mean else None
            gap = src_mean - arb
            res[lab + "_gap_recovered"] = (v - arb) / gap if abs(gap) > 1e-9 else None
            res[lab + "_of_oracle"] = (v - arb) / (best4 - arb) if abs(best4 - arb) > 1e-9 else None
        res["arbitrary_rel_to_src"] = (arb - src_mean) / abs(src_mean) if src_mean else None
        # what the reward costs on the other axis
        wg = pick(d, "reward_group") if "reward_group" in d.columns else None
        if wg is not None:
            res["quality_arbitrary"] = float(d[d.cand_idx == 0].qual_raw.mean())
            res["quality_reward"] = float(wg.qual_raw.mean())
            res["ecapa_arbitrary"] = float(d[d.cand_idx == 0].ecapa_to_prepared_ref.mean())
            res["ecapa_reward"] = float(wg.ecapa_to_prepared_ref.mean())
        out.append(res)
    return out


def through(markers):
    recs = []
    for m in markers:
        try:
            recs.append(json.load(open(m)))
        except Exception:
            pass
    if not recs:
        return {}
    CORPUS = 20125736
    sps = np.array([r["s_per_sample"] for r in recs])
    ns = int(sum(r["n_samples"] for r in recs))
    ab = [r for r in recs if r.get("ab_presidon")]
    # bytes per SAMPLE, not per shard: the measurement shards may be truncated
    # by --limit, and multiplying a 2,000-sample shard's size by 2,000 shards
    # under-reports the corpus by 5x.
    bps = float(np.mean([r.get("out_bytes", 0) / max(r["n_samples"], 1) for r in recs]))
    sid = np.array([r.get("sidon_ms_per_clip", np.nan) for r in recs], float)
    stg = {}
    for r in recs:
        for k, v in r["stage_s"].items():
            stg[k] = stg.get(k, 0.0) + v
    tot = sum(r["wall_s"] for r in recs)
    plan_gpu_h = 3588.0
    # Production config = no pre-SIDON A/B pass. Where the measurement shards
    # carried it, its share is removed rather than quietly left in the estimate.
    prod = [r for r in recs if not r.get("ab_presidon")]
    if prod:
        sps_prod = float(np.mean([r["s_per_sample"] for r in prod]))
        basis = f"{len(prod)} shards measured at production config"
    else:
        share_ab = stg.get("presidon_score", 0.0) / max(tot, 1e-9)
        sps_prod = float(sps.mean()) * (1.0 - share_ab)
        basis = (f"estimated: {len(recs)} shards carried the A/B pass "
                 f"({100*share_ab:.1f}% of wall), removed")
    proj = sps_prod * CORPUS / 3600
    return dict(n_shards=len(recs), n_shards_production_config=len(prod), n_samples=ns,
                s_per_sample_measured=float(sps.mean()),
                s_per_sample_production=sps_prod, production_basis=basis,
                s_per_sample_p50=float(np.median(sps)),
                s_per_sample_min=float(sps.min()), s_per_sample_max=float(sps.max()),
                sidon_ms_per_clip=float(np.nanmean(sid)),
                stage_share={k: round(v / max(tot, 1e-9), 4) for k, v in sorted(stg.items())},
                projected_gpu_h_full_run=proj,
                plan_gpu_h=plan_gpu_h, ratio_to_plan=proj / plan_gpu_h,
                projected_core_h=proj * 72,
                projected_h_on_48_nodes=proj / (48 * 4),
                realtime_x=float(np.mean([r["realtime_x"] for r in recs])),
                bytes_per_sample=bps,
                projected_out_TB=bps * CORPUS / 1e12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=f"{NB}/vprof/vc500/*/VC1/cand-*.parquet")
    ap.add_argument("--markers", default=f"{NB}/vprof/vc500/*/VC1/done-*.json")
    ap.add_argument("--out", default=f"{PROD}/out/analysis.json")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--norm", default=f"{PROD}/index/norm_stats.json")
    a = ap.parse_args()

    files = sorted(glob.glob(a.glob))
    if a.max_files:
        files = files[:a.max_files]
    print(f"[analyse] {len(files)} cand parquets")
    keep = ["source_run_dir", "source_audio_key", "cand_idx", "target_kind", "target_name",
            "target_sign",
            "block", "lang", "emo_target_raw", "qual_raw", "blend_vc", "genuineness_vc",
            "ecapa_to_prepared_ref", "wavlm_to_prepared_ref", "ecapa_to_original_ref",
            "reward_group", "reward_set", "reward_minmax", "is_winner_group",
            "emo_strength_src", "dim_target_src", "quality_src", "spk_sim_src",
            "blend_src", "genuineness_src", "dur_src",
            "pre_qual_raw", "pre_emo_target_raw", "pre_ecapa", "pre_wavlm",
            "pre_blend", "pre_genuineness"]
    import pyarrow.parquet as pq
    dfs = []
    for f in files:
        av = set(pq.ParquetFile(f).schema.names)
        dfs.append(pd.read_parquet(f, columns=[c for c in keep if c in av]))
    df = pd.concat(dfs, ignore_index=True)
    df, renormed = apply_norm(df, a.norm)
    print(f"[analyse] {len(df)} candidate rows, "
          f"{df.groupby(['source_run_dir','source_audio_key']).ngroups} sources, "
          f"reward_group {'recomputed from frozen constants' if renormed else 'as stored'}")

    res = dict(n_candidate_rows=int(len(df)), reward_group_recomputed=bool(renormed),
               n_sources=int(df.groupby(["source_run_dir", "source_audio_key"]).ngroups),
               ab=ab(df), recovery=recovery(df),
               throughput=through(sorted(glob.glob(a.markers))))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2, default=float)
    print(json.dumps(res, indent=2, default=float)[:8000])


if __name__ == "__main__":
    main()
