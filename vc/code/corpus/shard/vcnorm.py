"""Freeze the reward normalisation constants.

The reward is  normalise(target emotion strength) + normalise(quality).
The choice of `normalise` is not cosmetic and is recorded rather than assumed
(see PROTOCOL.md / the write-up in norm_stats.json["why"]).

What this computes: per-group mean and sd of the CONVERTED, SIDON-enhanced
candidates, estimated from the smoke shards and then frozen for the whole run.
Groups are `(<target kind>:<target name>)` for the emotion term — because each
of the 40 emotion experts and each VoiceNet dimension has its own calibration
and range, so one global z-score would silently weight "Anger" against "Awe" by
their scale rather than their meaning — and a single global group for quality,
because Overall-Quality is one head on one scale.

Robust (median / 1.4826 x MAD) estimates are computed alongside and reported;
if the two disagree materially the distribution is not one a z-score describes
and that needs to be known, not hidden.
"""
import os, sys, glob, json, time, argparse
import numpy as np, pandas as pd

NB = "/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox"
PROD = f"{NB}/vcbon/prod"

WHY = (
 "z-score against FROZEN GROUP statistics, not within the 4-candidate set.\n"
 "\n"
 "Both candidates for `normalise` put the two terms on a common scale, which is "
 "what the spec asks for. They differ in what else they do, and the difference "
 "decides the ranking:\n"
 "\n"
 "  within-set (z or min-max over the 4 candidates)\n"
 "    forces the two terms to contribute EQUALLY in every set, whatever the set "
 "    actually looks like. If four candidates differ in Overall-Quality by 0.002 "
 "    (i.e. they are tied, and the difference is scorer noise) but differ in "
 "    target emotion by 1.4, within-set normalisation inflates that quality noise "
 "    to the same magnitude as the real emotion signal and lets it cast half the "
 "    vote. With n=4 the sd estimate itself carries ~40 % relative error, so the "
 "    effective weighting between the two terms is re-randomised for every "
 "    sample. min-max is worse again: it is defined by the two most extreme -- "
 "    i.e. noisiest -- order statistics, saturates the winner at 1.0 and the "
 "    loser at 0.0 regardless of spread, and is degenerate on ties.\n"
 "\n"
 "  frozen group z-score  <- chosen\n"
 "    divides each term by the spread that term has ACROSS THE CORPUS. Neither "
 "    term can dominate by scale (the spec's requirement), and a term that is "
 "    flat within a given candidate set correctly contributes almost nothing to "
 "    that set's decision, so the choice is made on the axis that actually "
 "    varies. The constants are frozen for the whole run because a reward whose "
 "    normalisation drifts shard to shard is not comparable across the corpus.\n"
 "\n"
 "Grouping: the emotion term is z-scored per (target kind, target name). The "
 "corpus conditions on two different things -- an emotion label (emonet expert) "
 "or a signed VoiceNet dimension -- and the 40 emotion experts are separately "
 "calibrated with different ranges. One global z-score would weight 'Anger' "
 "against 'Awe' by their scale rather than their meaning. Quality is one head "
 "on one scale, so it gets one global group.\n"
 "\n"
 "This is a stored VIEW, not a commitment: reward_set (within-set z) and "
 "reward_minmax are written to the same parquet rows, along with every raw "
 "score, so re-deciding this costs a query and not one GPU-hour."
)


def stats(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return None
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    return dict(n=int(x.size), mean=float(x.mean()), sd=float(x.std(ddof=1)),
                median=med, mad_sd=mad, p01=float(np.percentile(x, 1)),
                p99=float(np.percentile(x, 99)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=f"{PROD}/smoke/*/*/cand-*.parquet")
    ap.add_argument("--out", default=f"{PROD}/index/norm_stats.json")
    ap.add_argument("--min-n", type=int, default=200,
                    help="groups thinner than this fall back to the kind-level pool")
    a = ap.parse_args()

    files = sorted(glob.glob(a.glob))
    if not files:
        print(f"no cand parquets at {a.glob}", file=sys.stderr)
        return 2
    cols = ["target_kind", "target_name", "target_sign", "emo_target_raw",
            "qual_raw", "block"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    df["sgn"] = df.target_sign.fillna(1.0).astype(int)
    print(f"[norm] {len(files)} files, {len(df)} candidate rows")

    q = stats(df.qual_raw)
    emo = {}
    # Group by (kind, name, SIGN). A voicenet gid aims a dimension high or low
    # and the stored strength is sign x regression, so pooling both directions
    # gives a bimodal group whose sd is the distance between two modes rather
    # than a spread -- measured -2.32 vs +2.64 for the same dimension.
    for (k, n, s_), g in df.groupby(["target_kind", "target_name", "sgn"], dropna=False):
        s = stats(g.emo_target_raw)
        if s and s["n"] >= a.min_n:
            emo[f"{k}:{n}:{s_}"] = s
    for (k, s_), g in df.groupby(["target_kind", "sgn"]):
        s = stats(g.emo_target_raw)
        if s:
            emo[f"{k}:*:{s_}"] = s
    for k, g in df.groupby("target_kind"):
        s = stats(g.emo_target_raw)
        if s:
            emo[f"{k}:*"] = s
    emo["*"] = stats(df.emo_target_raw)

    # sanity: does the sd a z-score assumes match a robust estimate?
    checks = []
    for name, s in list(emo.items()) + [("QUALITY", q)]:
        if not s:
            continue
        ratio = s["sd"] / max(s["mad_sd"], 1e-9)
        checks.append(dict(group=name, n=s["n"], sd=round(s["sd"], 4),
                           mad_sd=round(s["mad_sd"], 4), sd_over_madsd=round(ratio, 3)))
    bad = [c for c in checks if c["n"] >= a.min_n and not (0.5 <= c["sd_over_madsd"] <= 2.0)]

    out = dict(built_at=time.time(), source_files=files, n_rows=int(len(df)),
               method="frozen group z-score", why=WHY,
               emo=emo, qual=q, min_n=a.min_n,
               robust_check=checks, robust_outliers=bad)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"[norm] wrote {a.out}: {len(emo)} emotion groups, "
          f"quality sd={q['sd']:.4f} (robust {q['mad_sd']:.4f}), "
          f"{len(bad)} groups where sd and robust-sd disagree >2x")
    for c in checks[:12]:
        print("   ", c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
