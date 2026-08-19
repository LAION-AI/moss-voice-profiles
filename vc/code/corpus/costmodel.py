"""Cost model: measured s/sample -> GPU-hours, node-hours, budget share, storage."""
import json, os
import numpy as np, pandas as pd

NB = "/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox"
OUT = f"{NB}/vcbon/out"

GPU_PER_NODE = 4
CORE_H_PER_GPU_H = 72
BUDGET_CORE_H = 106e6
S1 = 20_125_736
S2_VOICES, S2_PER = 5_500, 40_256
S2 = S2_VOICES * S2_PER
MEAN_CLIP_S = 9.72
BYTES_PER_AUDIO_S = 160_000 / 8        # 160 kbps -> 20 kB/s
NODES_AVAIL = 48                       # the cap the repair pipeline already runs at

bp = json.load(open(f"{OUT}/bench_pipeline.json"))
arms = bp["arms"] if isinstance(bp, dict) else bp
ncurve = bp.get("n_curve", []) if isinstance(bp, dict) else []
best = {a["arm"]: a["s_per_sample"] for a in arms}
S_PER = arms[-1]["s_per_sample"]       # config E, the recommendation

by_n = {r["n_cand"]: min(x["s_per_sample"] for x in ncurve if x["n_cand"] == r["n_cand"])
        for r in ncurve} if ncurve else {8: S_PER}


def cost(n_samples, s_per):
    gpu_h = n_samples * s_per / 3600
    return dict(n_samples=n_samples, s_per_sample=s_per, gpu_h=gpu_h,
                node_h=gpu_h / GPU_PER_NODE,
                core_h=gpu_h * CORE_H_PER_GPU_H,
                budget_pct=gpu_h * CORE_H_PER_GPU_H / BUDGET_CORE_H * 100,
                days_on_48_nodes=gpu_h / GPU_PER_NODE / NODES_AVAIL / 24)


rows = []
for label, n in (("scenario1 (20.1 M)", S1), ("scenario2 (5,500 voices)", S2)):
    for nc in sorted(by_n):
        rows.append(dict(scenario=label, n_cand=nc, candidates=n * nc, **cost(n, by_n[nc])))
    # the config the pilot actually ran, for reference
    rows.append(dict(scenario=label, n_cand="8 (pilot config A)", candidates=n * 8,
                     **cost(n, arms[0]["s_per_sample"])))

st = []
for label, n in (("scenario1", S1), ("scenario2", S2)):
    audio_s = n * MEAN_CLIP_S
    st.append(dict(scenario=label, samples=n, audio_hours=audio_s / 3600,
                   winner_audio_TB=audio_s * BYTES_PER_AUDIO_S / 1e12,
                   all8_audio_TB=audio_s * 8 * BYTES_PER_AUDIO_S / 1e12,
                   winner_prov_GB=n * 1200 / 1e9,
                   all8_scores_GB=n * 8 * 350 / 1e9))

res = dict(constants=dict(gpu_per_node=GPU_PER_NODE, core_h_per_gpu_h=CORE_H_PER_GPU_H,
                          budget_core_h=BUDGET_CORE_H, nodes_avail=NODES_AVAIL,
                          s1_samples=S1, s2_voices=S2_VOICES, s2_per_voice=S2_PER,
                          s2_samples=S2, mean_clip_s=MEAN_CLIP_S,
                          existing_corpus_TB=3.9, existing_audio_hours=54000),
           config_arms=arms, n_curve=ncurve, recommended_s_per_sample=S_PER,
           cost=rows, storage=st)
json.dump(res, open(f"{OUT}/costmodel.json", "w"), indent=2, default=float)

print("=== cost ===")
d = pd.DataFrame(rows)
print(d.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
print("\n=== storage ===")
print(pd.DataFrame(st).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
print("\n=== both scenarios, N=8, recommended config ===")
g1 = cost(S1, by_n.get(8, S_PER)); g2 = cost(S2, by_n.get(8, S_PER))
print(f"  scenario1  {g1['gpu_h']:,.0f} GPU-h  {g1['node_h']:,.0f} node-h  "
      f"{g1['core_h']:,.0f} core-h  {g1['budget_pct']:.3f}% of budget  "
      f"{g1['days_on_48_nodes']:.2f} d on {NODES_AVAIL} nodes")
print(f"  scenario2  {g2['gpu_h']:,.0f} GPU-h  {g2['node_h']:,.0f} node-h  "
      f"{g2['core_h']:,.0f} core-h  {g2['budget_pct']:.3f}% of budget  "
      f"{g2['days_on_48_nodes']:.2f} d on {NODES_AVAIL} nodes")
tot = g1['gpu_h'] + g2['gpu_h']
print(f"  TOTAL      {tot:,.0f} GPU-h  {tot/GPU_PER_NODE:,.0f} node-h  "
      f"{tot*CORE_H_PER_GPU_H:,.0f} core-h  "
      f"{tot*CORE_H_PER_GPU_H/BUDGET_CORE_H*100:.2f}% of the 106 Mcore-h remainder")
print(f"\n  storage delta {st[0]['winner_audio_TB']+st[1]['winner_audio_TB']:.1f} TB "
      f"on top of the existing 3.9 TB")
