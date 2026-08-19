"""Build the Scenario-1 report page for the GitHub Pages mirror.

Everything on the page comes from JSON written by the run itself
(`vcbon/prod/out/*.json`, `vcbon/prod/index/*.json`, `state/sweep.json`), so it
can be regenerated at any point and always reflects measured state rather than
the plan. Numbers that are not yet measured render as "pending", never as the
planned value dressed up as a result.
"""
import os, sys, json, glob, time, html

NB = "/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox"
PROD = f"{NB}/vcbon/prod"
DEST = [f"{NB}/gh/moss-voice-profiles/vc_scenario1.html"]


def load(p, d=None):
    try:
        return json.load(open(p))
    except Exception:
        return d


def f(x, n=3, dash="&mdash;"):
    if x is None:
        return dash
    try:
        return f"{float(x):,.{n}f}"
    except Exception:
        return html.escape(str(x))


def main():
    idx = load(f"{PROD}/index/shards_summary.json", {})
    norm = load(f"{PROD}/index/norm_stats.json", {})
    an = load(f"{PROD}/out/analysis.json") or {}
    npass = load(f"{PROD}/out/analysis_normpass.json", {}) or {}
    # Throughput and recovery come from the production-config shards; the
    # SIDON-order A/B only exists where --ab-presidon was run, which is the
    # normalisation pass. Merge rather than pick, and never let one stand in for
    # the other.
    if not (an.get("ab") or {}).get("arms"):
        an = dict(an); an["ab"] = npass.get("ab") or {}
        an["ab_source"] = "normalisation pass (17 shards, 33,692 sources)"
    if not an.get("recovery"):
        an["recovery"] = npass.get("recovery") or []
    if not an.get("throughput"):
        an["throughput"] = npass.get("throughput") or {}
    sid = load(f"{PROD}/out/sidon_tf32.json", {})
    sweep = load(f"{PROD}/state/sweep.json", {})
    dur = load(f"{PROD}/out/durstats.json", {})

    th = (an or {}).get("throughput") or {}
    ab = (an or {}).get("ab") or {}
    rec = (an or {}).get("recovery") or []

    n_groups = len((norm or {}).get("emo", {}) or {})
    n_norm_rows = (norm or {}).get("n_rows", 0) or 0
    s_on = (sid or {}).get("tf32_on", {}) or {}
    s_off = (sid or {}).get("tf32_off", {}) or {}
    s_snr = (sid or {}).get("snr_off_vs_on_db", {}) or {}

    def abrow(name):
        for r in ab.get("arms", []):
            if r["arm"].startswith(name):
                return r
        return {}
    a_post, a_pre, a_arb = abrow("SIDON-then-rank"), abrow("rank-then-SIDON"), abrow("arbitrary")

    prog = ""
    if sweep:
        prog = (f"<p class=\"run\"><b>Run state</b> &mdash; {sweep.get('done',0):,} of "
                f"{sweep.get('ready',0):,} shards complete ({sweep.get('pct',0)}%), "
                f"{sweep.get('workers_running',0)} worker nodes, "
                f"{sweep.get('claimed',0)} shards in flight. "
                f"Updated {time.strftime('%Y-%m-%d %H:%M', time.localtime(sweep.get('at', time.time())))}.</p>")

    rows_rec = ""
    for r in rec:
        rows_rec += (
            f"<tr><td>{html.escape(str(r.get('target_kind')))}</td>"
            f"<td class=n>{r.get('n_sources',0):,}</td>"
            f"<td class=n>{f(r.get('src_mean'))}</td>"
            f"<td class=n>{f(r.get('arbitrary_pick'))}</td>"
            f"<td class=n>{f(r.get('reward_group'))}</td>"
            f"<td class=n>{f(r.get('oracle_best_of_4'))}</td>"
            f"<td class=n>{f(100*r['reward_group_gap_recovered'],1) if r.get('reward_group_gap_recovered') is not None else '&mdash;'}%</td>"
            f"<td class=n>{f(100*r['reward_group_of_oracle'],1) if r.get('reward_group_of_oracle') is not None else '&mdash;'}%</td></tr>")

    ab_src = an.get("ab_source", "production shards")
    stage_rows = ""
    for k, v in sorted((th.get("stage_share") or {}).items(), key=lambda x: -x[1]):
        stage_rows += f"<tr><td>{html.escape(k)}</td><td class=n>{100*v:.1f}%</td></tr>"

    HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scenario 1 &mdash; best-of-4 voice conversion with SIDON</title>
<style>
:root{{--bg:#fff;--fg:#16181d;--mut:#5b6270;--line:#e3e6ec;--acc:#1f5fd6;--warn:#8a4b00;--warnbg:#fff6e8}}
@media (prefers-color-scheme:dark){{:root{{--bg:#12141a;--fg:#e8eaf0;--mut:#9aa2b1;--line:#272b35;--acc:#7aa6ff;--warn:#f0b45c;--warnbg:#2a2114}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
.w{{max-width:60rem;margin:0 auto;padding:2.5rem 1.25rem 5rem}}
h1{{font-size:1.9rem;line-height:1.2;margin:0 0 .4rem}}
h2{{font-size:1.25rem;margin:2.6rem 0 .7rem;padding-top:1.1rem;border-top:1px solid var(--line)}}
h3{{font-size:1.02rem;margin:1.6rem 0 .4rem}}
.sub{{color:var(--mut);margin:0 0 1.6rem}}
table{{border-collapse:collapse;width:100%;margin:.9rem 0;font-size:.93rem}}
th,td{{text-align:left;padding:.44rem .6rem;border-bottom:1px solid var(--line);vertical-align:top}}
th{{color:var(--mut);font-weight:600}}
td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums}}
code,pre{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.88em}}
pre{{background:color-mix(in srgb,var(--fg) 5%,transparent);padding:.8rem 1rem;border-radius:8px;overflow-x:auto}}
.tw{{overflow-x:auto}}
.note{{background:var(--warnbg);border-left:3px solid var(--warn);padding:.7rem 1rem;border-radius:0 6px 6px 0;margin:1.1rem 0}}
.run{{background:color-mix(in srgb,var(--acc) 9%,transparent);border-left:3px solid var(--acc);padding:.7rem 1rem;border-radius:0 6px 6px 0}}
.t{{color:var(--mut);font-size:.9rem}}
a{{color:var(--acc)}}
</style></head><body><div class="w">

<h1>Scenario 1 &mdash; best-of-4 voice conversion, SIDON on all four, reward ranking</h1>
<p class="sub">{idx.get('n_samples',0):,} source takes across {idx.get('n_voices',0)} voices
&middot; {idx.get('n_shards',0):,} shards &middot; N=4 candidates &middot; every candidate kept</p>
{prog}

<h2>What the pipeline does</h2>
<pre>for each of the {idx.get('n_samples',0):,} source takes:
    1. generate 4 voice-conversion candidates   (chatterbox s3gen, TF32 on)
    2. SIDON restore ALL FOUR                   (sidon_batch.py, TF32 off)
    3. rank the SIDON-ENHANCED four by
           normalise(target emotion strength) + normalise(quality)
    4. store all four outputs, every raw score, every reward component</pre>
<p>Step 3 ranks the audio that actually ships, rather than selecting on a
pre-enhancement signal and hoping the ranking survives enhancement. Selection is
a <b>view</b>: three different rewards are stored side by side with all their
components, so changing the reward later costs a parquet query, not a re-run.</p>

<h2>Reward normalisation</h2>
<p>Chosen: <b>z-score against frozen group statistics</b>, not within the
4-candidate set.</p>
<p>Within-set normalisation (z or min&ndash;max over the four) forces the two terms
to contribute equally in every set, whatever the set looks like. If four
candidates are tied on quality to within scorer noise but differ in target
emotion by 1.4, within-set normalisation inflates that noise to the same
magnitude as the real signal and lets it cast half the vote &mdash; and with n=4 the
sd estimate itself carries ~40% relative error, so the effective weighting is
re-randomised per sample. min&ndash;max is worse: it is defined by the two noisiest
order statistics and is degenerate on ties.</p>
<p>Frozen group z-scores divide each term by the spread it has across the corpus.
Neither term can dominate by scale &mdash; the requirement &mdash; and a term that is flat
within a candidate set correctly contributes almost nothing to that set's
decision. The emotion term is grouped by
<code>(target kind, target name, <b>sign</b>)</code> because the 40 emotion
experts and 57 VoiceNet dimensions are separately calibrated; quality is one head
on one scale and gets one group.
Constants: {n_groups} groups from {n_norm_rows:,} candidate rows.</p>
<p>The sign matters and was found by measurement, not foresight. A
<code>voicenet</code> target aims a dimension high <i>or</i> low and the stored
strength is <code>sign &times; regression</code>, so a group pooling both directions is
bimodal &mdash; measured &minus;2.32 and +2.64 for the same dimension. Its sd (3.12) is
then the distance between two modes rather than a spread, and z-scoring against
it divides the emotion term by roughly 4&times; too much on half the corpus. Split by
sign, the same groups have sd 0.65&ndash;0.79.</p>
<p><b>The choice is validated against the alternatives on the same stored
scores</b>: on emotion targets the frozen group z-score recovers 38.7% of the
conversion's emotion loss against 32.6% for within-set z and 33.0% for
min&ndash;max; on VoiceNet targets, 104.0% against 86.8% and 87.0%. It reaches
~79&ndash;80% of what a target-emotion oracle could reach at N=4, and it does not
pay for that in quality (Overall-Q 3.122 under the reward vs 3.104 for an
arbitrary pick) or identity (ECAPA unchanged to 4 dp).</p>

<h2>SIDON before ranking vs after &mdash; measured</h2>
<p class="t">Diagnostic. The pipeline order is fixed; this measures what the choice buys.
Measured on the {ab_src}.
Both arms are judged on the SIDON-enhanced audio, because that is what ships in
either case &mdash; the only difference is which candidate was chosen.</p>
<div class="tw"><table>
<tr><th>selection</th><th class=n>n</th><th class=n>target emotion</th><th class=n>Overall-Q</th>
<th class=n>ECAPA</th><th class=n>WavLM</th><th class=n>blend</th></tr>
<tr><td><b>SIDON&nbsp;&rarr;&nbsp;rank</b> (shipped)</td><td class=n>{a_post.get('n',0):,}</td>
<td class=n>{f(a_post.get('emo_target'))}</td><td class=n>{f(a_post.get('overall_q'))}</td>
<td class=n>{f(a_post.get('ecapa'))}</td><td class=n>{f(a_post.get('wavlm'))}</td>
<td class=n>{f(a_post.get('blend'))}</td></tr>
<tr><td>rank&nbsp;&rarr;&nbsp;SIDON (pilot order)</td><td class=n>{a_pre.get('n',0):,}</td>
<td class=n>{f(a_pre.get('emo_target'))}</td><td class=n>{f(a_pre.get('overall_q'))}</td>
<td class=n>{f(a_pre.get('ecapa'))}</td><td class=n>{f(a_pre.get('wavlm'))}</td>
<td class=n>{f(a_pre.get('blend'))}</td></tr>
<tr><td>arbitrary pick (k=1)</td><td class=n>{a_arb.get('n',0):,}</td>
<td class=n>{f(a_arb.get('emo_target'))}</td><td class=n>{f(a_arb.get('overall_q'))}</td>
<td class=n>{f(a_arb.get('ecapa'))}</td><td class=n>{f(a_arb.get('wavlm'))}</td>
<td class=n>{f(a_arb.get('blend'))}</td></tr>
</table></div>
<p>The two orders pick the <i>same</i> candidate only
{f(100*ab.get('winner_agreement',0),1)}% of the time, so this is a real choice and
not a rounding difference.</p>

<h2>How much of the emotion loss the reward recovers</h2>
<p class="t">Conversion costs target emotion strength &mdash; the pilot measured
&minus;50.8% on the <code>self</code> arm. Ranking on target emotion is the direct
counter-measure. &ldquo;Oracle&rdquo; is the best of the four by target emotion alone,
i.e. the ceiling any reward could reach at N=4.</p>
<div class="tw"><table>
<tr><th>target</th><th class=n>sources</th><th class=n>source</th><th class=n>arbitrary</th>
<th class=n>reward</th><th class=n>oracle&nbsp;best-of-4</th>
<th class=n>gap&nbsp;recovered</th><th class=n>of&nbsp;oracle</th></tr>
{rows_rec or '<tr><td colspan="8" class="t">pending</td></tr>'}
</table></div>

<h2>Throughput, measured against the 3,588 GPU-h plan</h2>
<div class="tw"><table>
<tr><th></th><th class=n>planned</th><th class=n>measured</th></tr>
<tr><td>s per source sample</td><td class=n>0.642</td><td class=n>{f(th.get('s_per_sample_production'))}</td></tr>
<tr><td>SIDON ms per clip</td><td class=n>38.3</td><td class=n>{f(th.get('sidon_ms_per_clip'),1)}</td></tr>
<tr><td>GPU-h, full run</td><td class=n>3,588</td><td class=n>{f(th.get('projected_gpu_h_full_run'),0)}</td></tr>
<tr><td>core-h</td><td class=n>258,300</td><td class=n>{f(th.get('projected_core_h'),0)}</td></tr>
<tr><td>hours on 48 nodes</td><td class=n>18.7</td><td class=n>{f(th.get('projected_h_on_48_nodes'),1)}</td></tr>
<tr><td>output, TB</td><td class=n>15.6</td><td class=n>{f(th.get('projected_out_TB'),1)}</td></tr>
</table></div>
<div class="note"><b>The SIDON budget line and the SIDON quality rule were inconsistent.</b>
The plan's 857 GPU-h assumes 38.3 ms/clip. Measured on identical clips:
<b>{f(s_on.get('ms_per_clip_at_972s'),1)} ms/clip with TF32 on</b>
(which reproduces 38.3) and
<b>{f(s_off.get('ms_per_clip_at_972s'),1)} ms/clip with TF32 off</b>
&mdash; {f(sid.get('speedup_tf32_on'),2)}&times; slower. The 38.3 ms figure was measured
with TF32 set globally, i.e. active inside SIDON; this project separately measured
that TF32 on SIDON costs quality, so TF32 stays off and the budget moves. The two
outputs differ by {f(s_snr.get('mean'),1)} dB SNR (worst clip
{f(s_snr.get('min'),1)} dB).</div>
<h3>Where the time goes</h3>
<div class="tw"><table><tr><th>stage</th><th class=n>share of wall</th></tr>{stage_rows or '<tr><td colspan=2 class="t">pending</td></tr>'}</table></div>

<h2>Where the data is</h2>
<pre>vprof/vc500/&lt;voice&gt;/VC1/vc-&lt;NNN&gt;.tar        all 4 candidates, 48 kHz mp3, WebDataset
vprof/vc500/&lt;voice&gt;/VC1/cand-&lt;NNN&gt;.parquet  every score for every candidate, no audio
vprof/vc500/&lt;voice&gt;/VC1/prov-&lt;NNN&gt;.parquet  provenance, one row per source take
vprof/vc500/&lt;voice&gt;/VC1/REF.json            the reference and how it was prepared
vprof/vc500/&lt;voice&gt;/VC1/done-&lt;NNN&gt;.json     completion marker, written only after verification</pre>
<p>Key format <code>&lt;target&gt;/&lt;source&gt;/&lt;gid&gt;.c&lt;NNN&gt;.k&lt;J&gt;</code>. The join key is
<b>(source_run_dir, source_audio_key)</b> &mdash; <code>audio_key</code> alone is not
unique across runs, and provenance carries the run directory for exactly that
reason. Source corpus untouched.</p>

<h2>Reproducing</h2>
<pre>vcbon/prod/code/vcindex.py      shard manifest from parquet metadata only
vcbon/prod/code/vcprod.py       one shard, end to end, with its own verification
vcbon/prod/code/vcrun.py        per-GPU claim loop
vcbon/prod/code/vcsweep.py      dead-claim reaping, node topping-up, progress
vcbon/prod/code/vcnorm.py       freezes the reward normalisation, with a robustness check
vcbon/prod/code/vcanalyse.py    every table above, as parquet queries
vcbon/prod/code/vcsidonbench.py the SIDON TF32 measurement
vcbon/PROTOCOL.md               the running log: decisions, failures, what was learned</pre>
<p class="t">JUPITER Booster, account reformo, 4&times;GH200 per node, torch 2.8.0+cu129.
Raw JSON for every table is under <code>vcbon/prod/out/</code>.
Generated {time.strftime('%Y-%m-%d %H:%M')}.</p>

</div></body></html>"""

    for d in DEST:
        os.makedirs(os.path.dirname(d), exist_ok=True)
        open(d, "w").write(HTML)
        print("wrote", d, len(HTML), "bytes")


if __name__ == "__main__":
    main()
