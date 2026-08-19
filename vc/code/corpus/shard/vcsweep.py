"""Keep ~N worker nodes alive until every shard has a verified completion marker.

Three jobs, in this order of importance:

 1. Reap dead claims. A claim file with no completion marker whose Slurm job is
    gone is a node failure, not work in progress. Over an 18 h run node failures
    are certain, so this is the difference between finishing and stalling at 97 %.
 2. Keep the worker count at the target, without starving anything else: it
    submits nothing while the partition has fewer idle nodes than `idle_floor`
    (the same safety valve the identity-repair sweeper uses).
 3. Publish state to sweep.json so progress is observable without ssh.

Config is re-read every cycle, so the target can be changed while it runs.
"""
import os, sys, json, time, subprocess, argparse
import pandas as pd

NB = "/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox"
PROD = f"{NB}/vcbon/prod"
STATE = f"{PROD}/state"
CLAIMS = f"{STATE}/claims"
CONF = f"{STATE}/sweeper.conf.json"
OUT = f"{STATE}/sweep.json"
REPAIR = f"{NB}/vprof/repair/state/sweep.json"
DEFAULT_CONF = dict(max_workers=48, idle_floor=200, burst=8, stale_grace_s=1800,
                    walltime="12:00:00", run_tag="VC1", enabled=True,
                    wait_for_repair=True, repair_stall_h=4.0)


def repair_gate(c, hist):
    """Is the identity-repair run finished?

    The brief gates this run on `vprof/repair/state/sweep.json` reaching
    done == ready, with a release if the remainder can be *shown* to be stuck.
    Both are folded into the sweeper rather than living in a second daemon,
    because over a multi-day wait the failure mode that matters is the watcher
    dying, and one long-lived process is one thing to keep alive instead of two.

    "Stuck" requires evidence, not impatience: no completion for `repair_stall_h`
    hours AND no repair worker left in the queue. A repair run that is merely
    slow -- and this one is, ~6 h per voice -- is not stuck.
    """
    if not c.get("wait_for_repair", True):
        return True, "wait_for_repair disabled in config"
    try:
        st = json.load(open(REPAIR))
    except Exception as e:
        return False, f"repair state unreadable ({e}); holding"
    done, ready = st.get("done"), st.get("ready")
    if isinstance(done, int) and isinstance(ready, int) and done >= ready:
        return True, f"repair complete ({done}/{ready})"
    n_rep = sum(1 for l in sh("squeue -u $USER -h -o '%j'").splitlines()
                if l.strip().startswith("vrepair"))
    now = time.time()
    if hist.get("done") != done:
        hist["done"], hist["since"] = done, now
    stalled_h = (now - hist.get("since", now)) / 3600.0
    if stalled_h >= c.get("repair_stall_h", 4.0) and n_rep == 0:
        return True, (f"repair stalled: no completion for {stalled_h:.1f} h and "
                      f"zero repair workers in the queue (done {done}/{ready})")
    return False, (f"repair at {done}/{ready}, {n_rep} workers, "
                   f"{stalled_h:.1f} h since last completion")


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return ""


def load_conf():
    c = dict(DEFAULT_CONF)
    if os.path.exists(CONF):
        try:
            c.update(json.load(open(CONF)))
        except Exception:
            pass
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--out-root", default=f"{NB}/vprof/vc500")
    ap.add_argument("--once", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(CLAIMS, exist_ok=True)

    df = pd.read_parquet(f"{PROD}/index/shards.parquet")
    ids = list(df.shard_id)
    total = len(ids)
    hist, gate_open, announced = {}, False, False

    while True:
        c = load_conf()
        run_tag = c["run_tag"]
        if not gate_open:
            gate_open, why = repair_gate(c, hist)
            if gate_open and not announced:
                announced = True
                msg = (f"\n## {time.strftime('%Y-%m-%d %H:%M')} — GATE RELEASED\n\n"
                       f"{why}. The VC sweeper begins submitting workers now; it "
                       f"ramps to {c['max_workers']} nodes subject to its own "
                       f"`idle_floor` of {c['idle_floor']}.\n")
                try:
                    with open(f"{NB}/vcbon/PROTOCOL.md", "a") as fh:
                        fh.write(msg)
                except Exception:
                    pass
                print(f"[sweep] GATE RELEASED: {why}", flush=True)
            elif not gate_open:
                print(f"[sweep] holding on repair gate: {why}", flush=True)

        done = set()
        for sid in ids:
            v, s = sid.split("/")
            if os.path.exists(f"{a.out_root}/{v}/{run_tag}/done-{int(s):03d}.json"):
                done.add(sid)

        # `live` is EVERY job id this user has queued or running, not just the
        # vcbon-named ones. A claim is reaped when its owning job is gone, and
        # the owner may legitimately be a differently-named job (a smoke test, a
        # one-off re-run). Matching on a name prefix reaped live claims out from
        # under a running `vcprodsmoke` job -- harmless there, but the same
        # mistake against a production worker would let two GPUs write the same
        # shard's tar concurrently.
        live = set()
        q = sh("squeue -u $USER -h -o '%i %j %T'")
        for ln in q.splitlines():
            p = ln.split()
            if p:
                live.add(p[0].split("_")[0])
        n_running = sum(1 for ln in q.splitlines()
                        if len(ln.split()) >= 3 and ln.split()[1].startswith("vcbon")
                        and ln.split()[2] == "RUNNING")
        n_pending = sum(1 for ln in q.splitlines()
                        if len(ln.split()) >= 3 and ln.split()[1].startswith("vcbon")
                        and ln.split()[2] == "PENDING")

        # ---- reap stale claims -------------------------------------------------
        reaped = 0
        claimed = set()
        now = time.time()
        for fn in os.listdir(CLAIMS):
            if not fn.endswith(".json"):
                continue
            sid = fn[:-5].replace("__", "/")
            p = f"{CLAIMS}/{fn}"
            if sid in done:
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass
                continue
            try:
                info = json.load(open(p))
            except Exception:
                info = {}
            jid = str(info.get("jobid", "")).split("_")[0]
            age = now - float(info.get("t", 0) or 0)
            if jid and jid != "local" and jid not in live and age > c["stale_grace_s"]:
                try:
                    os.remove(p); reaped += 1
                except FileNotFoundError:
                    pass
            else:
                claimed.add(sid)

        todo = total - len(done)
        unclaimed = todo - len(claimed)

        idle = 0
        for ln in sh("sinfo -p booster -h -t idle -o '%D'").splitlines():
            try:
                idle += int(ln.strip())
            except ValueError:
                pass

        # Hard interlock: without frozen normalisation constants every shard
        # would silently fall back to within-set ranking, and the corpus would be
        # ranked by two different rewards depending on when the shard ran. That
        # is not a thing you can fix afterwards from stored scores alone, because
        # it is the *shipped* winner that would be inconsistent.
        norm_ok = False
        try:
            nd = json.load(open(f"{PROD}/index/norm_stats.json"))
            norm_ok = bool(nd.get("emo")) and bool(nd.get("qual"))
        except Exception:
            norm_ok = False

        submitted = 0
        if not norm_ok:
            print("[sweep] HOLDING: index/norm_stats.json missing or unusable", flush=True)
        if norm_ok and gate_open and c.get("enabled", True) and unclaimed > 0:
            want = min(c["max_workers"], max(0, (unclaimed + 3) // 4)) - (n_running + n_pending)
            want = min(want, c["burst"])
            if want > 0 and idle >= c["idle_floor"]:
                r = sh(f"sbatch --array=1-{want} --time={c['walltime']} "
                       f"--export=ALL,VC_RUNTAG={run_tag},VC_OUTROOT={a.out_root} "
                       f"{PROD}/code/vcworker.sbatch")
                if "Submitted" in r:
                    submitted = want
                print(f"[sweep] submitted {want}: {r.strip()}", flush=True)

        st = dict(at=time.time(), ready=total, done=len(done), todo=todo,
                  claimed=len(claimed), unclaimed=unclaimed, reaped_now=reaped,
                  workers_running=n_running, workers_pending=n_pending,
                  idle_nodes=idle, submitted=submitted, run_tag=run_tag,
                  norm_ok=norm_ok, enabled=bool(c.get("enabled", True)),
                  repair_gate_open=bool(gate_open),
                  pct=round(100.0 * len(done) / max(total, 1), 2))
        tmp = OUT + ".tmp"
        json.dump(st, open(tmp, "w"), indent=1)
        os.replace(tmp, OUT)
        print(f"[sweep] {json.dumps(st)}", flush=True)

        if todo == 0:
            print("[sweep] all shards complete", flush=True)
            break
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
