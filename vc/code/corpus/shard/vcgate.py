"""Watch the identity-repair sweeper and release the VC production run when it finishes.

The brief gates the bulk launch on `vprof/repair/state/sweep.json` reaching
`done == ready`. Polling that by hand for many hours is exactly the kind of
thing that gets missed at 4 a.m., so it is a daemon: it records every
observation (so the wait is evidence, not memory), and when the gate clears it
flips `enabled` in the VC sweeper's config and starts the VC sweeper.

It also releases on a genuine stall — but only on evidence, never on a guess:
no completion for `stall_h` hours AND no repair worker running. A repair run
that is simply slow is not stalled, and the difference is the whole point of
the gate.
"""
import os, sys, json, time, subprocess, argparse

NB = "/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox"
PROD = f"{NB}/vcbon/prod"
REPAIR = f"{NB}/vprof/repair/state/sweep.json"
CONF = f"{PROD}/state/sweeper.conf.json"
HIST = f"{PROD}/state/gate_history.jsonl"
PROTO = f"{NB}/vcbon/PROTOCOL.md"


def sh(c):
    try:
        return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return ""


def note(msg):
    with open(PROTO, "a") as f:
        f.write(msg)
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--stall-h", type=float, default=4.0)
    a = ap.parse_args()

    last_done, last_change = None, time.time()
    while True:
        try:
            st = json.load(open(REPAIR))
        except Exception as e:
            st = {"err": str(e)}
        done, ready = st.get("done"), st.get("ready")
        q = sh("squeue -u $USER -h -o '%j %T'")
        n_repair = sum(1 for l in q.splitlines() if l.split() and l.split()[0].startswith("vrepair"))

        if done != last_done:
            last_done, last_change = done, time.time()
        stalled_h = (time.time() - last_change) / 3600.0

        rec = dict(t=time.time(), done=done, ready=ready, todo=st.get("todo"),
                   repair_workers=n_repair, hours_since_progress=round(stalled_h, 2))
        with open(HIST, "a") as f:
            f.write(json.dumps(rec) + "\n")

        clear = (isinstance(done, int) and isinstance(ready, int) and done >= ready)
        stall = (stalled_h >= a.stall_h and n_repair == 0)
        if clear or stall:
            why = ("done == ready (%s/%s)" % (done, ready) if clear else
                   "no repair completion for %.1f h and zero repair workers running" % stalled_h)
            c = json.load(open(CONF))
            c["enabled"] = True
            c["_released"] = dict(at=time.time(), reason=why, repair_state=st)
            tmp = CONF + ".tmp"
            json.dump(c, open(tmp, "w"), indent=1)
            os.replace(tmp, CONF)
            r = sh(f"cd {PROD} && setsid nohup {NB}/env_mossaudio/bin/python -u "
                   f"{PROD}/code/vcsweep.py > {PROD}/logs/sweep.log 2>&1 < /dev/null & disown; echo started")
            note(f"\n## {time.strftime('%Y-%m-%d %H:%M')} — GATE RELEASED\n\n"
                 f"Reason: {why}. Repair state at release: "
                 f"`done={done} ready={ready} todo={st.get('todo')}`, "
                 f"{n_repair} repair workers still in the queue. "
                 f"VC sweeper enabled and started; it will ramp to 48 nodes "
                 f"subject to its own `idle_floor` of 200.\n")
            return 0
        print(f"[gate] done={done}/{ready} repair_workers={n_repair} "
              f"stalled={stalled_h:.2f}h -- holding", flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
