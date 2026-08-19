"""Per-GPU worker loop: claim a shard, convert it, repeat until time runs out.

Claiming is an O_CREAT|O_EXCL create on a shared filesystem, which is the only
primitive that is actually atomic here. A claim without a completion marker is
either in progress or dead; the sweeper decides which by asking Slurm whether
the claiming job still exists, so a node failure costs one shard, not the run.

Shards are claimed in an order rotated by worker identity so 192 workers do not
all collide on shard 0 every wave.
"""
import os, sys, json, time, socket, subprocess, argparse, random
import pandas as pd

NB = "/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox"
PROD = f"{NB}/vcbon/prod"
CODE = f"{PROD}/code"
STATE = f"{PROD}/state"
CLAIMS = f"{STATE}/claims"


def claim(shard_id, info):
    p = f"{CLAIMS}/{shard_id.replace('/', '__')}.json"
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        json.dump(info, f)
    return True


def release(shard_id):
    try:
        os.remove(f"{CLAIMS}/{shard_id.replace('/', '__')}.json")
    except FileNotFoundError:
        pass


def done_path(shard_id, out_root, run_tag):
    v, sh = shard_id.split("/")
    return f"{out_root}/{v}/{run_tag}/done-{int(sh):03d}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--out-root", default=f"{NB}/vprof/vc500")
    ap.add_argument("--run-tag", default="VC1")
    ap.add_argument("--deadline-s", type=int, default=0, help="stop claiming new shards after this")
    ap.add_argument("--per-shard-budget-s", type=int, default=9000)
    ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--shards", default="", help="comma list; overrides the queue (smoke tests)")
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    os.makedirs(CLAIMS, exist_ok=True)

    t_start = time.time()
    jobid = os.environ.get("SLURM_JOB_ID", "local")
    host = socket.gethostname()
    ident = f"{jobid}:{host}:gpu{a.rank}"

    if a.shards:
        # explicit list (smoke / normalisation pass): deal it out round-robin so
        # every worker gets a disjoint slice and no two touch the same shard
        allsh = [s for s in a.shards.split(",") if s]
        queue = allsh[a.rank::a.world] if a.world > 1 else allsh
    else:
        df = pd.read_parquet(f"{PROD}/index/shards.parquet")
        # biggest shards first: the tail of an 18 h run is decided by stragglers
        df = df.sort_values("n_samples", ascending=False).reset_index(drop=True)
        ids = list(df.shard_id)
        off = (hash(ident) % max(len(ids), 1)) if a.world > 1 else 0
        queue = ids[off:] + ids[:off]

    n_done = 0
    log = f"{PROD}/logs/worker-{jobid}-{a.rank}.jsonl"
    for shard_id in queue:
        if a.deadline_s and time.time() - t_start > a.deadline_s:
            print(f"[vcrun] {ident}: deadline reached, stopping", flush=True)
            break
        if a.max_shards and n_done >= a.max_shards:
            break
        if os.path.exists(done_path(shard_id, a.out_root, a.run_tag)):
            continue
        info = dict(ident=ident, jobid=jobid, host=host, gpu=a.rank, t=time.time())
        if not claim(shard_id, info):
            continue
        if os.path.exists(done_path(shard_id, a.out_root, a.run_tag)):
            continue                     # won the race but someone finished it first
        cmd = [sys.executable, "-u", f"{CODE}/vcprod.py", "--shard", shard_id,
               "--out-root", a.out_root, "--run-tag", a.run_tag]
        if a.extra:
            cmd += a.extra.split()
        t0 = time.time()
        print(f"[vcrun] {ident} -> {shard_id}", flush=True)
        try:
            rc = subprocess.call(cmd, timeout=a.per_shard_budget_s)
        except subprocess.TimeoutExpired:
            rc = 124
        dt = time.time() - t0
        rec = dict(ident=ident, shard=shard_id, rc=rc, wall_s=dt, t=time.time())
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if rc == 0 and os.path.exists(done_path(shard_id, a.out_root, a.run_tag)):
            n_done += 1
            print(f"[vcrun] {ident} OK {shard_id} {dt:.0f}s", flush=True)
        else:
            # release so the sweeper (or another worker) can retry it
            release(shard_id)
            print(f"[vcrun] {ident} FAIL {shard_id} rc={rc} {dt:.0f}s -> released", flush=True)
    print(f"[vcrun] {ident} finished, {n_done} shards", flush=True)


if __name__ == "__main__":
    main()
