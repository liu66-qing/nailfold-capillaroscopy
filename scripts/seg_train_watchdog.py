#!/usr/bin/env python
"""Run one training command under a stall watchdog.

On Windows, ultralytics' dataloader rebuild at close_mosaic deadlocked S0 with
workers=2 (GPU 0 %, no log output for 20 min, epoch 30/40). The watchdog waits
for results.csv to stop changing for STALL seconds, kills the process tree, and
resumes the exact run from last.pt (weights, optimizer, epoch) with workers=0.
Every arm is run the same way, so every arm gets the same treatment if it stalls.

  python scripts/seg_train_watchdog.py --run E:/nailfold_tmp/seg_runs/S1 -- \
      python scripts/mendeley_seg_train.py --name S1 ...
"""
import argparse
import os
import subprocess
import sys
import time

STALL = 900


def mtime(p):
    return os.path.getmtime(p) if os.path.exists(p) else 0.0


def done(run):
    import torch
    last = os.path.join(run, "weights", "last.pt")
    if not os.path.exists(os.path.join(run, "weights", "best.pt")):
        return False
    try:
        c = torch.load(last, map_location="cpu", weights_only=False)
        return c.get("epoch", 0) == -1          # ultralytics strips last.pt at the end
    except Exception:
        return False


def run_watched(cmd, run, log):
    with open(log, "ab") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    res = os.path.join(run, "results.csv")
    t0 = time.time()
    while p.poll() is None:
        time.sleep(30)
        last = max(mtime(res), mtime(log), t0)
        if time.time() - last > STALL:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
            p.wait()
            return "stalled"
    return "ok" if p.returncode == 0 else "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    for attempt in range(4):
        st = run_watched(cmd, a.run, a.log)
        print("attempt", attempt, st, flush=True)
        if done(a.run):
            return 0
        if st == "fail" and attempt > 0:
            return 1
        if "--resume" not in cmd:
            cmd = cmd + ["--resume"]
    return 1


if __name__ == "__main__":
    sys.exit(main())
