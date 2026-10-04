"""
benchmark.py — run Fara against the Support Console tasks and auto-score them.

For each task in tasks.jsonl:
    1. POST /api/reset            (canonical DB state)
    2. GET  /api/snapshot         (before)
    3. run Fara-7B via fara-cli on the support console
    4. GET  /api/snapshot         (after)
    5. verify_task(before, after) (execution-based DB check)

Produces a scorecard + bench_results.json. Each run's console log + screenshots
are saved under runs/bench/<task_id>/.

Prereqs:
    - support server running on :8000  (python -m uvicorn app:app --port 8000)
    - Fara endpoint reachable          (runpod_config.json -> tunnel on :5000)

Usage:
    cd support_env
    python benchmark.py                 # headless batch
    python benchmark.py --headful       # watch the browser
    python benchmark.py --task refund_single
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

from verify import verify_task

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SUPPORT_URL = "http://localhost:8000"
ENDPOINT_CONFIG = os.path.join(ROOT, "runpod_config.json")
RUNS_DIR = os.path.join(ROOT, "runs", "bench")


def final_answer_asks(run_dir: str) -> bool:
    """Heuristic: did the agent's final answer actually ask a clarifying question?"""
    log = os.path.join(run_dir, "console.log")
    if not os.path.exists(log):
        return False
    text = open(log, errors="ignore").read()
    import re
    m = re.search(r"Final Answer:\s*(.*)", text)
    tail = (m.group(1) if m else text[-800:]).lower()
    cues = ["?", "which ", "could you", "can you clarify", "please provide",
            "please confirm", "need the", "need more", "specify", "who is",
            "approval", "confirm whether", "let me know"]
    return any(c in tail for c in cues)


def _get(path: str) -> dict:
    with urllib.request.urlopen(SUPPORT_URL + path, timeout=10) as r:
        return json.load(r)


def _post(path: str, body: dict | None = None) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        SUPPORT_URL + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


def load_tasks(only: str | None, tasks_file: str = "tasks.jsonl"):
    wanted = set(only.split(",")) if only else None
    tasks = []
    path = tasks_file if os.path.isabs(tasks_file) else os.path.join(HERE, tasks_file)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if wanted is None or t["task_id"] in wanted:
                tasks.append(t)
    return tasks


def run_fara(task: dict, headful: bool, max_rounds: int, trial: int = 0) -> str:
    """Run fara-cli on the support console for one task; return the run folder."""
    out_dir = os.path.join(RUNS_DIR, task["task_id"], f"trial_{trial+1}")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "fara-cli", "--fara-7b",
        "--task", task["instruction"],
        "--endpoint_config", ENDPOINT_CONFIG,
        "--start_page", SUPPORT_URL + "/",
        "--save_screenshots",
        "--max_rounds", str(max_rounds),
        "--output_folder", out_dir,
    ]
    if headful:
        cmd.append("--headful")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    log_path = os.path.join(out_dir, "console.log")
    with open(log_path, "w") as log:
        proc = subprocess.run(
            cmd, input="\n\n", text=True, env=env,
            stdout=log, stderr=subprocess.STDOUT,
        )
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None, help="run a single task_id (or comma-separated)")
    ap.add_argument("--tasks_file", default="tasks.jsonl", help="task file to load")
    ap.add_argument("--headful", action="store_true")
    ap.add_argument("--max_rounds", type=int, default=20)
    ap.add_argument("--runs", type=int, default=1, help="trials per task")
    args = ap.parse_args()

    tasks = load_tasks(args.task, args.tasks_file)
    if not tasks:
        sys.exit("no tasks matched")

    # sanity: servers reachable
    try:
        _get("/api/customers")
    except Exception as e:
        sys.exit(f"support server not reachable at {SUPPORT_URL}: {e}")

    N = args.runs
    per_task = []  # [{task_id, passes, trials:[...]}]
    print(f"\nRunning {len(tasks)} task(s) x {N} trial(s) against Fara-7B...\n")
    for t in tasks:
        tid = t["task_id"]
        trials = []
        for i in range(N):
            _post("/api/reset")
            before = _get("/api/snapshot")
            t0 = time.time()
            out_dir = run_fara(t, args.headful, args.max_rounds, trial=i)
            dt = time.time() - t0
            after = _get("/api/snapshot")
            res = verify_task(t, before, after)
            # Optional: for "must ask" tasks, also require a clarifying question.
            if res["success"] and t.get("must_ask"):
                if not final_answer_asks(out_dir):
                    res = {"success": False,
                           "reason": "no DB change, but did not ask a clarifying question",
                           "mutations": res.get("mutations", {})}
            res.update({"trial": i + 1, "seconds": round(dt, 1), "run_dir": out_dir})
            trials.append(res)
            mark = "PASS ✅" if res["success"] else "FAIL ❌"
            print(f"  {tid} trial {i+1}/{N}: {mark} ({res['seconds']}s) — {res['reason']}")
        passes = sum(tr["success"] for tr in trials)
        per_task.append({
            "task_id": tid,
            "passes": passes,
            "trials_n": N,
            "avg_success": passes / N,
            "pass_at_n": any(tr["success"] for tr in trials),
            "decision": t.get("decision"),
            "trials": trials,
        })
        print(f"  -> {tid}: {passes}/{N} passed\n")

    # scorecard
    n_tasks = len(per_task)
    avg_rate = sum(p["avg_success"] for p in per_task) / n_tasks
    pass_at_n = sum(p["pass_at_n"] for p in per_task)
    print("=" * 66)
    print(f"SCORECARD  ({N} trial(s)/task)")
    print(f"  Avg success rate : {avg_rate*100:.1f}%   (mean over all trials)")
    print(f"  pass@{N}          : {pass_at_n}/{n_tasks} tasks solved at least once")
    print("=" * 66)
    for p in per_task:
        bar = "".join("✅" if tr["success"] else "❌" for tr in p["trials"])
        print(f"  {p['task_id']:<26} {p['passes']}/{N} {bar}")
    out_file = os.path.join(RUNS_DIR, "bench_results.json")
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({"runs": N, "avg_success_rate": avg_rate,
                   "pass_at_n": pass_at_n, "per_task": per_task}, f, indent=2)
    print(f"\nFull results: {out_file}")


if __name__ == "__main__":
    main()
