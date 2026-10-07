"""
bench_runner.py — evaluate a Fara model on the held-out OpsWorld eval set.

Runs the ACTUAL Fara agent (fara-cli, native CUA) against isolated CRM+Billing
environments reseeded to each task's held-out seed, then scores with the
multi-system verifier. Model-agnostic: point --endpoint at whichever Fara vLLM
endpoint (base, SFT'd, or GRPO'd). Produces a per-category scorecard.

Prereq: a Fara endpoint reachable via the config (e.g. RunPod tunnel on :5050).

Usage:
    cd opsworld
    python -m solver.bench_runner --endpoint ../runpod_config.json \
        --workers 2 --runs 1 --out runs/bench_base --label base
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.request
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OPSROOT = os.path.dirname(HERE)
sys.path.insert(0, OPSROOT)
from solver.collect import spawn_env, _get, _post   # reuse isolated-env machinery
from verify import verify_task

CATEGORIES = ["C1_refund", "C2_cancel", "C3_block", "C4_churn", "C5_below", "C6_noerror"]


def load_eval(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def run_fara(task, crm_url, bill_url, endpoint, start_url, out_dir, max_rounds):
    """Run fara-cli on one task against the given isolated env. Returns run dir."""
    os.makedirs(out_dir, exist_ok=True)
    aug = (task["instruction"]
           + f"\n\nThe Contoso CRM system is at {crm_url} and the Contoso Billing "
             f"system is at {bill_url}. Use only these two systems; to switch "
             f"systems, navigate to the appropriate URL.")
    cmd = ["fara-cli", "--fara-7b", "--task", aug,
           "--endpoint_config", endpoint, "--start_page", start_url,
           "--save_screenshots", "--max_rounds", str(max_rounds),
           "--output_folder", out_dir]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with open(os.path.join(out_dir, "console.log"), "w") as log:
        subprocess.run(cmd, input="\n\n", text=True, env=env,
                       stdout=log, stderr=subprocess.STDOUT)
    return out_dir


def worker(wenv, work_q, endpoint, out_root, max_rounds, results, rlock):
    crm_url, bill_url = wenv["crm"], wenv["billing"]
    while True:
        try:
            task, run_idx = work_q.get_nowait()
        except queue.Empty:
            return
        try:
            seed = task["seed"]
            _post(crm_url + f"/api/reset?seed={seed}")
            _post(bill_url + f"/api/reset?seed={seed}")
            before = {"crm": _get(crm_url + "/api/snapshot"),
                      "billing": _get(bill_url + "/api/snapshot")}
            start = (crm_url if task["start_system"] == "crm" else bill_url) + "/"
            out_dir = os.path.join(out_root, task["task_id"], f"run{run_idx}")
            run_fara(task, crm_url, bill_url, endpoint, start, out_dir, max_rounds)
            after = {"crm": _get(crm_url + "/api/snapshot"),
                     "billing": _get(bill_url + "/api/snapshot")}
            res = verify_task(task["verify"], before, after)
            with rlock:
                results.append({"task_id": task["task_id"], "category": task["category"],
                                "run": run_idx, "success": res["success"],
                                "reason": res["reason"]})
            print(f"[w{wenv['w']}] {task['task_id']} run{run_idx}: "
                  f"{'PASS' if res['success'] else 'FAIL'}", flush=True)
        except Exception as e:
            with rlock:
                results.append({"task_id": task["task_id"], "category": task["category"],
                                "run": run_idx, "success": False, "reason": f"ERROR {e}"})
            print(f"[w{wenv['w']}] {task['task_id']} run{run_idx}: ERROR {e}", flush=True)
        finally:
            work_q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=os.path.join(OPSROOT, "tasks", "eval_tasks.jsonl"))
    ap.add_argument("--endpoint", default=os.path.join(OPSROOT, "..", "runpod_config.json"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--max-rounds", type=int, default=25)
    ap.add_argument("--out", default="runs/bench_base")
    ap.add_argument("--label", default="model")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N tasks")
    args = ap.parse_args()

    endpoint = os.path.abspath(args.endpoint)
    tasks = load_eval(args.tasks)
    if args.limit:
        tasks = tasks[:args.limit]
    out_root = args.out if os.path.isabs(args.out) else os.path.join(OPSROOT, args.out)
    os.makedirs(out_root, exist_ok=True)

    work_q = queue.Queue()
    for r in range(args.runs):
        for t in tasks:
            work_q.put((t, r))

    print(f"Eval '{args.label}': {len(tasks)} tasks x {args.runs} runs = "
          f"{len(tasks)*args.runs} rollouts on {args.workers} workers")
    envs = [spawn_env(w, os.path.join(out_root, "_envs")) for w in range(args.workers)]
    print("envs up. running...\n")

    results, rlock = [], threading.Lock()
    t0 = time.time()
    try:
        threads = [threading.Thread(target=worker, args=(
            envs[w], work_q, endpoint, out_root, args.max_rounds, results, rlock))
            for w in range(args.workers)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    finally:
        for e in envs:
            for p in e["procs"]:
                p.terminate()

    # ---- scorecard ----
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["success"])
    json.dump(results, open(os.path.join(out_root, "results.json"), "w"), indent=2)

    print(f"\n==== SCORECARD [{args.label}] ({time.time()-t0:.0f}s) ====")
    overall = []
    for c in CATEGORIES:
        xs = by_cat.get(c, [])
        overall += xs
        rate = 100 * sum(xs) / len(xs) if xs else 0
        print(f"  {c:<12} {sum(xs):>3}/{len(xs):<3}  {rate:5.1f}%")
    rate = 100 * sum(overall) / len(overall) if overall else 0
    print(f"  {'OVERALL':<12} {sum(overall):>3}/{len(overall):<3}  {rate:5.1f}%")
    print(f"results: {os.path.join(out_root, 'results.json')}")


if __name__ == "__main__":
    main()
