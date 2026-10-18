"""
collect.py — parallel, rate-limited expert-trajectory collector for OpsWorld.

For each attempt: reseed an ISOLATED worker environment to a unique seed (fresh
world), phrase a scenario of the target category, run the GPT-5.4 SoM solver,
verify across both DBs, and KEEP only passing trajectories (converted to Fara SFT
samples). Per-worker isolation (own CRM+Billing servers + DBs) makes parallelism
safe; a shared RateLimiter keeps us under the Azure RPM limit.

Usage (headless, 6 workers, 120 RPM, 50 verified/category):
    cd opsworld
    python -m solver.collect --workers 6 --rpm 120 --per-category 50 \
        --out runs/pilot --train-seed-base 1

Outputs per kept trajectory:
    <out>/<category>/seed<seed>/{stepN.png, trajectory.json, sft_samples.jsonl}
and a manifest <out>/manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
OPSROOT = os.path.dirname(HERE)
sys.path.insert(0, OPSROOT)
from common.world import generate_world
from common.azure_llm import AzureLLM
from solver.gpt_solver import solve
from solver.rate_limiter import RateLimiter
from solver.phrase import phrase, CATEGORY
from solver.convert_trajectory import convert_run
from verify import verify_task

CATEGORIES = ["C1_refund", "C2_cancel", "C3_block", "C4_churn", "C5_below", "C6_noerror"]
KIND_BY_CATEGORY = {v: k for k, v in CATEGORY.items()}


def _get(u):
    with urllib.request.urlopen(u, timeout=15) as r:
        return json.load(r)


def _post(u, b=None):
    r = urllib.request.Request(u, data=json.dumps(b or {}).encode(),
                               headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(r, timeout=15) as resp:
        return json.load(resp)


def spawn_env(w: int, base_dir: str) -> dict:
    cport, bport = 8300 + 2 * w, 8301 + 2 * w
    wd = os.path.join(base_dir, f"w{w}")
    os.makedirs(wd, exist_ok=True)
    env = dict(os.environ, CRM_DB=os.path.join(wd, "crm.db"),
               BILLING_DB=os.path.join(wd, "billing.db"))
    devnull = open(os.devnull, "w")
    try:
        crm = subprocess.Popen([sys.executable, "-m", "uvicorn", "crm.app:app",
                                "--port", str(cport), "--log-level", "warning"],
                               cwd=OPSROOT, env=env, stdout=devnull, stderr=devnull)
        bill = subprocess.Popen([sys.executable, "-m", "uvicorn", "billing.app:app",
                                 "--port", str(bport), "--log-level", "warning"],
                                cwd=OPSROOT, env=env, stdout=devnull, stderr=devnull)
    finally:
        devnull.close()
    crm_url, bill_url = f"http://localhost:{cport}", f"http://localhost:{bport}"
    for _ in range(80):
        try:
            _get(crm_url + "/api/snapshot"); _get(bill_url + "/api/snapshot")
            return {"w": w, "crm": crm_url, "billing": bill_url, "procs": [crm, bill]}
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"worker {w} env failed to start")


class TaskSource:
    """Thread-safe source of attempts, balanced across categories until each
    reaches `per_category` verified (or the attempt cap is hit)."""
    def __init__(self, per_category: int, seed_base: int, max_attempts: int):
        self.per_category = per_category
        self.verified = {c: 0 for c in CATEGORIES}
        self.attempts = 0
        self.max_attempts = max_attempts
        self._seed = seed_base
        self._lock = threading.Lock()
        self._rr = 0

    def next(self):
        with self._lock:
            if self.attempts >= self.max_attempts:
                return None
            pending = [c for c in CATEGORIES if self.verified[c] < self.per_category]
            if not pending:
                return None
            cat = pending[self._rr % len(pending)]
            self._rr += 1
            seed = self._seed
            self._seed += 1
            self.attempts += 1
            return {"seed": seed, "category": cat, "kind": KIND_BY_CATEGORY[cat]}

    def record(self, cat: str, ok: bool):
        with self._lock:
            if ok:
                self.verified[cat] += 1

    def status(self):
        with self._lock:
            return dict(self.verified), self.attempts


def run_worker(wenv, source, llm, rl, out_dir, manifest_lock, manifest_path,
               max_steps, send_image):
    crm_url, bill_url = wenv["crm"], wenv["billing"]
    while True:
        item = source.next()
        if item is None:
            return
        seed, cat, kind = item["seed"], item["category"], item["kind"]
        try:
            accounts, scns = generate_world(seed)
            scn = next((s for s in scns if s.kind == kind), None)
            if scn is None:
                source.record(cat, False); continue
            instruction = phrase(scn, accounts)
            start = crm_url if kind == "reconcile_churn" else bill_url
            task = {"task_id": f"{cat}_seed{seed}", "instruction": instruction,
                    "start_url": start + "/", "verify": scn.verify}

            _post(crm_url + f"/api/reset?seed={seed}")
            _post(bill_url + f"/api/reset?seed={seed}")
            before = {"crm": _get(crm_url + "/api/snapshot"),
                      "billing": _get(bill_url + "/api/snapshot")}

            run_out = os.path.join(out_dir, cat, f"seed{seed}")
            result = solve(task, headful=False, max_steps=max_steps, out_dir=run_out,
                           send_image=send_image, verbose=False,
                           crm_url=crm_url, billing_url=bill_url, llm=llm, rate_limiter=rl)

            after = {"crm": _get(crm_url + "/api/snapshot"),
                     "billing": _get(bill_url + "/api/snapshot")}
            v = verify_task(scn.verify, before, after)
            ok = v["success"]
            source.record(cat, ok)

            rec = {"category": cat, "kind": kind, "seed": seed, "verified": ok,
                   "n_steps": result["n_steps"], "reason": v["reason"], "dir": run_out}
            if ok:
                conv = convert_run(run_out, instruction=instruction)
                rec["n_samples"] = conv.get("n_samples", 0)
                rec["convert_ok"] = conv["ok"]
            with manifest_lock:
                open(manifest_path, "a").write(json.dumps(rec) + "\n")
            vstat, att = source.status()
            print(f"[w{wenv['w']}] {cat} seed{seed}: "
                  f"{'PASS' if ok else 'FAIL'} ({result['n_steps']}s steps) | "
                  f"verified={vstat} attempts={att}", flush=True)
        except Exception as e:
            source.record(cat, False)
            print(f"[w{wenv['w']}] {cat} seed{seed}: ERROR {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rpm", type=int, default=120)
    ap.add_argument("--per-category", type=int, default=50)
    ap.add_argument("--train-seed-base", type=int, default=1)
    ap.add_argument("--max-attempts", type=int, default=0, help="0 = auto (per_cat*6*4)")
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--no-image", action="store_true")
    ap.add_argument("--out", default="runs/pilot")
    args = ap.parse_args()

    out_dir = args.out if os.path.isabs(args.out) else os.path.join(OPSROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    manifest_lock = threading.Lock()

    max_attempts = args.max_attempts or (args.per_category * len(CATEGORIES) * 4)
    source = TaskSource(args.per_category, args.train_seed_base, max_attempts)
    llm = AzureLLM()
    rl = RateLimiter(rpm=args.rpm, max_concurrent=args.workers)

    base_env_dir = os.path.join(out_dir, "_envs")
    print(f"Launching {args.workers} isolated worker environments...")
    envs = [spawn_env(w, base_env_dir) for w in range(args.workers)]
    print("All environments up. Collecting...\n")

    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_worker, envs[w], source, llm, rl, out_dir,
                              manifest_lock, manifest_path, args.max_steps,
                              not args.no_image) for w in range(args.workers)]
            for f in futs:
                f.result()
    finally:
        for e in envs:
            for p in e["procs"]:
                p.terminate()
        vstat, att = source.status()
        dt = time.time() - t0
        total = sum(vstat.values())
        print(f"\n==== DONE in {dt/60:.1f} min ====")
        print(f"verified per category: {vstat}")
        print(f"total verified: {total}  attempts: {att}  "
              f"yield: {100*total/max(1,att):.0f}%")
        print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
