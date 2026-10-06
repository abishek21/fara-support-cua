"""
make_eval_set.py — materialize the fixed, held-out evaluation set (C1-C6).

Each task uses a UNIQUE held-out seed (disjoint from training seeds), so every
model is evaluated on an identical set of fresh worlds it never trained on. Tasks
store the instruction + the start system + the ground-truth `verify` spec; the
benchmark runner reseeds both systems to the task's seed before running a model.

Usage:
    python -m tasks.make_eval_set --per-category 20 --seed-base 9000
Output: opsworld/tasks/eval_tasks.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OPSROOT = os.path.dirname(HERE)
sys.path.insert(0, OPSROOT)
from common.world import generate_world
from solver.phrase import phrase, CATEGORY

CATEGORIES = ["C1_refund", "C2_cancel", "C3_block", "C4_churn", "C5_below", "C6_noerror"]
KIND_BY_CATEGORY = {v: k for k, v in CATEGORY.items()}
OUT = os.path.join(HERE, "eval_tasks.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=9000)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    need = {c: args.per_category for c in CATEGORIES}
    tasks = []
    seed = args.seed_base
    rr = 0
    # round-robin categories; one unique seed per task
    while any(need[c] > 0 for c in CATEGORIES):
        pending = [c for c in CATEGORIES if need[c] > 0]
        cat = pending[rr % len(pending)]
        rr += 1
        kind = KIND_BY_CATEGORY[cat]
        accounts, scns = generate_world(seed)
        scn = next((s for s in scns if s.kind == kind), None)
        if scn is None:
            seed += 1
            continue
        a = accounts[scn.account_id]
        start_system = "crm" if kind == "reconcile_churn" else "billing"
        tasks.append({
            "task_id": f"{cat}_seed{seed}",
            "category": cat,
            "kind": kind,
            "seed": seed,
            "account_id": scn.account_id,
            "account_name": a.name,
            "start_system": start_system,
            "instruction": phrase(scn, accounts),
            "verify": scn.verify,
        })
        need[cat] -= 1
        seed += 1

    with open(args.out, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    from collections import Counter
    c = Counter(t["category"] for t in tasks)
    print(f"Wrote {len(tasks)} eval tasks -> {args.out}")
    print("per-category:", dict(sorted(c.items())))
    print(f"seed range: {tasks[0]['seed']}..{tasks[-1]['seed']} (held-out)")
    print("\nsample task:")
    print(json.dumps(tasks[0], indent=2)[:600])


if __name__ == "__main__":
    main()
