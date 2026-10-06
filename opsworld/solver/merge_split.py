"""
merge_split.py — merge all verified per-run SFT samples into a single dataset,
split into train/val by TRAJECTORY (no step leakage), stratified by category.

Only verified runs have `sft_samples.jsonl` (the collector converts passing
trajectories only), so merging those is already the "verified-only" filter.

Usage:
    python -m solver.merge_split --pilot runs/pilot --val-frac 0.05
Outputs under <pilot>/dataset/:
    train.jsonl, val.jsonl, stats.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OPSROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="runs/pilot")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = args.pilot if os.path.isabs(args.pilot) else os.path.join(OPSROOT, args.pilot)
    rng = random.Random(args.seed)

    # gather trajectories: {category: [(run_dir, [samples...]), ...]}
    by_cat: dict[str, list] = defaultdict(list)
    for sf in glob.glob(os.path.join(base, "*", "seed*", "sft_samples.jsonl")):
        run_dir = os.path.dirname(sf)
        category = os.path.basename(os.path.dirname(run_dir))
        samples = [json.loads(l) for l in open(sf) if l.strip()]
        if samples:
            # tag each sample for provenance
            seed = os.path.basename(run_dir).replace("seed", "")
            for s in samples:
                s["category"] = category
                s["seed"] = int(seed)
            by_cat[category].append((run_dir, samples))

    if not by_cat:
        raise SystemExit(f"no sft_samples.jsonl found under {base}")

    train, val = [], []
    traj_counts = {"train": Counter(), "val": Counter()}
    sample_counts = {"train": Counter(), "val": Counter()}

    for cat, trajs in sorted(by_cat.items()):
        rng.shuffle(trajs)
        n_val = max(1, round(len(trajs) * args.val_frac)) if len(trajs) > 1 else 0
        val_trajs = trajs[:n_val]
        train_trajs = trajs[n_val:]
        for _, samples in val_trajs:
            val.extend(samples)
            traj_counts["val"][cat] += 1
            sample_counts["val"][cat] += len(samples)
        for _, samples in train_trajs:
            train.extend(samples)
            traj_counts["train"][cat] += 1
            sample_counts["train"][cat] += len(samples)

    rng.shuffle(train)
    rng.shuffle(val)

    out_dir = os.path.join(base, "dataset")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "train.jsonl"), "w") as f:
        for s in train:
            f.write(json.dumps(s) + "\n")
    with open(os.path.join(out_dir, "val.jsonl"), "w") as f:
        for s in val:
            f.write(json.dumps(s) + "\n")

    stats = {
        "trajectories": {"train": dict(traj_counts["train"]),
                         "val": dict(traj_counts["val"]),
                         "total_train": sum(traj_counts["train"].values()),
                         "total_val": sum(traj_counts["val"].values())},
        "samples": {"train": dict(sample_counts["train"]),
                    "val": dict(sample_counts["val"]),
                    "total_train": len(train), "total_val": len(val)},
        "val_frac": args.val_frac, "split_by": "trajectory (no step leakage)",
    }
    json.dump(stats, open(os.path.join(out_dir, "stats.json"), "w"), indent=2)

    print("=== DATASET MERGED + SPLIT ===")
    print(f"out: {out_dir}")
    print(f"trajectories: train={stats['trajectories']['total_train']} "
          f"val={stats['trajectories']['total_val']}")
    print(f"SFT samples:  train={len(train)} val={len(val)}")
    print("\nper-category trajectories (train / val):")
    for cat in sorted(by_cat):
        print(f"  {cat:<12} {traj_counts['train'][cat]:>3} / {traj_counts['val'][cat]}")
    print("\nper-category samples (train):")
    for cat in sorted(by_cat):
        print(f"  {cat:<12} {sample_counts['train'][cat]}")


if __name__ == "__main__":
    main()
