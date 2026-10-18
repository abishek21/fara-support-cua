"""Write train-only OpsWorld task prompts for verl's RLHFDataset."""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset

from grpo.episode import training_task
from solver.phrase import CATEGORY


def rows(seeds: range, categories=None):
    for seed in seeds:
        for category in categories or CATEGORY.values():
            task = training_task(seed, category)
            yield {
                "data_source": "opsworld",
                "prompt": [{"role": "user", "content": task["instruction"]}],
                "extra_info": {"seed": seed, "category": category,
                               "index": task["task_id"]},
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/grpo_data"))
    parser.add_argument("--smoke-seed", type=int)
    parser.add_argument("--smoke-category", choices=tuple(CATEGORY.values()))
    args = parser.parse_args()
    if (args.smoke_seed is None) != (args.smoke_category is None):
        parser.error("--smoke-seed and --smoke-category must be supplied together")
    if args.smoke_seed is not None and not 1 <= args.smoke_seed < 300:
        parser.error("--smoke-seed must be a train seed (1-299)")
    args.out.mkdir(parents=True, exist_ok=True)
    train_seeds = range(args.smoke_seed, args.smoke_seed + 1) if args.smoke_seed is not None else range(1, 300)
    val_seeds = range(300, 301) if args.smoke_seed is not None else range(300, 319)
    categories = [args.smoke_category] if args.smoke_category is not None else None
    for name, seeds in (("train", train_seeds), ("val", val_seeds)):
        data = Dataset.from_list(list(rows(seeds, categories)))
        output = args.out / f"{name}.parquet"
        data.to_parquet(str(output))
        print(f"{name}: {len(data)} tasks -> {output}")


if __name__ == "__main__":
    main()