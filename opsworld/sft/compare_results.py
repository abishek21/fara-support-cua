"""
compare_results.py — before/after scorecard diff between two bench_runner runs.

Reads two results.json files (list of {category, success, ...} rows produced by
solver/bench_runner.py) and prints a per-category table with the delta, focusing
attention on C1_refund and C4_churn (the weak categories SFT should lift).

Usage:
    python compare_results.py \
        --before ../runs/bench_base/results.json \
        --after  ../runs/bench_sft/results.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

CATEGORIES = ["C1_refund", "C2_cancel", "C3_block", "C4_churn", "C5_below", "C6_noerror"]
FOCUS = {"C1_refund", "C4_churn"}


def rates(path):
    rows = json.load(open(path))
    by = defaultdict(list)
    for r in rows:
        by[r["category"]].append(bool(r["success"]))
    out = {}
    for c in CATEGORIES:
        xs = by.get(c, [])
        out[c] = (sum(xs), len(xs))
    allxs = [bool(r["success"]) for r in rows]
    out["OVERALL"] = (sum(allxs), len(allxs))
    return out


def pct(sl):
    s, n = sl
    return 100 * s / n if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    args = ap.parse_args()

    b = rates(args.before)
    a = rates(args.after)

    print(f"{'category':<12} {'before':>10} {'after':>10} {'delta':>9}")
    print("-" * 45)
    for c in CATEGORIES + ["OVERALL"]:
        bp, ap_ = pct(b[c]), pct(a[c])
        d = ap_ - bp
        star = "  <<<" if c in FOCUS else ""
        print(f"{c:<12} {b[c][0]:>3}/{b[c][1]:<3}{bp:>5.0f}% "
              f"{a[c][0]:>3}/{a[c][1]:<3}{ap_:>5.0f}% {d:>+7.1f}pp{star}")

    print("\nFocus categories (SFT target):")
    for c in sorted(FOCUS):
        print(f"  {c}: {pct(b[c]):.0f}% -> {pct(a[c]):.0f}%  "
              f"({pct(a[c]) - pct(b[c]):+.1f}pp)")


if __name__ == "__main__":
    main()
