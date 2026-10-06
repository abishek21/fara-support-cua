"""
hf_upload.py — push the merged SFT dataset to a PRIVATE Hugging Face dataset with
images EMBEDDED (so a single `load_dataset` on the GPU pod pulls everything).

Each sample becomes a row:
    images      : Sequence(Image)   -- embedded PNG bytes (up to 3 most recent)
    messages    : string (JSON)     -- system/user/assistant; image content is a
                                       placeholder {"type":"image"} mapped by order
                                       to the `images` column
    label       : string            -- the assistant target (thought + <tool_call>)
    category    : string
    seed        : int
    id          : string

Auth: uses common.hf_auth.get_token() (env HF_TOKEN / gitignored hf_config.json / cache).

Usage:
    python -m solver.hf_upload --repo abishekk210698/opsworld-sft --pilot runs/pilot
    python -m solver.hf_upload --repo abishekk210698/opsworld-sft --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OPSROOT = os.path.dirname(HERE)
sys.path.insert(0, OPSROOT)
from common.hf_auth import get_token


def _rows(jsonl_path: str):
    rows = []
    for line in open(jsonl_path):
        line = line.strip()
        if not line:
            continue
        s = json.loads(line)
        imgs = [p for p in s.get("images", []) if os.path.exists(p)]
        # strip absolute paths from messages; keep image as a bare placeholder
        msgs = []
        for m in s["messages"]:
            c = m["content"]
            if isinstance(c, list):
                new = []
                for part in c:
                    if part.get("type") == "image":
                        new.append({"type": "image"})
                    else:
                        new.append(part)
                msgs.append({"role": m["role"], "content": new})
            else:
                msgs.append(m)
        label = s["messages"][-1]["content"] if s["messages"] else ""
        rows.append({
            "images": imgs,
            "messages": json.dumps(msgs),
            "label": label,
            "category": s.get("category", ""),
            "seed": int(s.get("seed", -1)),
            "id": s.get("id", ""),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. abishekk210698/opsworld-sft")
    ap.add_argument("--pilot", default="runs/pilot")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from datasets import Dataset, DatasetDict, Features, Sequence, Image, Value

    base = args.pilot if os.path.isabs(args.pilot) else os.path.join(OPSROOT, args.pilot)
    ds_dir = os.path.join(base, "dataset")
    train_rows = _rows(os.path.join(ds_dir, "train.jsonl"))
    val_rows = _rows(os.path.join(ds_dir, "val.jsonl"))
    print(f"train rows: {len(train_rows)}  val rows: {len(val_rows)}")
    n_imgs = sum(len(r["images"]) for r in train_rows + val_rows)
    print(f"total embedded image refs: {n_imgs}")

    feats = Features({
        "images": Sequence(Image()),
        "messages": Value("string"),
        "label": Value("string"),
        "category": Value("string"),
        "seed": Value("int64"),
        "id": Value("string"),
    })
    dd = DatasetDict({
        "train": Dataset.from_list(train_rows, features=feats),
        "validation": Dataset.from_list(val_rows, features=feats),
    })
    print(dd)

    if args.dry_run:
        print("\n[dry-run] not pushing. Sample row (train[0]) label:")
        print(train_rows[0]["label"][:300])
        return

    token = get_token()
    if not token:
        raise SystemExit("No HF token (set HF_TOKEN or opsworld/hf_config.json).")
    private = not args.public
    print(f"\nPushing to {args.repo} (private={private}) ...")
    dd.push_to_hub(args.repo, private=private, token=token)
    print("Done:", f"https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
