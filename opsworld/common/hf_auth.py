"""
hf_auth.py — resolve the Hugging Face token WITHOUT ever committing it.

Looks for the token in this order:
  1. $HF_TOKEN environment variable
  2. opsworld/hf_config.json        (gitignored)   -> {"hf_token": "hf_..."}
  3. ~/.cache/huggingface/token     (from `hf auth login`)

Usage:
    from common.hf_auth import get_token, whoami_name
    api = HfApi(token=get_token())
"""
from __future__ import annotations

import json
import os


def get_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok.strip()
    cfg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "hf_config.json")
    if os.path.exists(cfg):
        try:
            return json.load(open(cfg)).get("hf_token", "").strip() or None
        except Exception:
            pass
    cached = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(cached):
        return open(cached).read().strip() or None
    return None


def whoami_name() -> str:
    from huggingface_hub import whoami
    tok = get_token()
    if not tok:
        raise SystemExit("No HF token found (set HF_TOKEN or opsworld/hf_config.json).")
    return whoami(token=tok)["name"]


if __name__ == "__main__":
    print("HF user:", whoami_name())
