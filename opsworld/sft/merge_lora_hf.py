"""
merge_lora_hf.py — merge the peft LoRA adapter back into Fara-7B base weights,
producing a plain HF model directory that vLLM can serve directly.

Usage:
    python merge_lora_hf.py \
        --base /root/.cache/modelscope/models/microsoft--Fara-7B/snapshots/master \
        --adapter /root/opsworld-sft-out/final \
        --out /root/opsworld-sft-merged
"""
from __future__ import annotations

import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", default="/root/opsworld-sft-merged")
    args = ap.parse_args()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    print("Loading base ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16)
    print("Applying adapter ...")
    model = PeftModel.from_pretrained(model, args.adapter)
    print("Merging ...")
    model = model.merge_and_unload()
    print("Saving merged model ->", args.out)
    model.save_pretrained(args.out, safe_serialization=True)
    # ship the processor/tokenizer alongside so vLLM can load standalone
    AutoProcessor.from_pretrained(args.base).save_pretrained(args.out)
    print("Done. Serve with:  bash serve_sft.sh", args.out)


if __name__ == "__main__":
    main()
