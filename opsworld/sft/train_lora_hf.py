"""
train_lora_hf.py — LoRA SFT of microsoft/Fara-7B (Qwen2.5-VL-7B) using *plain*
transformers + peft (no ms-swift). Single A40 (46 GB), transparent and
version-robust.

What it does, matching the locked config:
  - LoRA rank 16 / alpha 32 on the LANGUAGE-model linear layers only
    (q/k/v/o/gate/up/down_proj). The ViT (visual.*) gets NO adapter and the base
    is frozen -> vision tower is fully frozen.
  - bf16, 2 epochs, grad-accum 16, gradient checkpointing -> fits 46 GB.
    - Final-assistant-only loss: all earlier turns, including prior actions and
        screenshots, are masked (-100). Qwen's image-token expansion is entirely
        inside the masked prompt.

Data: use prepare_swift_data.py --format fara to build Fara-native jsonl:
    {"messages":[{"role":"system","content":"..."},
                                 {"role":"user","content":"<image>Task ..."},
                                 {"role":"assistant","content":"prior action"},
                                 {"role":"user","content":"<image>Current URL: ..."},
                 {"role":"assistant","content":"thought\n<tool_call>...</tool_call>"}],
         "images":["images/step0.png", "images/step1.png"]}

Usage:
    python train_lora_hf.py \
        --model /root/.cache/modelscope/models/microsoft--Fara-7B/snapshots/master \
        --data_dir /root/opsworld-fara-native \
        --output_dir /root/opsworld-sft-native-out \
        --epochs 2
"""
from __future__ import annotations

import argparse
import json
import os
import re

import torch
from PIL import Image


IMG_TAG = "<image>"


# ----------------------------- data -----------------------------------------
def parse_content(text: str):
    """Split a stored content string with leading <image> tags into a Qwen-style
    content list: [{"type":"image"} * n, {"type":"text","text": rest}]."""
    n = text.count(IMG_TAG)
    rest = text.replace(IMG_TAG, "")
    parts = [{"type": "image"} for _ in range(n)]
    if rest.strip():
        parts.append({"type": "text", "text": rest})
    return parts, n


def to_messages(row):
    """Convert a stored row into (messages_for_template, n_images_in_user)."""
    msgs = []
    n_total = 0
    for m in row["messages"]:
        if m["role"] == "assistant":
            msgs.append({"role": "assistant",
                         "content": [{"type": "text", "text": m["content"]}]})
        else:
            parts, n = parse_content(m["content"])
            n_total += n
            msgs.append({"role": m["role"], "content": parts})
    return msgs, n_total


class JsonlDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.image_root = os.path.dirname(os.path.abspath(path))
        self.rows = [json.loads(l) for l in open(path) if l.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i].copy()
        row["images"] = [p if os.path.isabs(p) else os.path.join(self.image_root, p)
                         for p in row.get("images", [])]
        return row


# ----------------------------- collator -------------------------------------
class Collator:
    def __init__(self, processor, max_length):
        self.p = processor
        self.max_length = max_length

    def _one(self, row):
        messages, _ = to_messages(row)
        images = [Image.open(p).convert("RGB") for p in row.get("images", [])]

        # full conversation (with assistant answer)
        full_text = self.p.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        # prompt only (everything up to & including the assistant header)
        prompt_text = self.p.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True)

        full = self.p(text=[full_text], images=images or None,
                      return_tensors="pt")
        # prompt length in tokens WITH the same image expansion (images are all
        # in the prompt), so the boundary is exact.
        prompt = self.p(text=[prompt_text], images=images or None,
                        return_tensors="pt")
        plen = prompt["input_ids"].shape[1]

        input_ids = full["input_ids"][0]
        labels = input_ids.clone()
        labels[:plen] = -100  # mask the prompt (incl. image tokens)

        item = {
            "input_ids": input_ids,
            "attention_mask": full["attention_mask"][0],
            "labels": labels,
        }
        if "pixel_values" in full:
            item["pixel_values"] = full["pixel_values"]
            item["image_grid_thw"] = full["image_grid_thw"]
        return item

    def __call__(self, batch):
        items = [self._one(r) for r in batch]
        maxlen = max(it["input_ids"].shape[0] for it in items)
        if maxlen > self.max_length:
            raise ValueError(f"sample length {maxlen} exceeds max_length {self.max_length}; "
                             "truncation would invalidate image grids or the action label")
        pad_id = self.p.tokenizer.pad_token_id or self.p.tokenizer.eos_token_id

        input_ids, attn, labels = [], [], []
        for it in items:
            ids = it["input_ids"][:maxlen]
            am = it["attention_mask"][:maxlen]
            lb = it["labels"][:maxlen]
            pad = maxlen - ids.shape[0]
            if pad > 0:
                ids = torch.cat([ids, torch.full((pad,), pad_id, dtype=ids.dtype)])
                am = torch.cat([am, torch.zeros(pad, dtype=am.dtype)])
                lb = torch.cat([lb, torch.full((pad,), -100, dtype=lb.dtype)])
            input_ids.append(ids)
            attn.append(am)
            labels.append(lb)

        out = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attn),
            "labels": torch.stack(labels),
        }
        # Qwen2.5-VL: pixel_values are flattened patches; concat across the batch.
        pvs = [it["pixel_values"] for it in items if "pixel_values" in it]
        if pvs:
            out["pixel_values"] = torch.cat(pvs, dim=0)
            out["image_grid_thw"] = torch.cat(
                [it["image_grid_thw"] for it in items if "image_grid_thw" in it], dim=0)
        return out


# ----------------------------- main -----------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local Fara-7B dir (modelscope snapshot) or HF id")
    ap.add_argument("--data_dir", default="/root/opsworld-swift")
    ap.add_argument("--output_dir", default="/root/opsworld-sft-out")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=12000)
    ap.add_argument("--max_pixels", type=int, default=12845056)
    ap.add_argument("--min_pixels", type=int, default=3136)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--logging_steps", type=int, default=5)
    ap.add_argument("--save_steps", type=int, default=50)
    ap.add_argument("--resume_from_checkpoint", default=None)
    args = ap.parse_args()

    from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,
                              Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model

    print("Loading processor ...")
    processor = AutoProcessor.from_pretrained(
        args.model, max_pixels=args.max_pixels, min_pixels=args.min_pixels)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print("Loading model (bf16) ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa")
    model.config.use_cache = False

    # freeze the vision tower explicitly (belt-and-suspenders; LoRA also skips it)
    n_frozen = 0
    for name, p in model.named_parameters():
        if name.startswith("visual.") or ".visual." in name:
            p.requires_grad_(False)
            n_frozen += 1
    print(f"ViT params frozen: {n_frozen}")

    lora = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    train_ds = JsonlDataset(os.path.join(args.data_dir, "train.jsonl"))
    val_ds = JsonlDataset(os.path.join(args.data_dir, "val.jsonl"))
    print(f"train={len(train_ds)}  val={len(val_ds)}")
    collator = Collator(processor, args.max_length)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        dataloader_num_workers=4,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        label_names=["labels"],
        seed=42,
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(os.path.join(args.output_dir, "final"))
    processor.save_pretrained(os.path.join(args.output_dir, "final"))
    print("Done. Adapter saved to", os.path.join(args.output_dir, "final"))


if __name__ == "__main__":
    main()
