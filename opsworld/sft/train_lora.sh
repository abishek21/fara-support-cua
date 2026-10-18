#!/usr/bin/env bash
# train_lora.sh — LoRA SFT of microsoft/Fara-7B (Qwen2.5-VL-7B) on OpsWorld
# solver trajectories, using ms-swift. Tuned to fit on a single A40 (46 GB).
#
# Config (locked):
#   - LoRA rank 16 (alpha 32), target = all linear in the LLM
#   - ViT FROZEN (vision tower + merger not trained) -> only adapt the policy head
#   - bf16, 2 epochs
#   - gradient checkpointing + small per-device batch + grad-accum to fit 46 GB
#
# Prereqs on the pod:
#   pip install "ms-swift" "transformers>=4.49" qwen-vl-utils decord datasets
#   python prepare_swift_data.py --repo abishekk210698/opsworld-sft --out /root/opsworld-swift
#
# Usage:
#   bash train_lora.sh
set -euo pipefail

DATA_DIR="${DATA_DIR:-/root/opsworld-swift}"
MODEL="${MODEL:-microsoft/Fara-7B}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/opsworld-sft-out}"
EPOCHS="${EPOCHS:-2}"

# Keep vision inputs bounded so a 46 GB card doesn't OOM on big screenshots.
# Qwen2.5-VL uses image-pixel budgets expressed in vision tokens.
export MAX_PIXELS="${MAX_PIXELS:-1003520}"   # ~1280x784 cap per image
export SIZE_FACTOR="${SIZE_FACTOR:-28}"
export NPROC_PER_NODE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

swift sft \
    --model "${MODEL}" \
    --model_type qwen2_5_vl \
    --template qwen2_5_vl \
    --dataset "${DATA_DIR}/train.jsonl" \
    --val_dataset "${DATA_DIR}/val.jsonl" \
    --train_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --torch_dtype bfloat16 \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing true \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --max_length 20000 \
    --eval_strategy steps \
    --eval_steps 50 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --dataloader_num_workers 4 \
    --output_dir "${OUTPUT_DIR}" \
    --report_to tensorboard \
    --seed 42

echo
echo "Training done. Checkpoints under: ${OUTPUT_DIR}"
echo "Best/last adapter is in the latest checkpoint-*/ dir."
echo "Next: bash merge_lora.sh <path-to-checkpoint>"
