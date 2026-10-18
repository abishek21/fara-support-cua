#!/usr/bin/env bash
# merge_lora.sh — merge the trained LoRA adapter back into the base Fara-7B weights
# so the result can be served directly by vLLM (no runtime adapter needed).
#
# Usage:
#   bash merge_lora.sh /root/opsworld-sft-out/checkpoint-XXX
#   # -> writes a merged fp model next to the checkpoint: <ckpt>-merged
set -euo pipefail

CKPT="${1:?usage: bash merge_lora.sh <path-to-checkpoint-dir>}"

swift export \
    --adapters "${CKPT}" \
    --merge_lora true \
    --torch_dtype bfloat16

echo
echo "Merged model written to: ${CKPT}-merged"
echo "Serve it with:  bash serve_sft.sh ${CKPT}-merged"
