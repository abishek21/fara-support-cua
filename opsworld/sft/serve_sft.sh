#!/usr/bin/env bash
# serve_sft.sh — serve the merged SFT'd Fara-7B with vLLM, same contract the base
# model is served with (so solver/bench_runner.py works unchanged).
#
# Matches the base serve flags from project_context.md §11 (CUDA-13 pod):
#   VLLM_USE_FLASHINFER_SAMPLER=0 to avoid the flashinfer JIT sampler (needs ninja).
#
# Usage:
#   bash serve_sft.sh /root/opsworld-sft-out/checkpoint-XXX-merged
set -euo pipefail

MODEL_PATH="${1:?usage: bash serve_sft.sh <merged-model-dir>}"
PORT="${PORT:-5000}"

VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve "${MODEL_PATH}" \
    --served-model-name microsoft/Fara-7B \
    --port "${PORT}" \
    --dtype auto \
    --max-model-len 20000 \
    --gpu-memory-utilization 0.92 \
    --limit-mm-per-prompt '{"image": 3}'

# NOTE: --served-model-name is microsoft/Fara-7B so runpod_config.json ("model":
# "microsoft/Fara-7B") and fara-cli need no changes. Tunnel pod :5000 -> local
# :5050 (macOS owns 5000 for AirPlay), then re-run the eval (see RUNBOOK.md).
