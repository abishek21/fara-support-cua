#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="/root/verl/.venv/bin:${PATH}"
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 RAY_memory_usage_threshold=0.85
export OPSWORLD_GRPO_OUTPUT="${OPSWORLD_GRPO_OUTPUT:-${ROOT}/runs/base_grpo_100}"
RUN_LOG="${OPSWORLD_GRPO_RUN_LOG:-runs/base_grpo_100.log}"
RUN_EXIT="${OPSWORLD_GRPO_RUN_EXIT:-runs/base_grpo_100.exit}"

python -c 'import netrc; assert netrc.netrc().authenticators("api.wandb.ai")' || {
    printf '%s\n' 'Run wandb login as root before starting training.' >&2
    exit 1
}

cd "${ROOT}"
mkdir -p runs
set +e
MODEL_PATH=/root/models/Fara-7B TOTAL_STEPS=100 EXPERIMENT_NAME=base-grpo \
bash grpo/run_grpo.sh \
    '+actor_rollout_ref.model.override_config.attn_implementation=sdpa' \
    'actor_rollout_ref.rollout.enforce_eager=true' \
    'actor_rollout_ref.rollout.max_num_seqs=16' \
    'actor_rollout_ref.rollout.gpu_memory_utilization=0.35' \
    "data.train_files=${ROOT}/runs/grpo_data/train.parquet" \
    "data.val_files=${ROOT}/runs/grpo_data/val.parquet" \
    2>&1 | tee "${RUN_LOG}"
exit_code=${PIPESTATUS[0]}
printf '%s\n' "${exit_code}" > "${RUN_EXIT}"
exit "${exit_code}"