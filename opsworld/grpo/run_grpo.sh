#!/usr/bin/env bash
# verl v0.9.1; run from an environment with verl, Fara, and Firefox installed.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to base Fara or a merged SFT model}"
GPU_COUNT="${GPU_COUNT:-2}"
TOTAL_STEPS="${TOTAL_STEPS:-1}"
LOGGERS='[console]'
if [[ -n "${WANDB_API_KEY:-}" ]] || python -c 'import netrc; assert netrc.netrc().authenticators("api.wandb.ai")' >/dev/null 2>&1; then
    LOGGERS='[console,wandb]'
fi
if [[ -n "${ADAPTER_PATH:-}" ]]; then
    printf '%s\n' 'Merge the SFT adapter into MODEL_PATH before GRPO: verl disables LoRA for the KL reference.' >&2
    exit 1
fi
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export OPSWORLD_GRPO_OUTPUT="${OPSWORLD_GRPO_OUTPUT:-${ROOT}/runs/grpo_rollouts}"
export OPSWORLD_GRPO_REWARD_LOG="${OPSWORLD_GRPO_REWARD_LOG:-${OPSWORLD_GRPO_OUTPUT}/rewards.jsonl}"
export OPSWORLD_GRPO_MAX_ROUNDS="${OPSWORLD_GRPO_MAX_ROUNDS:-15}"
export FARA_POST_ACTION_SLEEP="${FARA_POST_ACTION_SLEEP:-0.6}"

cd "${ROOT}"
python -c 'import inspect; from verl.utils.fsdp_utils import collect_lora_params; assert "refusing full-model summon" in inspect.getsource(collect_lora_params), "Apply grpo/verl_lora_sync.patch to the pinned verl checkout before GRPO"'
python -m verl.trainer.main_ppo \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "actor_rollout_ref.model.lora_rank=16" \
    "actor_rollout_ref.model.lora_alpha=32" \
    'actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]' \
    'actor_rollout_ref.actor.freeze_vision_tower=true' \
    'actor_rollout_ref.actor.fsdp_config.model_dtype=bf16' \
    'actor_rollout_ref.ref.fsdp_config.model_dtype=bf16' \
    'actor_rollout_ref.model.use_fused_kernels=true' \
    'actor_rollout_ref.model.fused_kernel_options.impl_backend=torch' \
    'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1' \
    'actor_rollout_ref.actor.ppo_mini_batch_size=1' \
    'actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768' \
    'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1' \
    'actor_rollout_ref.actor.use_kl_loss=true' \
    'actor_rollout_ref.actor.kl_loss_coef=0.001' \
    'actor_rollout_ref.rollout.name=vllm' \
    'actor_rollout_ref.rollout.mode=async' \
    'actor_rollout_ref.rollout.n=4' \
    'actor_rollout_ref.rollout.temperature=1.0' \
    'actor_rollout_ref.rollout.calculate_log_probs=true' \
    'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1' \
    'actor_rollout_ref.rollout.tensor_model_parallel_size=2' \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=true' \
    'actor_rollout_ref.rollout.load_format=safetensors' \
    'actor_rollout_ref.rollout.layered_summon=true' \
    'actor_rollout_ref.rollout.max_model_len=32768' \
    'actor_rollout_ref.rollout.multi_turn.enable=true' \
    'actor_rollout_ref.rollout.agent.default_agent_loop=opsworld' \
    'actor_rollout_ref.rollout.agent.num_workers=1' \
    "actor_rollout_ref.rollout.agent.agent_loop_config_path=${ROOT}/grpo/agent_loop.yaml" \
    'algorithm.adv_estimator=grpo' \
    'algorithm.use_kl_in_reward=false' \
    'reward_model.enable=false' \
    'reward.num_workers=1' \
    "data.train_files=${ROOT}/runs/grpo_data/train.parquet" \
    "data.val_files=${ROOT}/runs/grpo_data/val.parquet" \
    'data.train_batch_size=1' \
    'data.shuffle=false' \
    'data.max_prompt_length=4096' \
    'data.max_response_length=24000' \
    'data.filter_overlong_prompts=false' \
    'data.dataloader_num_workers=0' \
    'trainer.project_name=opsworld' \
    "trainer.experiment_name=${EXPERIMENT_NAME:-fara-base-grpo}" \
    "trainer.logger=${LOGGERS}" \
    'trainer.val_before_train=false' \
    'trainer.test_freq=-1' \
    'trainer.save_freq=10' \
    'trainer.max_actor_ckpt_to_keep=2' \
    "trainer.total_training_steps=${TOTAL_STEPS}" \
    "trainer.n_gpus_per_node=${GPU_COUNT}" \
    "$@"