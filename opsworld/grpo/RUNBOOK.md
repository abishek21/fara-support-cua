# OpsWorld GRPO: local state and next pod

## What is saved (2026-10-10)

- Complete corrected SFT training output (37 files, 1.8 GB):
  `opsworld/runs/sft_native_training_run/`. Includes TensorBoard events,
  checkpoints 220/230/236, final adapter, optimizer and trainer state.
  File counts and SHA-256 for final and checkpoint-236 weights and trainer state
  were compared with the departing pod. Keep this directory local; `runs/` is ignored.
- Earlier preserved corrected adapters: `opsworld/runs/sft_native_checkpoints/`
  (steps 100, 120, 236, final). The failed older SFT's checkpoint-50 is **not**
  from this run. Corrected training ended at step **236**, not 237.
- Private Hub PEFT adapters: `abishekk210698/opsworld-fara-native-step100`
  and `abishekk210698/opsworld-fara-native-step236`. Both refer to base
  `microsoft/Fara-7B`; they are **not** merged model weights. The training
  trajectories are in private dataset `abishekk210698/opsworld-sft`.
- Local corrected native dataset: `opsworld/runs/sft_native/` (1900 train,
  104 val). Held-out task definitions are in `opsworld/tasks/eval_tasks.jsonl`.
- Baseline: 66/120. Corrected SFT step 100 and final: each 120/120 on one
  rollout per held-out task. Raw local results: `runs/bench_base/`,
  `runs/bench_sft_native_step100/`, `runs/bench_sft_native_full/`.

The full SFT run is **not** needed for GRPO from base or from the saved adapter.
It is retained for reproducibility, logs, or a future exact SFT resume.
If a future SFT resume needs it, restore the entire backup from the Mac with
`scp -P <PORT> -i ~/.ssh/id_ed25519 -r runs/sft_native_training_run
root@<HOST>:/root/opsworld-sft-native-out` while in `opsworld/`. Use the
trainer's `--resume_from_checkpoint` with an existing restored checkpoint;
the saved `final/` directory is an adapter, not an optimizer checkpoint.

## Local GRPO environment

`grpo/episode.py` generates only training seeds 1-318 and provides
`OpsWorldEpisode(worker, output_dir, max_rounds=20)`. Call `await start(seed,
category)` to get Fara-native OpenAI-style system/user messages with an image;
sample one assistant response externally; pass its complete text to
`await step(response)` to get `(next_messages, done, reward, verifier_details)`.
Repeat until done, then `await close()` in `finally`. Reward is 1 only for an
exact multi-DB verifier pass, otherwise 0; malformed/off-sandbox actions fail.
Each episode has an isolated pair of CRM and Billing servers, seeded before
browser launch. Workers use ports `8300+2*worker` and `8301+2*worker`; assign
unique worker numbers to concurrently running episodes. Local smoke tests:

```bash
cd opsworld
../fara/.venv/bin/python -m unittest grpo.test_episode -q
```

The **verl v0.9.1** adapter is `grpo/verl_loop.py`, registered by
`grpo/agent_loop.yaml`. Its `OpsWorldAgentLoop.run` samples one assistant action
per turn with verl's `LLMServerClient.generate`, executes it in the local
episode, appends a screenshot observation with continuous-token alignment,
masks observation tokens from policy loss, and returns terminal verifier
`reward_score`. Parallel group members lease distinct CRM/Billing ports.
`grpo/test_verl_loop.py` validates the sampler/response-mask/image/reward
contract with a fake verl server; `grpo/test_episode.py` uses a real browser.

The **grouping and optimizer live inside pinned verl**, not our adapter:
`verl/trainer/ppo/ray_trainer.py` repeats each task `rollout.n` times in
`RayPPOTrainer.fit`, calls `async_rollout_manager.generate_sequences`, then
`compute_advantage(..., num_repeat=rollout.n)` and `_update_actor(batch)`.
`grpo/run_grpo.sh` sets `rollout.n=4`, `algorithm.adv_estimator=grpo`, 4 tasks
per batch and 16 sampled episodes per update. Set `ppo_mini_batch_size=4`:
verl multiplies it by `rollout.n`, yielding a 16-episode actor minibatch.
The pinned default actor optimizer is AdamW at 1e-6, one PPO epoch and 0.2
clip ratio; the launcher uses LoRA rank 16/alpha 32, frozen vision tower,
temperature 1, KL loss coefficient 0.001, max 15 browser actions by default
(overridable with `OPSWORLD_GRPO_MAX_ROUNDS`), up to 512
tokens per action, 4096 initial prompt tokens, 24000 response tokens and
32768 total context tokens. Checkpoints are saved every 10 updates and at
the last step. No current pod has run a GPU
training step with the unrestricted 20-action setting; see the bounded 2xA40
smoke below. Do not extrapolate its memory fit to full-length trajectories.

The continuous-token stream retains all screenshots to stay append-only,
whereas the baseline Fara CLI trims to the latest 3 images. This training vs.
evaluation context difference must be measured before interpreting a GRPO
gain. Watch prompt/response token budgets on 20-step tasks; if a sampled
context exceeds the configured budgets the job must fail visibly rather than
silently truncate image tokens. C5/C6 no-change rewards cannot distinguish
investigation from immediate termination; check for policy collapse. Train
seeds 1-318 only; eval seeds 9000-9119 must never enter training.

## Bootstrap details and prior pod failures

1. Provision GPU(s) only after checking memory for the chosen verl rollout +
   trainer configuration; 2xH100-80GB is a **planning estimate**, not a
   validated requirement. The previous pod was an A40 with CUDA 13, vLLM
   0.31.0, `transformers==4.51.3`, `peft==0.15.2` for SFT/merge. Use separate
   Python environments for vLLM, trainer, and browser if dependencies conflict.
2. Transfer the **current working tree** (not just a Git clone): `grpo/`,
   `common/`, `crm/`, `billing/`, `solver/`, `tasks/`, `sft/`, `verify.py`.
   Several local files are untracked and would not appear in a clone. Exclude
   `runs/`, `*.db`, `server.log`, `hf_config.json` and cloud credentials.
   The Fara clone is at revision
   `a675d6d61c41c47ae87bacefeab22caad18e3e84`, with a **local edit** in
   `src/fara/fara_7b/fara_agent.py` making `FARA_POST_ACTION_SLEEP` configurable;
   transfer that patched file separately or apply the equivalent change.
3. On a bare pod, Python lives at `/opt/conda/bin/python`. Create a venv from
  it, install the Fara clone (`pip install -e /root/fara`), browser packages
  (`fastapi`, `uvicorn`, Playwright Firefox via
  `python -m playwright install firefox`), then verl **v0.9.1** (commit
  `1876b06d0a3e4e71e06230be10af14492ca8a75b`) with its pinned dependencies.
  Check compatibility with Fara's Qwen2.5-VL processor before training;
  the prior SFT/merge used `transformers==4.51.3` but verl may require a
  different stack. The environment starts two uvicorn processes per rollout
  and needs Firefox. Run `python -m unittest grpo.test_episode grpo.test_verl_loop`
  inside `opsworld` before GPUs.
  On the 2xA40 pod, verl's lockfile required Python 3.12 (the image had 3.11).
  `uv sync --python 3.12 --extra fsdp --extra vllm --no-dev` hit a 404 for
  its pinned flash-attn 2.8.3 wheel; the rest installed with
  `--no-install-package flash-attn`. Install the patched Fara checkout into
  `/root/verl/.venv` too, including its declared dependencies. The browser
  and mock-verl tests passed there. Actor initialization otherwise requires
  FlashAttention2; `+actor_rollout_ref.model.override_config.attn_implementation=sdpa`
  is being tested as a fallback for the one-step smoke. Verl's default FP32
  FSDP actor OOMed during `collect_lora_params` with 35.7 GiB actor and 8.6 GiB
  vLLM resident on one A40, before any browser rollout. The launcher now sets
  both actor and reference `fsdp_config.model_dtype=bf16`, matching SFT.
  A BF16 one-step retry with `rollout.enforce_eager=true` and
  `rollout.max_num_seqs=16` stayed below 12 GiB per A40 at its last snapshot,
  but was manually stopped after over six minutes without starting an episode
  or completing weight sync. Its trailing SIGTERM/core-dump message reflects
  that manual stop, not an observed GPU OOM. It reached checkpoint-manager
  initialization but not trainer-ready: the initial vLLM replica sleep did not
  return. Disabling sleep reached trainer-ready but then OOMed during LoRA
  weight extraction on an A40. A later capped retry with sleep enabled reached
  trainer-ready, then failed at the first `checkpoint_manager.update_weights()`:
  Ray killed workers under pod RAM pressure. The pod is limited to 100 GB RAM
  (`/sys/fs/cgroup/memory.max`); its peak reached that limit. Ray's first kill
  listed two actor workers at ~20.7 GB each and two vLLM TP workers at ~16.9 GB
  each. This is not disk storage or GPU VRAM exhaustion. The launcher now uses
  `rollout.load_format=safetensors` and `rollout.layered_summon=true`, matching
  verl's LoRA recipe: vLLM loads the same base checkpoint directly instead of
  using dummy weights that force a full-base transfer on the first sync. The
  pod has all four Fara safetensors shards. A subsequent attempt reached
  trainer-ready without a new RAM OOM, but `layered_summon` returned empty on
  Fara and verl fell back to full FSDP summon; do not assume layered adapter
  collection is working. That attempt stopped when verl's default `wandb`
  logger requested an API key at the start of `fit()`. The launcher now selects
  console-only logging. The next capped run reached four running rollout
  episodes, but Ray began killing agent-loop workers as pod RAM rose to ~97 GB
  out of 100 GB; no episodes had finished at the last observation. Ray's first
  kill listed two actor workers at ~18.7 GB each and two vLLM TP workers at
  ~17.1 GB each. The launcher now limits agent-loop workers to 4 (the task
  batch size) and reward workers to 2 instead of verl's default 8 each.
  Verify RAM headroom and completed episodes on the next run before increasing
  concurrency. The reduced-worker run also failed: Ray killed three agent-loop
  workers under RAM pressure, then vLLM TP rank 1 reported a CUDA unspecified
  launch failure during first inference and the engine died. No episode result
  was published to the transfer queue. After stopping the run, a fresh
  `CUDA_VISIBLE_DEVICES=0` PyTorch tensor allocation succeeded, but the same
  test with `CUDA_VISIBLE_DEVICES=1` failed with `cudaErrorDevicesUnavailable`
  even though `nvidia-smi` showed 0 MiB used and no ECC errors. A standalone
  TP2 image+LoRA probe (`grpo/probe_vllm.py`) therefore failed at device
  initialization before generating tokens. A standalone TP1 probe with
  `CUDA_VISIBLE_DEVICES=0`, the OpsWorld screenshot, and the SFT LoRA generated
  54 tokens successfully; GPU 0 returned to 0 MiB afterward. This validates
  single-GPU multimodal inference, not TP2 or GPU 1. Do not retry TP2 GRPO on this pod
  until GPU 1 can allocate a tensor in a fresh process. Recover through the
  provider's pod restart/reprovisioning controls if available; do not infer
  that the model or LoRA kernel caused the original launch failure from the
  later device-unavailable result. No policy update has completed yet.

   Three startup warnings in the BF16 log are separate from that stall:
   - `No CUDA runtime is found` comes from PyTorch's CUDA-extension lookup in
     the CPU-side trainer controller and `GlobalRequestLoadBalancer`. Check
     CUDA availability in the vLLM GPU workers and `nvidia-smi` if inference
     fails; this warning alone does not mean the GPU workers lack CUDA.
   - `LoRA dynamic loading & unloading is enabled` is expected for verl's
     local training server, which synchronizes a trainable adapter to vLLM.
     Do not expose this API server to untrusted clients; no GRPO flag change
     is needed for this local run.
   - `generation_config.json` overrides vLLM's default sampling parameters,
     including `max_tokens: 24000`. The agent loop explicitly sets
     `max_tokens` to at most 512 per request, and the launcher sets
     temperature to 1. The model default does not cause a 24000-token action.
    Use `--generation-config vllm` only if changing the model's other default
    sampling values is intentional; it does not address the memory failure.
4. Download `microsoft/Fara-7B` as base (the Fara clone provides
  `scripts/download_model.py`). Authenticate to HF **on the pod**, not in
  checked-in config. For SFT-to-GRPO download the private step-236 adapter
  and merge it with `sft/merge_lora_hf.py`; for base-to-GRPO omit it.
  If serving either model for eval, `sft/serve_sft.sh` uses
   `VLLM_USE_FLASHINFER_SAMPLER=0` (CUDA-13 flashinfer JIT workaround),
   `--max-model-len 20000`, `--limit-mm-per-prompt '{"image": 3}'` and
   served name `microsoft/Fara-7B`. Do not load vLLM and a memory-heavy
   trainer onto the same single A40. Keep long-lived serving in tmux; a
   one-shot SSH background command previously gave exit 255.
5. Eval uses local tunnel **5050** to pod **5000** (macOS AirPlay owns 5000):
   `ssh -N -L 5050:localhost:5000 -p <PORT> -i ~/.ssh/id_ed25519 root@<HOST>`.
   Update `runpod_config.json` as needed; never commit keys. Run the existing
   `solver.bench_runner` with `FARA_POST_ACTION_SLEEP=0.6`, 3 workers, 1 run,
   `--max-rounds 20 --screens fail`, and a new output directory. Score with
   the exact two-DB verifier, not an LLM judge. See `sft/RUNBOOK.md` for the
   SFT/merge/serve commands and `project_context.md` for task taxonomy.

## Replacement 2xA40 pod smoke (2026-10-11)

On the replacement pod, both GPUs and a standalone two-rank PyTorch NCCL
all-reduce passed. vLLM 0.24 TP2 image inference stalled with and without the
SFT adapter at default settings, but passed with
`disable_custom_all_reduce=True`: 58 base-model tokens and 54 LoRA tokens.
The rollout launcher now passes this option through verl's
`rollout.engine_kwargs.vllm`; the probe exposes the same optional flag.

A one-step TP2 GRPO smoke with one task, four rollouts, one agent-loop worker,
one reward worker, BF16 FSDP, and Ray's 85% memory guard reached trainer-ready
but failed at the initial weight sync, before any episode or optimizer update.
`layered_summon` returned empty for Fara and verl fell back to a full FSDP
summon; Ray killed a worker at 79.18/93.13 GB reported node usage. The 100 GB
container cgroup recorded a near-limit peak, so do not raise the memory guard
and retry TP2 without reducing the sync peak. Orphaned vLLM workers were
stopped and both GPUs returned to 0 MiB.

A separate single-GPU TP1 training-pipeline smoke (one worker per role) failed
earlier during vLLM initialization: at 35% GPU memory utilization, no KV-cache
blocks were available beside the actor (21.42/44.42 GB used after FSDP).
Increasing vLLM's reservation without freeing actor GPU memory is unsafe.
Neither smoke produced a completed GRPO optimizer update.

### Pinned verl fixes for the 2xA40 pod

The replacement pod uses verl 0.9.1 at `1876b06d0a3e4e71e06230be10af14492ca8a75b`.
Transfer `grpo/verl_lora_sync.patch`, `grpo/verl_sdpa.patch`, and
`grpo/test_lora_sync.py` alongside the launcher. Apply the patches once to
the clean verl checkout, **before** starting GRPO:

```bash
patch --dry-run -p1 -d /root/verl < grpo/verl_lora_sync.patch
patch --dry-run -p1 -d /root/verl < grpo/verl_sdpa.patch
patch -p1 -d /root/verl < grpo/verl_lora_sync.patch
patch -p1 -d /root/verl < grpo/verl_sdpa.patch
/root/verl/.venv/bin/torchrun --standalone --nproc_per_node=2 -m unittest grpo.test_lora_sync -v
/root/verl/.venv/bin/python -m unittest grpo.test_verl_loop -q
```

Run these from `opsworld/` with the verl virtualenv installed. The LoRA patch
initializes the root FSDP state, extracts fully qualified PEFT adapter keys
one nested FSDP unit at a time, and raises if no adapters are found; it never
falls back to a full-model summon in layered mode. The two-rank regression
checks names, tensor values, and the empty-result guard. The separate Qwen-VL
patch supplies causal SDPA when FlashAttention2 is absent (no padding,
sliding-window attention, or sequence parallelism); it preserves boundaries
between packed sequences. Do not apply either patch a second time. Revalidate
both when upgrading verl or installing FlashAttention2.
The launcher checks that the fail-closed LoRA patch is present before starting
the trainer. The custom agent loop also emits numeric `min_global_steps` and
`max_global_steps`: verl requires them for post-update metrics. When the vLLM
server reports no version, the loop uses the synchronized dispatch step minus
one; otherwise it tracks the reported range across all turns.

The bounded one-step command on the replacement pod is:

```bash
RAY_memory_usage_threshold=0.85 VLLM_USE_FLASHINFER_SAMPLER=0 \
OPSWORLD_GRPO_MAX_ROUNDS=6 \
MODEL_PATH=/root/models/Fara-7B TOTAL_STEPS=1 EXPERIMENT_NAME=smoke-bounded \
bash grpo/run_grpo.sh \
  '+actor_rollout_ref.model.override_config.attn_implementation=sdpa' \
  'actor_rollout_ref.rollout.enforce_eager=true' \
  'actor_rollout_ref.rollout.max_num_seqs=16' \
  'actor_rollout_ref.rollout.gpu_memory_utilization=0.35' \
  'actor_rollout_ref.rollout.agent.num_workers=1' 'reward.num_workers=1' \
  'data.train_batch_size=1' 'actor_rollout_ref.actor.ppo_mini_batch_size=1' \
  'trainer.save_freq=1'
```

`OPSWORLD_GRPO_MAX_ROUNDS` bounds actions per episode for the A40 smoke;
the launcher now defaults to 15. It does not omit screenshots or skip the DB verifier.
Unrestricted 20-turn sampling hit GPU 0's 44.42 GiB limit in actor
`logprobs_from_logits_v2` during a later retry, though a shorter sampled
episode completed an update and wrote both rank-0/rank-1 checkpoint shards.
Do not treat a six-turn smoke as evidence that full-length GRPO fits on A40s.
For a repeatable verifier-reward check, make a separate one-task train file
without changing `runs/grpo_data/`:

```bash
python -m grpo.make_dataset --smoke-seed 1 --smoke-category C5_below \
  --out runs/grpo_smoke_c5
```

Pass `data.train_files=<absolute path>/runs/grpo_smoke_c5/train.parquet`
and `data.val_files=<absolute path>/runs/grpo_smoke_c5/val.parquet` to the
bounded launch. The smoke generator rejects eval seeds and writes a separate
validation task on seed 300. A completed update with four identical rewards
has zero group-relative advantage, so also inspect `critic/score/{min,max}`
and `actor/grad_norm` in verl's step metrics before claiming a learning signal.

**Measured bounded result (2xA40, six turns, train seed 1, C5_below):** one
step finished in 156.77 seconds with four verifier scores spanning 0 to 1
(mean 0.75), `actor/pg_loss=-0.27028`, `actor/grad_norm=1.03906`, and
`training/global_step=1`. Verifier score is the strict DB-diff result; the
trainable LoRA rank-0 shard changed in all 584 adapter tensors compared with
the zero-gradient control checkpoint. In particular, 292 LoRA-B tensors have
13,194,939 nonzero rank-0 entries after training versus 0 in the control.
Both rank-0/rank-1 FSDP model and
optimizer shards saved to `checkpoints/opsworld/smoke-c5-mixed/global_step_1/`;
the checkpoint tracker contains `1`. The pod recorded no cgroup OOM, and both
GPUs returned to 0 MiB. The full 20-action workload remains GPU-memory
limited on this pod and needs its own memory strategy before a longer run.
The final pod log is also saved locally as `runs/grpo_smoke_c5_mixed.log`.

### Fast replay on a fresh 2xA40 pod

This is the **verified one-step base-Fara smoke**, not a recipe for unrestricted
20-turn training. Keep the local `grpo/` working tree (currently untracked) and
`runs/grpo_smoke_c5_mixed.log` before retiring the pod; the checkpoint itself
is only on the pod at the path above. Transfer it separately if an exact
resume is needed. A new run from the base model is a new update, not a resume.

1. Prepare the same 2xA40/100 GB class pod with Python 3.12 in
  `/root/verl/.venv`, verl **0.9.1** at
  `1876b06d0a3e4e71e06230be10af14492ca8a75b`, vLLM **0.24.0**,
  PyTorch **2.11.0+cu130**, Transformers **5.9.0**, PEFT **0.19.1**,
  Playwright **1.51.0** and Firefox. The pinned verl lockfile's flash-attn
  wheel 404ed; `uv sync --python 3.12 --extra fsdp --extra vllm --no-dev
  --no-install-package flash-attn` installed the rest. Install the local Fara
  clone at `a675d6d61c41c47ae87bacefeab22caad18e3e84` into this venv,
  including its local `FARA_POST_ACTION_SLEEP` edit. Provision the base model
  at `/root/models/Fara-7B`. The historical bootstrap above has installation
  details; do not reuse its old SFT/merge package versions for this verl venv.
2. Transfer the **current** `opsworld/` source tree (including the untracked
  `grpo/` patches) to `/root/cua/opsworld`, omitting `runs/`, DBs, logs and
  credentials. Transfer `runs/grpo_smoke_c5/` separately or regenerate it
  from `opsworld/` with `python -m grpo.make_dataset --smoke-seed 1
  --smoke-category C5_below --out runs/grpo_smoke_c5`; install `datasets`
  first if regenerating. The input files must be at
  `/root/cua/opsworld/runs/grpo_smoke_c5/{train,val}.parquet` on the pod.
3. From `/root/cua/opsworld`, apply **both** patches and run the two-rank LoRA
  regression and agent-loop test using the commands in "Pinned verl fixes"
  above. The launcher fails closed if the LoRA patch is absent. Run the local
  browser test `python -m unittest grpo.test_episode -q` before using GPUs.
4. From `/root/cua/opsworld`, replay the *measured mixed-reward* configuration:

```bash
RAY_memory_usage_threshold=0.85 VLLM_USE_FLASHINFER_SAMPLER=0 \
OPSWORLD_GRPO_MAX_ROUNDS=6 \
MODEL_PATH=/root/models/Fara-7B TOTAL_STEPS=1 EXPERIMENT_NAME=smoke-c5-mixed \
bash grpo/run_grpo.sh \
  '+actor_rollout_ref.model.override_config.attn_implementation=sdpa' \
  'actor_rollout_ref.rollout.enforce_eager=true' \
  'actor_rollout_ref.rollout.max_num_seqs=16' \
  'actor_rollout_ref.rollout.gpu_memory_utilization=0.35' \
  'actor_rollout_ref.rollout.agent.num_workers=1' 'reward.num_workers=1' \
  'data.train_batch_size=1' 'actor_rollout_ref.actor.ppo_mini_batch_size=1' \
  'data.train_files=/root/cua/opsworld/runs/grpo_smoke_c5/train.parquet' \
  'data.val_files=/root/cua/opsworld/runs/grpo_smoke_c5/val.parquet' \
  'trainer.save_freq=1'
```

The chain that made this possible: TP2 vLLM needed
`disable_custom_all_reduce=true` (already in the launcher); BF16 FSDP and
eager vLLM with bounded sequences reduced A40 pressure; `safetensors` and
layered adapter sync avoided shipping the base model; the pinned LoRA patch
fixed the otherwise empty nested FSDP collection and its fatal full-model
fallback under the 100 GB RAM cap; the SDPA patch replaced absent
FlashAttention; numeric model-version tags satisfied verl's trainer. Finally,
six-turn episodes avoided the later actor log-softmax VRAM OOM, and a one-task
C5 group yielded three verifier passes and one failure, nonzero gradient and
a saved checkpoint. Four identical rewards can still yield zero advantage;
inspect `critic/score/{min,max}` and `actor/grad_norm`, not just exit code.
Do **not** raise Ray's 0.85 RAM guard or run the 15-turn, four-task
default launch as if it had been validated. No multi-step GRPO training or held-out
GRPO improvement has yet been measured; longer episodes need a separate
memory strategy before scaling beyond the smoke.

### Fresh 2xA40 pod profile (2026-10-18)

The new A40 host has driver 570.195.03 (CUDA 12.8), while pinned verl installs
PyTorch 2.11.0+cu130. Installing `cuda-compat-13-0` in the container and setting
`LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH` enabled GPU
allocations. Two-rank NCCL stalled until both `NCCL_P2P_DISABLE=1` and
`NCCL_SHM_DISABLE=1` were set; use these on this host for GRPO.

The one-step, four-rollout C5 seed-1 run with six actions per rollout completed
in 223.35 seconds and saved a checkpoint, but all rewards were 1: advantage and
gradient norm were zero. A separate one-step C1 seed-1 run with a 15-action cap
reached actor update, then OOMed in `logprobs_from_logits_v2` at `F.log_softmax`:
GPU 0 had only 512 KiB free with 43.86 GiB in use. It finished 0/1 steps and
saved no checkpoint. One-second GPU sampling missed the transient OOM peak.
Do not launch 15-action base-to-GRPO training on this 2xA40 configuration;
reduce actor log-probability memory use or move to more VRAM first.

## GRPO launch (first a single training step)

From `opsworld/` on the Mac, generate and transfer `runs/grpo_data/` to the
**same relative location** on the future pod:

```bash
../fara/.venv/bin/python -m grpo.make_dataset
scp -P <PORT> -i ~/.ssh/id_ed25519 -r runs/grpo_data root@<HOST>:/root/cua/opsworld/runs/
```

Once verl/Fara/Firefox are installed and the base model is available locally,
use the **Fast replay** command above first. The generic launcher below uses
four tasks per batch and up to 15 turns; it is **not** validated on 2xA40.
Both 15-action C1 and 20-action configurations OOMed there. Do not scale to this setting
on that pod without a separate memory check.
For a pod where that workload has been proven to fit, run in `opsworld/`:

```bash
MODEL_PATH=/root/models/Fara-7B TOTAL_STEPS=1 EXPERIMENT_NAME=base-smoke bash grpo/run_grpo.sh
# After the smoke passes, increase TOTAL_STEPS for the base -> GRPO arm.
```

For the SFT arm, **merge step 236 with base Fara first**, then train a fresh
GRPO LoRA on the merged SFT model. The KL reference in verl's LoRA path is
the model with the trainable adapter disabled; merely passing the SFT adapter
as `lora_adapter_path` would silently regularize toward *base Fara*. Use
`sft/merge_lora_hf.py` to create `/root/models/Fara-7B-sft236-merged`, then:

```bash
MODEL_PATH=/root/models/Fara-7B-sft236-merged TOTAL_STEPS=1 \
  EXPERIMENT_NAME=sft-smoke bash grpo/run_grpo.sh
```

Set a *different* experiment name and checkpoint directory for each arm. The
launch settings are an initial smoke configuration,
not a confirmed 2xH100 memory budget. Do not launch the full run until
tokenization, reference/policy logprobs, 0/1 rewards, and one optimizer update
are checked end-to-end. Preserve all images in the rollout; replacing them
with text-only tool calls invalidates the experiment.

For base -> GRPO, run one saved/reloadable update, then a 10-20 step pilot.
Inspect per-category verifier successes and mixed-reward groups before a
50-100 update run (200-400 distinct training tasks, 800-1600 episodes).
One pass through the 1794 train tasks would take about 449 updates/7176
episodes; do not assume an epoch is affordable until the first update gives
real throughput and GPU memory measurements. Validation uses 114 tasks on
seeds 300-318; the fixed 120 held-out tasks (seeds 9000-9119) are for
benchmarking saved models, not for tuning rewards or training. Baseline Fara
scored 66/120 before GRPO.