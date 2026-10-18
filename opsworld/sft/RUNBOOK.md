# OpsWorld SFT Runbook — LoRA adapt Fara-7B on a single A40

End-to-end recipe to SFT **microsoft/Fara-7B** on the OpsWorld solver trajectories
(`abishekk210698/opsworld-sft`, 1,900 train / 104 val samples), merge the adapter,
serve it, and re-run the 120-task held-out eval for a **before/after on C1/C4**.

All files live in `opsworld/sft/`:

| file | purpose |
|---|---|
| `prepare_swift_data.py` | HF dataset → Fara-native JSONL (`--format fara`) and action overlays |
| `train_lora_hf.py` | transformers + PEFT LoRA (rank 16, ViT frozen, bf16) |
| `merge_lora_hf.py` | Merge PEFT adapter into base weights |
| `serve_sft.sh` | vLLM serve merged model (served-model-name = microsoft/Fara-7B) |
| `compare_results.py` | before/after scorecard diff (focus C1_refund, C4_churn) |

---

## 0. Hardware / assumptions
- 1× **A40 (46 GB)** RunPod pod (same class as baseline serving).
- CUDA-13 pod caveats from `project_context.md §11` apply (flashinfer sampler off when serving).
- You can train on the **same pod** you later serve from, or a separate one.

## 1. Pod setup
```bash
# fresh pod: python via conda, dedicated venv
/opt/conda/bin/python -m venv /root/sft-env
source /root/sft-env/bin/activate
pip install 'transformers==4.51.3' 'peft==0.15.2' datasets accelerate pillow tensorboard
```

Auth for the **private** HF dataset (pick one):
```bash
export HF_TOKEN=hf_xxx          # or: scp opsworld/hf_config.json up and set HF_TOKEN from it
# prepare_swift_data.py also falls back to common/hf_auth.py if run from the repo
```

## 2. Build and inspect Fara-native data
```bash
cd opsworld
../fara/.venv/bin/python sft/prepare_swift_data.py --format fara \
    --repo abishekk210698/opsworld-sft \
  --out runs/sft_native
```
This requires the local `fara` clone and HF access. The output contains 1900 train
and 104 validation rows split by trajectory. It uses Fara's actual tool prompt,
alternating turns, 1428x896 images and model-frame coordinates. Inspect
`runs/sft_native/inspect/` before training: the red target must land inside the
intended control. To move it to the pod:
```bash
scp -P 22162 -i ~/.ssh/id_ed25519 -r runs/sft_native \
  root@69.30.85.20:/root/opsworld-fara-native
```

## 3. Train (LoRA, ViT frozen)
Stop the pod's vLLM server before training; both processes cannot fit on the A40.
First verify training and validation loss with one optimizer step:
```bash
source /root/sft-env/bin/activate
python /root/sft/train_lora_hf.py \
  --model /root/.cache/modelscope/models/microsoft--Fara-7B/snapshots/master \
  --data_dir /root/opsworld-fara-native \
  --output_dir /root/opsworld-sft-native-smoke \
  --max_steps 1 --eval_steps 1 --logging_steps 1
```
Check that both `loss` and `eval_loss` are finite before a longer run. This
one-step check does not establish that the policy improves on held-out tasks.

For the full run (after the smoke-test gate):
```bash
# On the pod, copy sft/train_lora_hf.py to /root/sft/ first.
source /root/sft-env/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python /root/sft/train_lora_hf.py \
  --model /root/.cache/modelscope/models/microsoft--Fara-7B/snapshots/master \
  --data_dir /root/opsworld-fara-native \
  --output_dir /root/opsworld-sft-native-out --epochs 2 --save_steps 10
```
Do not launch a full run before checking a short training run and held-out action
rollouts. `--max_pixels` now matches Fara; lowering it silently reintroduces the
image/coordinate mismatch. The collator rejects overlong examples rather than
truncating image grids or action targets. Training loss is logged every 5 steps,
validation loss every 50, and up to three checkpoints are retained every 10.

TensorBoard logs both training and validation loss under the output directory.
On the pod, start `tensorboard --logdir /root/opsworld-sft-native-out/runs
--host 127.0.0.1 --port 6006`. From the Mac, tunnel with
`ssh -N -L 6006:localhost:6006 -p 22162 -i ~/.ssh/id_ed25519 root@69.30.85.20`,
then open `http://localhost:6006`. Use `/root/opsworld-sft-native-smoke/runs`
as `--logdir` when inspecting the one-step check instead.

## 4. Merge the adapter
```bash
python /root/sft/merge_lora_hf.py \
  --base /root/.cache/modelscope/models/microsoft--Fara-7B/snapshots/master \
  --adapter /root/opsworld-sft-native-out/final --out /root/opsworld-sft-native-merged
```

## 5. Serve the SFT'd model
Run **foreground in a dedicated pod terminal** (SSH backgrounding gives cosmetic exit 255):
```bash
source /root/vllm-env/bin/activate   # the vllm env from baseline, or pip install vllm here
bash /root/sft/serve_sft.sh /root/opsworld-sft-native-merged
```
`serve_sft.sh` sets `--served-model-name microsoft/Fara-7B`, so `runpod_config.json`
and `fara-cli` need **no changes**.

Tunnel pod `:5000` → local `:5050` (macOS owns 5000 for AirPlay):
```bash
# on the mac
ssh -N -L 5050:localhost:5000 -p 22162 -i ~/.ssh/id_ed25519 root@69.30.85.20
```

To publish the preserved LoRA adapters from the Mac (private by default):
```bash
cd opsworld
../fara/.venv/bin/python sft/upload_checkpoints.py --dry-run
../fara/.venv/bin/python sft/upload_checkpoints.py
```
The corrected final checkpoint is **step 236**, not 237. The uploader creates
separate PEFT adapter repositories for steps 100 and 236; neither contains the
merged base model. It updates the uploaded adapter configs to reference
`microsoft/Fara-7B` without modifying the local checkpoints.

## 6. Re-run the 120-task eval (SFT'd model)
```bash
cd opsworld
export FARA_POST_ACTION_SLEEP=0.6
export PATH="../fara/.venv/bin:$PATH"
../fara/.venv/bin/python -m solver.bench_runner \
    --endpoint ../runpod_config.json \
    --tasks tasks/eval_tasks.jsonl \
    --workers 3 --runs 1 --max-rounds 20 --screens fail \
    --out runs/bench_sft_native --label sft-native-Fara-7B
```

## 7. Before/after (focus C1/C4)
```bash
cd opsworld/sft
python compare_results.py \
    --before ../runs/bench_base/results.json \
  --after  ../runs/bench_sft_native/results.json
```
Baseline to beat (from `results/BASELINE_RESULTS.md`): overall **55%**, and the
two weak SFT targets **C1_refund 35%**, **C4_churn 25%**. A successful SFT run
should lift C1/C4 without regressing the refrain categories (C5/C6) — watch those
for over-acting regressions.

---

## Notes / rationale
- **Why freeze the ViT?** The screenshots are clean in-domain console renders; the
  failures are *policy* errors (URL wandering, duplicate-charge judgment, looping),
  not perception. Freezing the vision tower saves memory and avoids perturbing a
  strong pretrained encoder — adapt only the action head.
- **Why LoRA rank 16?** Small, in-domain dataset (~1.9k samples); full FT would
  overfit and won't fit comfortably on 46 GB. Rank 16 is enough to shift behavior.
- **Loss masking** in `train_lora_hf.py` supervises only the final assistant turn;
  earlier assistant turns remain in the prompt as Fara history.
- **No step leakage**: train/val were split BY TRAJECTORY upstream (§6), so val loss
  is a fair proxy; the real signal is the held-out 120-task eval (seeds 9000+).
