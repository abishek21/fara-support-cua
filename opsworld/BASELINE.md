# Baseline Eval Runbook (base Fara-7B on OpsWorld)

Everything is built; this is the exact checklist once the RunPod GPU is up.

## 1. On the pod — serve Fara-7B
```bash
# (first time only) create the CUDA-12 vLLM venv, see commit history
/root/vllm-env/bin/vllm serve microsoft/Fara-7B \
  --port 5000 --dtype auto --max-model-len 20000 --gpu-memory-utilization 0.92
# wait for: "Application startup complete."
```

## 2. On the Mac — tunnel (use :5050, NOT :5000 — macOS AirPlay owns 5000)
```bash
ssh -N -f -o ServerAliveInterval=30 -L 5050:localhost:5000 \
  -p <POD_SSH_PORT> -i ~/.ssh/id_ed25519 root@<POD_IP>
curl -s localhost:5050/v1/models        # expect 200 + microsoft/Fara-7B
```
`runpod_config.json` already points `base_url` -> `http://localhost:5050/v1`.

## 3. Start the single-env sandbox is NOT needed
The benchmark runner spawns its OWN isolated CRM+Billing envs (ports 8300+).

## 4. Smoke test (6 tasks, ~2-3 min)
```bash
cd opsworld && source ../fara/.venv/bin/activate && export FARA_POST_ACTION_SLEEP=0.6
python -m solver.bench_runner --endpoint ../runpod_config.json \
  --workers 1 --runs 1 --max-rounds 20 --limit 6 --out runs/bench_smoke --label smoke
```

## 5. Full baseline (120 tasks, ~30-45 min with 3 workers)
```bash
python -m solver.bench_runner --endpoint ../runpod_config.json \
  --workers 3 --runs 1 --max-rounds 20 --out runs/bench_base --label base-Fara-7B
```
Optional publication-grade (3 runs, ~1.5 h): add `--runs 3`.

## 6. Read the scorecard
The runner prints per-category pass rate + a **readiness flag**:
- `<10%`  -> **SFT needed (not GRPO-ready)**
- `10-40%` -> weak signal (SFT recommended)
- `>=40%` -> GRPO-viable
and an overall VERDICT on whether SFT is required before GRPO.

## Notes
- Keep `--workers` at 2-3 (multiple Fara/Firefox + concurrent vLLM on one A40).
- Lower `--max-rounds` bounds the slow failing rollouts.
- Results saved to `runs/bench_base/results.json` (+ per-task console logs/screens).
- Same command with a different `--endpoint`/`--label` later evaluates the SFT'd model.
