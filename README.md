# Fara CUA — Support-Ops Sandbox & Benchmark

Research scaffolding for adapting a small **Computer Use Agent** (Microsoft
**Fara-7B**, a Qwen2.5-VL model fine-tuned for browser actions) to **enterprise
support workflows** via a self-built, **verifiable** sandbox — with an eye toward
SFT (expert-demonstration distillation) and GRPO.

## What's here

| Path | Purpose |
|---|---|
| `support_env/` | The **Support Console FaraEnv** — a sandboxed SaaS billing console |
| `support_env/db.py` | SQLite schema + **procedural, seed-controlled** data (110 customers; randomized scenario roles to prevent name/stereotype leakage) |
| `support_env/app.py` | FastAPI backend + console UI; `/api/reset`, `/api/snapshot`, search |
| `support_env/static/console.html` | Clickable console (search-first; details on open) |
| `support_env/verify.py` | **Execution-based verifier** — diffs DB before/after (the reward signal) |
| `support_env/benchmark.py` | Runs Fara against tasks; `--runs N`, process/outcome scoring |
| `support_env/tasks.jsonl` | Curated L1–L6 difficulty-ladder tasks |
| `support_env/generate_hard_tasks.py` | Derives L3–L6 tasks with DB-grounded truth (GPT-5.4 phrasing optional) |
| `support_env/propose_tasks.py` | GPT-5.4 task proposer (L1–L2) |
| `support_env/azure_llm.py` | Azure GPT-5.4 helper (proposer / user-simulator / judge) |
| `view_trajectory.py` | Renders a run's screenshots + thoughts/actions as an HTML timeline |
| `run_task.sh`, `tunnel.sh` | Convenience: run a task; SSH tunnel to a GPU-hosted Fara endpoint |

> The upstream model harness lives in a separate clone (`fara/`, gitignored).

## The pipeline

```
Environment (verifiable sandbox) → Tasks → Solver → Trajectories → Verify → SFT → GRPO
        ✅ built                     ✅        next        next       ✅      ...    ...
```

## Key design decisions
- **Execution-based verification** (DB diff), not an LLM judge → reliable reward.
- **Randomized scenario roles per seed** → no "name = role" leakage; train/eval use
  disjoint seed ranges to test generalization, not memorization.
- **L1–L6 difficulty ladder** — from single named actions up to disambiguation,
  deferral (must-ask), compositional, and batch-sweep tasks.

## Status (checkpoint)
- ✅ Verifiable 110-customer sandbox with search + randomized roles
- ✅ L1–L6 benchmark + verifier (`refund_any_of`, `cancel`, `block`, `no_change`,
  `all_of`, `block_all`, `must_ask`)
- ✅ Base Fara-7B baselines with characterized failure modes
  (name-stereotype bias, cancel-vs-block confusion, long-horizon wandering)
- ✅ GPT-5.4 task generators (ground-truth-safe)
- ⏭️ Next: GPT-5.4 **SoM solver + collector** to produce verified expert
  trajectories (runs on Azure + local — no GPU needed), then LoRA SFT.

## Secrets
`azure_openai_config.json` and `runpod_config.json` are **gitignored**. Copy the
templates and fill locally.
