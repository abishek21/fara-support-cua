# OpsWorld — Self-Evolving CUA Project: Context & Handoff

## 1. What this project is
Building a **verifiable, multi-system enterprise sandbox** to adapt **Fara-7B** (Microsoft's
7B computer-use agent, = Qwen2.5-VL-7B SFT'd to emit `thought + <tool_call>{action, coordinate}`)
to support-ops workflows via **SFT + GRPO**, using **GPT-5.4 as a synthetic expert solver**.

Goal: a research result / outreach to **Akshay Nambi (MSR India)** who is hiring for
"self-evolving agentic systems" (RL, post-training, harness evolution, reward modeling,
verifiable environments, synthetic data, evals).

## 2. Repo
- GitHub: **github.com/abishek21/fara-support-cua** (private), branch `main`
- Local: `/Users/abishek/cua`
- Key dirs:
  - `fara/` — microsoft/fara clone (gitignored; separate repo). ONE local edit: `src/fara/fara_7b/fara_agent.py` line ~580 made post-action sleep configurable via `FARA_POST_ACTION_SLEEP` env var.
  - `support_env/` — older SINGLE-system sandbox (reference, still works)
  - `opsworld/` — the CURRENT multi-system sandbox (the main project)
  - `runs/` — gitignored (trajectories, DBs, logs, screenshots)

## 3. Architecture (opsworld/)
Two independent web apps = two systems the agent must switch between:
- **Contoso CRM** (`:8001`, blue UI) — accounts, status (active/churned/suspended), case notes.
  Actions: `add_note`, `update_status`.
- **Contoso Billing** (`:8002`, teal UI) — invoices, subscription, refunds.
  Actions: `refund`, `cancel_billing`, `block_subscription`. Also a filterable **Reports** view (`/reports` + `/api/report`).
- `common/world.py` — ONE seeded world generator: 80+ accounts, shared `account_id`/email across both systems, **randomized scenario roles per `DB_SEED`** (NO name→role bias — verified). Ground truth derives from the seed.
- `common/metrics.py` — deterministic ground-truth aggregates for report tasks.
- `verify.py` — **strict multi-DB verifier** with change-accounting: requires exact required actions AND catches over-reach/wrong-action/extra/partial across BOTH DBs. 22/22 test cases pass (`test_verify.py`). Verify types: `refund_any_of`, `cancel_billing`, `block_subscription`, `note_added`, `update_status`, `no_change_all`, `all_of`, `block_all`, `report_answer` (exact value).

## 4. Key design principles (locked)
- **Verifiable only** (DB-diff or exact-value answer; NO LLM-judge).
- **Randomized person per seed** → no memorization; instructions use NAMES ONLY (no IDs).
- **Disjoint seeds**: train = seeds 1–318; **eval = held-out seeds 9000–9119**.
- **Search-first UI**: customer list hidden until you type (prevents info leak).
- **Student-replicable actions only**: converter rejects `select` (native dropdowns) — Fara can't emit them.

## 5. The solver (GPT-5.4 + SoM adapter) — WORKS
- GPT-5.4 can't emit pixel coords, so **SoM adapter**: inject numbered marks on real DOM elements → GPT picks element NUMBER → we execute on real DOM node → log clean screenshot + thought + action + **bbox-center coordinate**.
- `solver/gpt_solver.py` + `solver/som_browser.py` + `solver/run_solver.py`.
- Azure config: gitignored `support_env/azure_openai_config.json`, deployment `gpt-5.4`, endpoint `https://akfoundry21.cognitiveservices.azure.com/`, api_version `2024-12-01-preview`.
- `solver/convert_trajectory.py` — converts solver trajectories → Fara-format SFT samples (maps SoM action → `left_click(x,y)`/`type`/`visit_url`/`terminate`; clean screenshot; ChatML + loss mask). REJECTS trajectories with `select` (not student-replicable).

## 6. Data collection — DONE
- `solver/collect.py` — **parallel** collector: K isolated worker env pairs (own ports `8300+2w`/`8301+2w`, own DBs), reseed per task via `/api/reset?seed=N`, shared `common/rate_limiter.py` (RPM + max-concurrent), headless, verify → keep-if-passed → convert to SFT.
- **PILOT COLLECTED: 308 verified trajectories (97% yield), ~1,900 SFT samples**, balanced 50/category C1–C6.
- `solver/merge_split.py` → split BY TRAJECTORY (no step leakage), stratified: 291 train / 15 val trajectories = 1,900 train / 104 val samples.
- `solver/hf_upload.py` → **uploaded to HF (private): `abishekk210698/opsworld-sft`** (images embedded). Note HF username is `abishekk210698`. Token in gitignored `opsworld/hf_config.json`, read via `common/hf_auth.py`.

## 7. Task taxonomy (C1–C6, phase 1)
- **C1_refund** (act, hard): duplicate charge → refund one + CRM note
- **C2_cancel** (act): customer requests → cancel billing
- **C3_block** (act, easy): 3+ failed payments → block subscription
- **C4_churn** (act, HARDEST, cross-system): CRM says churned but Billing still on → cancel billing + CRM note
- **C5_below** (DON'T act): 2 failed payments (below 3-policy) → do nothing
- **C6_noerror** (DON'T act): legit single charge → do nothing
- Eval set: `opsworld/tasks/eval_tasks.jsonl` = **120 tasks** (20×6), seeds 9000–9119.
- Deferred: C7 analytics (needs clickable-filter UI), and the **dropdown/new-action tier** (see §10).

## 8. ⭐ BASELINE RESULT (measured, committed)
**base Fara-7B = 55% (66/120)** on held-out eval:
| Category | Rate | Readiness |
|---|---|---|
| C5_below | 80% | GRPO-viable |
| C3_block | 75% | GRPO-viable |
| C6_noerror | 75% | GRPO-viable |
| C2_cancel | 40% | GRPO-viable |
| **C1_refund** | **35%** | weak → SFT |
| **C4_churn** | **25%** | weak → SFT (hardest) |

Clean difficulty gradient: easy refrain/single-action 75–80%, hard multi-step/cross-system 25–40%.
Nothing at 0% → GRPO-only viable; C1/C4 weak → SFT clearly helps. Results: `runs/bench_base/results.json`, `results/BASELINE_RESULTS.md`.

## 9. Characterized failure modes (from reading failed C1/C4 logs)
Three compounding Fara-7B weaknesses, each maps to a fix:
1. **URL/API wandering** (tries REST endpoints / real-web URLs instead of the UI) → **SFT** (demos stay in-console)
2. **Judgment error** (reads two same-period same-amount charges but concludes "no duplicate") → **SFT** (refund demos)
3. **Looping without recovery** (clicks same coord 5×, hits max_rounds) → **GRPO** (outcome reward teaches recovery; no orchestrator/loop-breaker in distilled Fara-7B)

## 10. Research direction
Shift from weak "GRPO rescues broken model" to stronger **"self-evolving verifier-driven loop"**:
- Base Fara 55% with real headroom on hard cross-system tasks (C4 25%).
- **3-way ablation**: base / GRPO-only / SFT→GRPO (base non-zero everywhere → GRPO-only is a legit arm).
- **Generalization**: train seeds 1–318, eval held-out 9000+ with different name→role mappings → learned skill not instances.
- **Bonus 2nd contribution — action-space extension**: add native `<select>` dropdown tasks (C7). Base Fara ≈ 0% (options render in OS layer, NOT in screenshot/DOM → no pixel to click). Key RL insight: **GRPO alone CANNOT learn a zero-probability action → need SFT to seed the new action/keyboard-recipe, then GRPO refines.** "What does it take to extend a fixed action space?"
- Outreach: email akshayn@microsoft.com (he's at COLM), lead with the WORK + a finding + repo, ask for feedback/collaboration (not the job — job needs PhD). 

## 11. Infra / GPU notes
- RunPod pod (A40 46GB). **macOS owns port 5000 (AirPlay)** → tunnel on **:5050**. `runpod_config.json` base_url = `http://localhost:5050/v1`.
- Fresh pod gotcha: bare image, no python → use `/opt/conda`; make venv `/root/vllm-env`; `pip install vllm`.
- **CUDA 13 pod**: vLLM 0.31 defaults to flashinfer JIT sampler needing `ninja` (missing) → **serve with `VLLM_USE_FLASHINFER_SAMPLER=0`**. (Old CUDA-12 pod used vllm 0.19.1 which didn't hit this.)
- Serve: `VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve microsoft/Fara-7B --port 5000 --dtype auto --max-model-len 20000 --gpu-memory-utilization 0.92`
- SSH backgrounding over one-shot ssh gives exit 255 (cosmetic) — run vLLM foreground in a dedicated pod terminal.

## 12. Benchmark runner
`solver/bench_runner.py` — runs Fara (via tunnel) on eval_tasks.jsonl in parallel isolated envs, reset-per-seed, multi-DB verify, per-category scorecard + GRPO-readiness flags. Flags: `--workers`, `--runs`, `--max-rounds`, `--screens {none,fail,all}` (default `fail` = keep screenshots only for failures, delete on pass), `--limit`, `--label`, `--endpoint`.
Run: `export FARA_POST_ACTION_SLEEP=0.6; python -m solver.bench_runner --endpoint ../runpod_config.json --workers 3 --runs 1 --max-rounds 20 --screens fail --out runs/bench_base --label base-Fara-7B`

## 13. NEXT STEPS (in order)
1. **SFT first** (fastest, 1×A40 enough, ~2–4h): build ms-swift adapter (HF `abishekk210698/opsworld-sft` → ms-swift multimodal format) + LoRA config (rank 16, freeze ViT, bf16, 2 epochs) + runbook. Then merge/serve adapter, re-run the 120-eval → before/after on C1/C4.
2. **verl GRPO harness** (big lift, needs 2×H100-80GB): wrap OpsWorld browser + DB verifier as a verl MULTI-TURN MULTIMODAL environment (new screenshot obs each step, DB-verify reward at episode end). This is the gating engineering work. Then run GRPO-only + SFT→GRPO.
3. **Dropdown C7 tier** for the action-space-extension experiment.
4. Confirm failure-mode pattern on 1–2 more C1/C4 logs. Write 2-pager + figure. Email Akshay.

## 14. Status summary
✅ Multi-system verifiable sandbox  ✅ strict verifier (22/22)  ✅ GPT-5.4+SoM solver
✅ 308 trajectories / 1,900 SFT samples on HF  ✅ 120 held-out eval set
✅ BASELINE: base Fara-7B = 55% (C4 25%, C1 35%)  ⬜ SFT  ⬜ GRPO  ⬜ dropdown tier