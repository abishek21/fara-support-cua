# OpsWorld Task Taxonomy & Data Plan (blueprint)

This is the blueprint for task design, the held-out eval set, and the on-the-fly
expert-trajectory collection. Numbers are proposals to lock before building the
generators.

---

## 1. Design principles

- **Verifiable**: every task maps to a deterministic ground-truth `verify` spec
  (multi-DB diff or exact-value answer). No LLM-judged/soft tasks.
- **Fair & unbiased**: the *person* for each scenario is randomized per `DB_SEED`
  (no name→role leakage). Instructions use **names only, never IDs**.
- **Balanced act / don't-act**: a model must not win by always acting or always
  deferring. ~⅓ of tasks should be "correct = do nothing".
- **Generalization, not memorization**: **train and eval use DISJOINT seed ranges.**
- **Student-replicable actions only**: solver uses `click / type / visit_url /
  terminate` (converter rejects anything Fara can't emit, e.g. `select`).

---

## 2. Categories (skills)

| # | Category | System span | Correct behavior | Core action(s) | Verify type |
|---|---|---|---|---|---|
| C1 | **Refund duplicate** | Billing (+CRM note opt.) | act | refund | `refund_any_of` |
| C2 | **Cancel billing** | Billing | act | cancel_billing | `cancel_billing` |
| C3 | **Block (policy ≥3 failed)** | Billing | act | block | `block_subscription` |
| C4 | **Churn reconciliation** | **CRM + Billing** | act (2 actions) | cancel_billing + note | `all_of` |
| C5 | **Don't-act: below policy** (2 failed) | Billing | **do nothing** | — | `no_change_all` |
| C6 | **Don't-act: no billing error** | Billing | **do nothing** | — | `no_change_all` |
| C7 | **Analytics → report** | Billing (Reports) | report a value | read/aggregate | `answer == metric` |
| C8 | **Analytics → batch action** | Billing (Reports) | act on a set | block each qualifying | `block_all` |

**Phase-2 categories (after Admin system + clickable filters):**
| C9 | **Security reconciliation** | CRM+Billing+Admin | act (3 systems) | reset pw/MFA + pause + note | `all_of` |
| C10 | **Disambiguation** | any | act (find right one) | depends | depends |
| C11 | **Defer / must-ask** (under-specified) | any | **ask, do nothing** | — | `no_change_all` + must_ask |
| C12 | **Compositional** (refund + cancel) | Billing(+CRM) | act (≥2) | multiple | `all_of` |

> ⚠️ **Prereq for C7/C8 training data:** the Reports filters are native `<select>`
> dropdowns. Convert them to **clickable controls** (segmented buttons / custom
> dropdown of divs) so every action is `left_click`/`type` and thus Fara-replicable.

---

## 3. Difficulty ladder (orthogonal to category)

| Level | Definition | Example categories |
|---|---|---|
| L1 | single action, named target | C2, C3 |
| L2 | judgment / policy (act vs not) | C5, C6 |
| L3 | disambiguation (search + pick right one) | C10 |
| L4 | under-specified → defer/ask | C11 |
| L5 | compositional / cross-system (≥2 actions) | C4, C12 |
| L6 | batch / analytical sweep / 3-system | C8, C9 |

---

## 4. Eval set (held-out) — PROPOSAL

Fixed, materialized once on **held-out seeds** so every model is tested on the
identical set. Run **×3**, report average + pass@3.

| Category | Eval tasks |
|---|---|
| C1 Refund duplicate | 25 |
| C2 Cancel billing | 25 |
| C3 Block (policy) | 25 |
| C4 Churn reconciliation | 25 |
| C5+C6 Don't-act | 25 |
| C7/C8 Analytics (report + batch) | 25 |
| **Total (phase-1)** | **~150** |

- Seeds: `DB_SEED ∈ EVAL_SEEDS` (e.g. **9000–9199**), never used for training.
- Materialize to `opsworld/tasks/eval_tasks.jsonl` (instruction + verify + seed).
- 150 × 3 runs = **450 rollouts** per model — enough signal, cheap.
- Expand (+25/category) only if a category shows high variance.

---

## 5. Train trajectories (on-the-fly) — PROPOSAL

Streaming loop; **no pre-written task file** required (optionally materialize for repro).

```
for i in range(N_ATTEMPTS):
    seed = TRAIN_SEED_BASE + i                 # disjoint from EVAL_SEEDS
    reseed CRM + Billing with DB_SEED=seed     # fresh world (new names/roles)
    world = generate_world()                   # ground truth known for this seed
    scenario = pick_round_robin(categories)    # balance across C1..C8
    task = phrase(scenario)                    # name only, no IDs
    reset DBs; before = snapshot
    traj = gpt54_solver.solve(task)            # expert acts via SoM
    after = snapshot
    if verify(scenario.verify, before, after).success:
        samples = convert(traj)                # thought + action + coord + clean img
        append to dataset  (+ push to HF)
    else:
        drop (never train on an incorrect demo)
```

**Targets:**
| Item | Number |
|---|---|
| Verified trajectories to KEEP | **~1,500** |
| Expected verifier yield | ~40–55% |
| Attempts to run | **~3,000–4,500** |
| ≈ SFT step-samples (×~8 steps) | **~12,000** |
| Train seeds | `1 – ~4500` (disjoint from eval) |
| Per-category quota | ~equal (±, cap any one at ~25%) |

**Pilot first:** collect **~300 verified** (~$30–110) to confirm SFT moves the
needle before committing to 1,500.

---

## 6. Storage / dataset

- **Images are required** for SFT (Fara is a VLM). Stored once per *step*
  (samples point at paths). ~1.5K traj × ~8 steps ≈ 2–4 GB.
- **Private HF dataset** under `abishek21/`:
  - `opsworld-trajectories` (raw): `image, thought, action_json, coord, scenario,
    seed, verified, split`
  - `opsworld-sft` (derived): `images, messages, label` — directly SFT-loadable
- Images **embedded** in parquet → one `load_dataset` on the GPU pod.

---

## 7. LOCKED numbers (2026-10-05)

Scope: **C1–C6 only** for phase-1. Analytics (C7/C8) deferred to eval-only until
the Reports filters are converted to clickable controls.

| Decision | LOCKED |
|---|---|
| Categories (phase-1) | **C1–C6** |
| **Eval tasks total** | **120** (20 × 6), ×3 runs = 360 rollouts/model |
| Eval seed range | **9000–9199** (held-out, never trained on) |
| **Train pilot (verified)** | **300** = **50 per category** |
| Attempts to run (≈ yield) | **~450–550** (higher for C4 churn, lower for C5/C6) |
| Train seed range | **1+**, disjoint from eval; **one seed per trajectory** |
| act / don't-act balance | 200 act (C1–C4) + 100 don't-act (C5–C6) = ~33% |
| SFT step-samples (≈ ×8) | ~2,400 |
| SoM image in solver | **on** (reliable); A/B text-only later for cost |
| HF dataset (private) | `abishek21/opsworld-*` |

Plan: pilot 300 → LoRA SFT → eval on 120 → GRPO. If GRPO reaches target with few
demos, great; else collect more (scale to ~1,500).

---

## 8. Build order (after locking)
1. `phrase(scenario)` generator (+ optional GPT-5.4 phrasing variety).
2. **Materialize eval set** (150 tasks, held-out seeds) → `eval_tasks.jsonl`.
3. **Collector** (on-the-fly train loop) → verified trajectories → SFT samples.
4. **HF uploader**.
5. Baseline eval (base Fara) → LoRA SFT → eval → GRPO.
