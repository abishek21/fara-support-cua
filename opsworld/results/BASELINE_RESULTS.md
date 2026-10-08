# Baseline: base Fara-7B on OpsWorld held-out eval (120 tasks, seeds 9000-9119)

Run: 1x, 3 workers, max_rounds 20, Fara-7B via vLLM (RunPod A40). ~80 min.

| Category | Pass | Rate | Type |
|---|---|---|---|
| C1_refund  | 7/20  | 35% | act (hard) |
| C2_cancel  | 8/20  | 40% | act |
| C3_block   | 15/20 | 75% | act (easy) |
| C4_churn   | 5/20  | 25% | act (cross-system, hardest) |
| C5_below   | 16/20 | 80% | don't-act |
| C6_noerror | 15/20 | 75% | don't-act |
| **OVERALL** | **66/120** | **55%** | |

Headroom: C4 (cross-system reconciliation) and C1 (refund) are the weak,
long-horizon multi-step skills -> primary SFT/GRPO targets.
All categories non-zero -> GRPO-only arm is viable.
