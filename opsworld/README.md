# OpsWorld — multi-system support sandbox

A **behavioral** (not pixel-perfect) simulation of a support engineer's reality:
two separate internal systems with their own UIs and databases that the agent
must **navigate between and reconcile** — mirroring Dynamics ↔ Salesforce,
CRM ↔ Billing, etc.

| System | Port | Holds | Actions |
|---|---|---|---|
| **Contoso CRM** | 8001 | account, status (active/churned/suspended), plan, **case notes** | `add_note`, `update_status` |
| **Contoso Billing** | 8002 | invoices, subscription, refunds, failed payments | `refund`, `cancel_billing`, `block_subscription` |

## Why this is harder (and more real)
- **Cross-system reconciliation**: the correct action often requires reading one
  system and acting in another (e.g. CRM says *churned* but Billing still
  *charging* → stop billing **and** note it in CRM).
- **Tab/app switching + state carry**: same `account_id`/email across systems.
- **CUA-necessary**: no single API spans both systems with consistent state —
  a human (or CUA) bridging the UIs is the realistic path.

## Fairness & determinism (critical)
- One **seed-controlled world** (`common/world.py`) is the single source of truth;
  each system seeds its own DB from it, so systems are consistent **by
  construction** — except where a scenario deliberately makes them disagree.
- **Random but deterministic**: names and scenario ROLES are randomly assigned
  from `DB_SEED`, so there is **no name→role bias**. Change `DB_SEED` → a new world.
- Use **disjoint seed ranges** for train vs. eval (generalization, not memorization).

## Scenarios (all auto-verifiable via DB-diff)
| Kind | Systems | Correct outcome |
|---|---|---|
| `reconcile_churn` | CRM+Billing | CRM churned but still billed → **cancel billing + add CRM note** |
| `reconcile_refund` | CRM+Billing | duplicate charge, active owner → **refund + add CRM note** |
| `block_policy` | Billing | ≥3 failed payments → **block** |
| `below_policy` | Billing | 2 failed → **no change** |
| `no_error_no_refund` | CRM+Billing | "forgot to cancel" → **no change** |

## Run
```bash
cd opsworld
DB_SEED=20261005 ./run_all.sh          # launches CRM :8001 + Billing :8002
python inspect_world.py                # print world + ground truth + bias check
open http://localhost:8001             # CRM
open http://localhost:8002             # Billing
```

## Verify a rollout (multi-DB)
```python
from verify import verify_task
# before/after = {"crm": <crm snapshot>, "billing": <billing snapshot>}
verify_task(scenario.verify, before, after)
```

## Files
| Path | Purpose |
|---|---|
| `common/world.py` | seed-controlled world generator (fair ground truth) |
| `common/dbutil.py` | sqlite helpers |
| `crm/app.py`, `crm/static/crm.html` | CRM system (blue) |
| `billing/app.py`, `billing/static/billing.html` | Billing system (teal) |
| `verify.py` | multi-system DB-diff verifier |
| `inspect_world.py` | debug: print world + consistency + bias checks |
| `run_all.sh` | launch both systems |

## Status
- ✅ Two-system world, shared identity, randomized fair ground truth
- ✅ Cross-system reconciliation scenarios + multi-DB verifier
- ⏭️ Next: task generators (phrasing) + multi-system benchmark runner + solver
- 🔜 Later: Admin system (password/MFA reset), Ticketing
