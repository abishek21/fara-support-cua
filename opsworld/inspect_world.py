"""
inspect_world.py — print the generated world + scenario roles + cross-system
consistency, for debugging ground truth. Also verifies the two systems agree
where they should and disagree only where a scenario intends it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.world import generate_world, SEED

accs, scns = generate_world()
print(f"World SEED={SEED}: {len(accs)} accounts, {len(scns)} scenarios\n")

print("=== SCENARIOS (ground truth) ===")
for s in scns:
    a = accs[s.account_id]
    dup = sum(1 for i in a.invoices if "(dup" in i.id) if a.invoices else 0
    print(f"[{s.kind:<18}] {a.account_id} {a.name:<20} "
          f"crm={a.crm_status:<9} billing={'on ' if a.billing_on else 'off'} "
          f"failed={a.failed_payments} inv={len(a.invoices)} -> verify={s.verify['type']}")

print("\n=== CROSS-SYSTEM CONSISTENCY CHECK ===")
# churn-mismatch accounts SHOULD be inconsistent (crm churned but billing on);
# everyone else should be consistent (active crm <-> billing on).
bad = 0
scenario_ids = {s.account_id: s.kind for s in scns}
for aid, a in accs.items():
    kind = scenario_ids.get(aid)
    if kind == "reconcile_churn":
        if not (a.crm_status == "churned" and a.billing_on):
            print(f"  ! {aid} churn scenario not set up correctly"); bad += 1
    elif a.crm_status == "churned" and a.billing_on:
        print(f"  ! {aid} unexpected churn/billing mismatch (not a scenario)"); bad += 1
print("  OK — intended (in)consistencies hold" if bad == 0 else f"  {bad} issues")

print("\n=== NAME-BIAS CHECK (first names of scenario accounts) ===")
from collections import Counter
fn = Counter(accs[s.account_id].name.split()[0] for s in scns)
print("  scenario first-name distribution:", dict(fn),
      "\n  (should look arbitrary; change DB_SEED to reshuffle)")
