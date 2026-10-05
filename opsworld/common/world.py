"""
world.py — the single, seed-controlled source of truth for the OpsWorld
multi-system sandbox (Contoso CRM + Contoso Billing).

Design principles
-----------------
- ONE world generator produces all accounts with a SHARED account_id/email.
  Each system (CRM, Billing) seeds its own DB from this same world, so the
  systems are consistent *by construction* — except where a scenario
  deliberately makes them disagree (the reconciliation tasks).
- Deterministic but random: everything derives from DB_SEED. Names are random
  and scenario ROLES are randomly assigned to accounts, so there is no
  "name -> role" leakage/bias. Change DB_SEED -> a brand-new world.
- Ground truth is explicit: each scenario carries a cross-system `verify`
  spec derived from the data, so tasks are always auto-scorable.

Nothing here renders UI or touches HTTP — it's pure data.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Optional

SEED = int(os.environ.get("DB_SEED", "20261005"))
N_ACCOUNTS = 80          # accounts shared across both systems
BLOCK_POLICY = 3         # consecutive failed payments required to block

FIRST = ["Olivia", "Raj", "Mei", "Liam", "Nina", "Carlos", "Aisha", "Tom",
         "Sofia", "Hiro", "Elena", "Noah", "Priya", "Lucas", "Fatima", "Jack",
         "Yuki", "Omar", "Clara", "Ben", "Ava", "Diego", "Sara", "Ivan",
         "Lena", "Max", "Zara", "Leo", "Nora", "Sam", "Amara", "Kenji"]
LAST = ["Martins", "Patel", "Chen", "O'Connor", "Alvarez", "Rossi", "Khan",
        "Nguyen", "Garcia", "Tanaka", "Novak", "Silva", "Haddad", "Meyer",
        "Costa", "Dubois", "Park", "Ferreira", "Walsh", "Ibrahim", "Kowalski",
        "Reyes", "Andersen", "Blum", "Fischer", "Moreau", "Lopez", "Singh"]
PLANS = [("Starter Monthly", 19.0), ("Pro Monthly", 49.0),
         ("Team Monthly", 99.0), ("Business Annual", 990.0)]
DOMAINS = ["contoso", "fabrikam", "northwind", "adventure"]


@dataclass
class Invoice:
    id: str
    period: str
    amount: float
    status: str  # paid | failed | refunded


@dataclass
class Account:
    account_id: str          # shared across systems, e.g. "ACC-1001"
    name: str
    email: str
    plan: str
    price: float
    # --- CRM facts ---
    crm_status: str          # active | churned | suspended
    # --- Billing facts ---
    billing_on: bool
    failed_payments: int
    invoices: list[Invoice] = field(default_factory=list)
    # --- bookkeeping ---
    scenario: Optional[str] = None  # role this account plays (None = distractor)


@dataclass
class Scenario:
    scenario_id: str
    kind: str
    account_id: str
    brief: str               # decision-free situation description for task phrasing
    systems: list[str]       # which systems are involved
    verify: dict             # cross-system ground-truth spec


def _sub_id(acc: str) -> str:
    return "SUB-" + acc.split("-")[1]


def generate_world():
    """Return (accounts: dict[id->Account], scenarios: list[Scenario])."""
    rng = random.Random(SEED)
    used_names: set[str] = set()

    def rand_name() -> str:
        while True:
            n = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
            if n not in used_names:
                used_names.add(n)
                return n

    # Build a plain pool of normal accounts first.
    accounts: dict[str, Account] = {}
    ids = [f"ACC-{1001 + i}" for i in range(N_ACCOUNTS)]
    for aid in ids:
        name = rand_name()
        plan, price = rng.choice(PLANS)
        email = (name.lower().replace(" ", ".").replace("'", "")
                 + f"@{rng.choice(DOMAINS)}.com")
        acc = Account(
            account_id=aid, name=name, email=email, plan=plan, price=price,
            crm_status="active", billing_on=True, failed_payments=0,
            invoices=[Invoice(f"INV-{aid.split('-')[1]}A", "Sep 2026", price, "paid")],
        )
        accounts[aid] = acc

    scenarios: list[Scenario] = []

    # Randomly pick DISTINCT accounts to carry each scenario role (no name bias).
    pool = ids[:]
    rng.shuffle(pool)

    def take() -> str:
        return pool.pop()

    # ---- Scenario A: churn mismatch (CRM churned, Billing still charging) ----
    # Correct: stop billing in Billing AND add a case note in CRM.
    for _ in range(3):
        aid = take()
        a = accounts[aid]
        a.scenario = "reconcile_churn"
        a.crm_status = "churned"
        a.billing_on = True
        a.invoices = [Invoice(f"INV-{aid.split('-')[1]}A", "Sep 2026", a.price, "paid")]
        scenarios.append(Scenario(
            scenario_id=f"sc_churn_{aid}", kind="reconcile_churn", account_id=aid,
            brief=("the CRM shows this customer as churned, but a teammate noticed "
                   "they appear to still be getting billed"),
            systems=["crm", "billing"],
            verify={"type": "all_of", "checks": [
                {"system": "billing", "type": "cancel_billing", "subscription_id": _sub_id(aid)},
                {"system": "crm", "type": "note_added", "account_id": aid},
            ]},
        ))

    # ---- Scenario B: reconciled refund (Billing duplicate, CRM active owner) ----
    # Correct: refund the duplicate in Billing AND add a case note in CRM.
    for _ in range(3):
        aid = take()
        a = accounts[aid]
        a.scenario = "reconcile_refund"
        a.crm_status = "active"
        tag = aid.split("-")[1]
        a.invoices = [
            Invoice(f"INV-{tag}A", "Sep 2026", a.price, "paid"),
            Invoice(f"INV-{tag}B", "Sep 2026", a.price, "paid"),   # duplicate (same period/amount)
            Invoice(f"INV-{tag}C", "Aug 2026", a.price, "paid"),
        ]
        scenarios.append(Scenario(
            scenario_id=f"sc_refund_{aid}", kind="reconcile_refund", account_id=aid,
            brief="this customer says they were charged twice this month and wants it fixed",
            systems=["crm", "billing"],
            verify={"type": "all_of", "checks": [
                {"system": "billing", "type": "refund_any_of",
                 "invoice_ids": [f"INV-{tag}A", f"INV-{tag}B"]},
                {"system": "crm", "type": "note_added", "account_id": aid},
            ]},
        ))

    # ---- Scenario C: block policy (Billing-only, >=3 failed) ----
    for _ in range(2):
        aid = take()
        a = accounts[aid]
        a.scenario = "block_policy"
        a.failed_payments = 3
        tag = aid.split("-")[1]
        a.invoices = [
            Invoice(f"INV-{tag}A", "Jun 2026", a.price, "paid"),
            Invoice(f"INV-{tag}B", "Jul 2026", a.price, "failed"),
            Invoice(f"INV-{tag}C", "Aug 2026", a.price, "failed"),
            Invoice(f"INV-{tag}D", "Sep 2026", a.price, "failed"),
        ]
        scenarios.append(Scenario(
            scenario_id=f"sc_block_{aid}", kind="block_policy", account_id=aid,
            brief=("billing flagged this account for repeated failed payments; policy "
                   f"is to block a subscription once it reaches {BLOCK_POLICY}+ "
                   "consecutive failed payments"),
            systems=["billing"],
            verify={"type": "all_of", "checks": [
                {"system": "billing", "type": "block_subscription", "subscription_id": _sub_id(aid)},
            ]},
        ))

    # ---- Scenario D: below policy (2 failed) -> NO action ----
    for _ in range(2):
        aid = take()
        a = accounts[aid]
        a.scenario = "below_policy"
        a.failed_payments = 2
        tag = aid.split("-")[1]
        a.invoices = [
            Invoice(f"INV-{tag}A", "Jul 2026", a.price, "paid"),
            Invoice(f"INV-{tag}B", "Aug 2026", a.price, "failed"),
            Invoice(f"INV-{tag}C", "Sep 2026", a.price, "failed"),
        ]
        scenarios.append(Scenario(
            scenario_id=f"sc_below_{aid}", kind="below_policy", account_id=aid,
            brief=("billing flagged this account for failed payments; policy is to "
                   f"block only once it reaches {BLOCK_POLICY}+ consecutive failed payments"),
            systems=["billing"],
            verify={"type": "no_change_all"},
        ))

    # ---- Scenario E: no billing error -> NO refund ----
    for _ in range(2):
        aid = take()
        a = accounts[aid]
        a.scenario = "no_error_no_refund"
        a.crm_status = "active"
        # single legit paid invoice (already the default) -> nothing to fix
        scenarios.append(Scenario(
            scenario_id=f"sc_noerr_{aid}", kind="no_error_no_refund", account_id=aid,
            brief=("this customer asked for a refund because they forgot to cancel in "
                   "time; only a genuine billing error (e.g. a duplicate charge) warrants a refund"),
            systems=["crm", "billing"],
            verify={"type": "no_change_all"},
        ))

    return accounts, scenarios


# convenience for debugging
if __name__ == "__main__":
    accs, scns = generate_world()
    print(f"World SEED={SEED}: {len(accs)} accounts, {len(scns)} scenarios")
    for s in scns:
        a = accs[s.account_id]
        print(f"  [{s.kind:<18}] {a.account_id} {a.name:<18} "
              f"crm={a.crm_status:<9} billing={'on' if a.billing_on else 'off'} "
              f"failed={a.failed_payments} -> {s.verify['type']}")
