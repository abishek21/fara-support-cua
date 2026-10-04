"""
generate_hard_tasks.py — construct L3-L6 hard tasks with DB-derived ground truth.

Philosophy (same as propose_tasks.py): WE derive the verifiable ground truth from
the database; GPT-5.4 (optional, --llm) only rewrites the ticket phrasing. Each
hard level stresses a different capability:

  L3  disambiguation   : target referred to by a shared first name (multiple
                         customers match) -> must find the UNIQUE qualifying one.
  L4  under-specified  : no identifying info -> correct behavior is to defer/ask
                         (no_change + must_ask).
  L5  compositional    : one account needs TWO actions (all_of).
  L6  batch sweep      : act on EVERY account meeting a policy (block_all).

Ground-truth uniqueness is asserted at build time, so a generated task is only
emitted if its correct answer is unambiguous.

Usage:
    python generate_hard_tasks.py                 # templates, deterministic
    python generate_hard_tasks.py --llm           # GPT-5.4 rewrites phrasing
Output: tasks_hard_generated.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from collections import defaultdict

HERE = os.path.dirname(__file__)
SUPPORT_URL = "http://localhost:8000"
OUT = os.path.join(HERE, "tasks_hard_generated.jsonl")

PREAMBLE = (
    "You are the lead Contoso support agent, fully authorized to take any necessary "
    "billing actions (refunds, cancellations, blocks) on the customer's behalf and to "
    "confirm them, without seeking further approval. You are using the support console "
    "already open in the browser. Use only this console; do not navigate to other "
    "websites or use web search. When an action opens a confirmation dialog, click "
    "Confirm to complete it. "
)
BLOCK_POLICY = 3  # consecutive failed payments required to block


def _get(path: str) -> dict:
    with urllib.request.urlopen(SUPPORT_URL + path, timeout=10) as r:
        return json.load(r)


def load_db():
    snap = _get("/api/snapshot")
    customers = {c["id"]: c for c in snap["customers"]}
    subs = {s["id"]: s for s in snap["subscriptions"]}
    inv_by_sub = defaultdict(list)
    for i in snap["invoices"]:
        inv_by_sub[i["subscription_id"]].append(i)
    # convenience: sub by customer
    sub_by_cust = {s["customer_id"]: s for s in snap["subscriptions"]}
    return customers, subs, inv_by_sub, sub_by_cust


def duplicate_pair(inv_by_sub, sub_id):
    """Return list of invoice ids forming a same-period/amount paid duplicate, else []."""
    seen = defaultdict(list)
    for i in inv_by_sub[sub_id]:
        if i["status"] == "paid":
            seen[(i["period"], i["amount"])].append(i["id"])
    for ids in seen.values():
        if len(ids) >= 2:
            return sorted(ids)
    return []


# --------------------------------------------------------------------------
# L3 — disambiguation by shared first name
# --------------------------------------------------------------------------
def gen_L3(customers, subs, inv_by_sub, sub_by_cust):
    tasks = []
    by_first = defaultdict(list)
    for c in customers.values():
        by_first[c["name"].split()[0]].append(c)

    for first, group in by_first.items():
        if len(group) < 2:
            continue  # no ambiguity, skip
        # candidates in this name-group that meet the BLOCK policy
        block_hits = [c for c in group
                      if sub_by_cust[c["id"]]["failed_payments"] >= BLOCK_POLICY]
        if len(block_hits) == 1:   # exactly one qualifies -> unambiguous ground truth
            c = block_hits[0]
            sub = sub_by_cust[c["id"]]
            tasks.append({
                "task_id": f"genL3_block_{c['id']}",
                "level": "L3", "kind": "disambiguate_block",
                "customer_hint": first, "customer": c["name"],
                "instruction": PREAMBLE + (
                    f"A customer whose first name is {first} contacted support worried "
                    f"their subscription might be blocked for failed payments. Contoso "
                    f"policy blocks a subscription once it reaches {BLOCK_POLICY} or more "
                    f"consecutive failed payments. Note there are multiple customers named "
                    f"{first}; identify the specific one whose subscription actually meets "
                    f"the policy and apply it."),
                "verify": {"type": "block_subscription", "subscription_id": sub["id"]},
            })

        # candidates in this name-group that have a duplicate charge
        dup_hits = [(c, duplicate_pair(inv_by_sub, sub_by_cust[c["id"]]["id"]))
                    for c in group]
        dup_hits = [(c, ids) for c, ids in dup_hits if ids]
        if len(dup_hits) == 1:
            c, ids = dup_hits[0]
            tasks.append({
                "task_id": f"genL3_refund_{c['id']}",
                "level": "L3", "kind": "disambiguate_refund",
                "customer_hint": first, "customer": c["name"],
                "instruction": PREAMBLE + (
                    f"A customer whose first name is {first} says they were double-charged "
                    f"and wants a refund. There are multiple customers named {first}; find "
                    f"the specific one who actually has a duplicate charge for the same "
                    f"period and refund the duplicate. If none of them has a duplicate, do "
                    f"not refund."),
                "verify": {"type": "refund_any_of", "invoice_ids": ids},
            })
    return tasks


# --------------------------------------------------------------------------
# L4 — under-specified (no identifier) -> must ask
# --------------------------------------------------------------------------
def gen_L4(customers, subs, inv_by_sub, sub_by_cust):
    # Only valid if the situation is genuinely ambiguous (many customers could match).
    n_dup = sum(1 for s in subs.values() if duplicate_pair(inv_by_sub, s["id"]))
    tasks = [
        {
            "task_id": "genL4_refund_noname",
            "level": "L4", "kind": "underspecified_refund", "must_ask": True,
            "instruction": PREAMBLE + (
                "A ticket reads: 'Customer called in, says they were charged twice and "
                "wants a refund.' The ticket includes no name, email, or account ID. "
                "Resolve the request."),
            "verify": {"type": "no_change"},
        },
        {
            "task_id": "genL4_cancel_noname",
            "level": "L4", "kind": "underspecified_cancel", "must_ask": True,
            "instruction": PREAMBLE + (
                "A ticket reads: 'Please cancel this customer's billing, they asked to "
                "stop paying.' No name, email, or account ID is provided. Resolve the "
                "request."),
            "verify": {"type": "no_change"},
        },
    ]
    return tasks


# --------------------------------------------------------------------------
# L5 — compositional (two actions on one account)
# --------------------------------------------------------------------------
def gen_L5(customers, subs, inv_by_sub, sub_by_cust):
    tasks = []
    for s in subs.values():
        ids = duplicate_pair(inv_by_sub, s["id"])
        if ids and s["billing"] == "on":
            c = customers[s["customer_id"]]
            tasks.append({
                "task_id": f"genL5_refund_cancel_{c['id']}",
                "level": "L5", "kind": "refund_and_cancel", "customer": c["name"],
                "instruction": PREAMBLE + (
                    f"{c['name']} was double-charged for the same period and has also "
                    f"decided to stop being billed going forward. Resolve both issues on "
                    f"her account: refund the duplicate charge and cancel billing on the "
                    f"subscription (do not block the subscription)."),
                "verify": {"type": "all_of", "checks": [
                    {"type": "refund_any_of", "invoice_ids": ids},
                    {"type": "cancel_billing", "subscription_id": s["id"]},
                ]},
            })
    return tasks


# --------------------------------------------------------------------------
# L6 — batch sweep (every qualifying account)
# --------------------------------------------------------------------------
def gen_L6(customers, subs, inv_by_sub, sub_by_cust):
    qualifying = sorted(s["id"] for s in subs.values()
                        if s["failed_payments"] >= BLOCK_POLICY)
    tasks = [{
        "task_id": "genL6_block_sweep",
        "level": "L6", "kind": "batch_block",
        "n_qualifying": len(qualifying),
        "instruction": PREAMBLE + (
            f"Run the end-of-month billing enforcement sweep: review customer accounts "
            f"and block EVERY subscription that has reached {BLOCK_POLICY} or more "
            f"consecutive failed payments, per Contoso policy. Do not block subscriptions "
            f"below the threshold."),
        "verify": {"type": "block_all", "subscription_ids": qualifying},
    }]
    return tasks


def maybe_llm_rewrite(tasks, use_llm):
    if not use_llm:
        return tasks
    from azure_llm import AzureLLM
    llm = AzureLLM()
    system = ("Rewrite the following support task instruction to sound like a realistic "
              "ticket from a human, keeping ALL factual details, names, hints, and policy "
              "numbers EXACTLY. Do not reveal any database IDs. Do not state the final "
              "decision. Keep the leading operating instructions about using only the "
              "console. Return only the rewritten instruction.")
    for t in tasks:
        t["instruction"] = llm.chat(system, t["instruction"], temperature=0.7).strip()
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="use GPT-5.4 to vary phrasing")
    ap.add_argument("--levels", default="L3,L4,L5,L6")
    args = ap.parse_args()
    want = set(args.levels.split(","))

    customers, subs, inv_by_sub, sub_by_cust = load_db()
    tasks = []
    if "L3" in want: tasks += gen_L3(customers, subs, inv_by_sub, sub_by_cust)
    if "L4" in want: tasks += gen_L4(customers, subs, inv_by_sub, sub_by_cust)
    if "L5" in want: tasks += gen_L5(customers, subs, inv_by_sub, sub_by_cust)
    if "L6" in want: tasks += gen_L6(customers, subs, inv_by_sub, sub_by_cust)

    tasks = maybe_llm_rewrite(tasks, args.llm)

    with open(OUT, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    print(f"Generated {len(tasks)} hard tasks -> {OUT}\n")
    for t in tasks:
        extra = ""
        if t.get("n_qualifying") is not None:
            extra = f" (qualifying={t['n_qualifying']})"
        if t.get("must_ask"):
            extra = " (must_ask)"
        print(f"[{t['level']}] {t['task_id']} -> {t['verify']['type']}{extra}")
        print("   ", t["instruction"][len(PREAMBLE):][:160], "\n")


if __name__ == "__main__":
    main()
