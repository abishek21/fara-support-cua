"""
propose_tasks.py — GPT-5.4 task proposer for the Support Console (FaraGen-style).

Design (reliability-first):
  * WE scan the DB and detect scenario-ready accounts, deriving the ground-truth
    `verify` predicate deterministically (never trust the LLM for ground truth).
  * GPT-5.4 writes only the natural-language support ticket — varied by persona /
    phrasing style / channel — WITHOUT revealing IDs or the decision, mirroring the
    paper's task-proposal dimensions. If the key isn't set, falls back to templates.

Output: tasks_generated.jsonl (same schema as tasks.jsonl, so benchmark.py can run it).

Usage:
    python propose_tasks.py --n 2                 # 2 phrasings per detected scenario
    python propose_tasks.py --n 2 --no-llm        # template phrasings (no Azure call)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import urllib.request
from collections import defaultdict

HERE = os.path.dirname(__file__)
SUPPORT_URL = "http://localhost:8000"
OUT = os.path.join(HERE, "tasks_generated.jsonl")

PERSONAS = ["busy professional", "frustrated long-time customer", "polite first-time user",
            "terse power user", "confused elderly customer", "detail-oriented analyst"]
STYLES = ["formal email", "casual chat message", "short ticket note",
          "voicemail transcript", "escalation from tier-0"]

PREAMBLE = ("You are a Contoso support agent using the support console already open "
            "in the browser. Use only this console; do not navigate to other websites "
            "or use web search. ")


def _get(path: str) -> dict:
    with urllib.request.urlopen(SUPPORT_URL + path, timeout=10) as r:
        return json.load(r)


# --------------------------------------------------------------------------
# Scenario detection — derive ground truth from the DB
# --------------------------------------------------------------------------
def detect_scenarios(snapshot: dict) -> list[dict]:
    """Return scenario instances: {kind, customer, evidence, verify}."""
    customers = {c["id"]: c for c in snapshot["customers"]}
    subs = snapshot["subscriptions"]
    invs = snapshot["invoices"]
    inv_by_sub = defaultdict(list)
    for i in invs:
        inv_by_sub[i["subscription_id"]].append(i)

    scenarios = []

    # duplicate charge -> refund one of the pair
    for sub in subs:
        paid = [i for i in inv_by_sub[sub["id"]] if i["status"] == "paid"]
        seen = defaultdict(list)
        for i in paid:
            seen[(i["period"], i["amount"])].append(i["id"])
        for (period, amount), ids in seen.items():
            if len(ids) >= 2:
                scenarios.append({
                    "kind": "refund_duplicate",
                    "customer": customers[sub["customer_id"]],
                    "evidence": f"two identical charges for {period} (${amount})",
                    "verify": {"type": "refund_any_of", "invoice_ids": sorted(ids)},
                })

    # subscription at/above block policy (>=3 failed) -> block
    for sub in subs:
        if sub["failed_payments"] >= 3:
            scenarios.append({
                "kind": "block_policy",
                "customer": customers[sub["customer_id"]],
                "evidence": f"{sub['failed_payments']} consecutive failed payments",
                "verify": {"type": "block_subscription", "subscription_id": sub["id"]},
            })

    # below policy (1-2 failed) -> do NOT block
    for sub in subs:
        if 1 <= sub["failed_payments"] <= 2:
            scenarios.append({
                "kind": "below_policy_no_block",
                "customer": customers[sub["customer_id"]],
                "evidence": f"only {sub['failed_payments']} failed payment(s), below the 3-strike policy",
                "verify": {"type": "no_change"},
            })

    # normal account, customer requests cancellation -> cancel billing
    for sub in subs:
        if sub["billing"] == "on" and sub["status"] == "active" and sub["failed_payments"] == 0:
            scenarios.append({
                "kind": "cancel_request",
                "customer": customers[sub["customer_id"]],
                "evidence": f"active {sub['plan']} subscription, customer wants to stop billing",
                "verify": {"type": "cancel_billing", "subscription_id": sub["id"]},
            })

    # single legit paid invoice, no duplicate, no failures -> do NOT refund
    for sub in subs:
        sinv = inv_by_sub[sub["id"]]
        if (len(sinv) == 1 and sinv[0]["status"] == "paid"
                and sub["failed_payments"] == 0):
            scenarios.append({
                "kind": "no_error_no_refund",
                "customer": customers[sub["customer_id"]],
                "evidence": "a single legitimate charge, no billing error",
                "verify": {"type": "no_change"},
            })

    return scenarios


# scenario kind -> what the ticket should ask for (without revealing the answer)
KIND_BRIEF = {
    "refund_duplicate": "believes they were charged twice this month and wants it fixed",
    "block_policy": "flagged for failed payments; apply the policy of blocking at 3+ consecutive failed payments",
    "below_policy_no_block": "flagged for failed payments; apply the policy of blocking at 3+ consecutive failed payments",
    "cancel_request": "wants to stop being billed going forward",
    "no_error_no_refund": "is asking for a refund on a recent charge",
}


def template_instruction(sc: dict) -> str:
    c = sc["customer"]
    brief = KIND_BRIEF[sc["kind"]]
    return (PREAMBLE + f"{c['name']} contacted support and {brief}. "
            "Find the account, review the relevant billing details, decide the correct "
            "action per policy, and take it. If no action is warranted, do not make changes.")


def llm_instruction(llm, sc: dict, persona: str, style: str) -> str:
    c = sc["customer"]
    brief = KIND_BRIEF[sc["kind"]]
    system = (
        "You write realistic customer-support tickets for a SaaS billing console. "
        "Output ONE short ticket (2-4 sentences) as plain text. "
        "Rules: mention the customer's full name; do NOT reveal any IDs (customer/invoice/"
        "subscription); do NOT state the final decision or whether to act — leave the judgment "
        "to the support agent; do not invent policy numbers beyond what is given."
    )
    user = (
        f"Customer: {c['name']}\n"
        f"Situation: the customer {brief}.\n"
        f"Persona: {persona}\nChannel/style: {style}\n\n"
        "Write the ticket text only (no preamble)."
    )
    body = llm.chat(system, user, temperature=0.8).strip()
    return PREAMBLE + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1, help="phrasings per scenario instance")
    ap.add_argument("--no-llm", action="store_true", help="use templates, skip Azure")
    ap.add_argument("--max-per-kind", type=int, default=3,
                    help="cap scenario instances per kind (distractors are plentiful)")
    args = ap.parse_args()

    snap = _get("/api/snapshot")
    scenarios = detect_scenarios(snap)

    # cap per kind so we don't emit 100 near-identical 'no_error' tasks
    by_kind = defaultdict(list)
    for s in scenarios:
        by_kind[s["kind"]].append(s)
    rng = random.Random(0)
    picked = []
    for kind, items in by_kind.items():
        rng.shuffle(items)
        picked.extend(items[: args.max_per_kind])

    print(f"Detected {len(scenarios)} scenario instances; using {len(picked)} "
          f"(<= {args.max_per_kind}/kind) across {len(by_kind)} kinds:")
    for k, v in by_kind.items():
        print(f"  {k}: {len(v)} available")

    llm = None
    if not args.no_llm:
        from azure_llm import AzureLLM
        llm = AzureLLM()

    out = []
    for sc in picked:
        for j in range(args.n):
            if llm:
                persona = rng.choice(PERSONAS)
                style = rng.choice(STYLES)
                instruction = llm_instruction(llm, sc, persona, style)
            else:
                instruction = template_instruction(sc)
            out.append({
                "task_id": f"gen_{sc['kind']}_{sc['customer']['id']}_{j}",
                "generated": True,
                "kind": sc["kind"],
                "customer": sc["customer"]["name"],
                "evidence": sc["evidence"],
                "instruction": instruction,
                "verify": sc["verify"],
            })

    with open(OUT, "w") as f:
        for t in out:
            f.write(json.dumps(t) + "\n")
    print(f"\nWrote {len(out)} tasks -> {OUT}")
    for t in out[:4]:
        print(f"\n[{t['kind']}] {t['customer']} -> {t['verify']['type']}")
        print(" ", t["instruction"][:200])


if __name__ == "__main__":
    main()
