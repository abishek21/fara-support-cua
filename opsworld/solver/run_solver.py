"""
run_solver.py — run the GPT-5.4 SoM solver on an OpsWorld task and verify it.

Resets both systems, snapshots before, runs the solver, snapshots after, and
checks the multi-DB verifier. Use this to confirm the solver + SoM adapter work
before scaling to task generation / trajectory collection.

Usage:
    cd opsworld
    ./run_all.sh                                  # start CRM + Billing
    python -m solver.run_solver --scenario reconcile_churn --headful
    python -m solver.run_solver --scenario block_policy
    python -m solver.run_solver --list            # show available scenarios
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.world import generate_world
from solver.gpt_solver import solve, SYSTEMS
from verify import verify_task

CRM, BILL = SYSTEMS["CRM"], SYSTEMS["Billing"]


def get(u):
    return json.load(urllib.request.urlopen(u))


def post(u, b=None):
    r = urllib.request.Request(u, data=json.dumps(b or {}).encode(),
                               headers={"Content-Type": "application/json"}, method="POST")
    return json.load(urllib.request.urlopen(r))


def snap():
    return {"crm": get(CRM + "/api/snapshot"), "billing": get(BILL + "/api/snapshot")}


def build_instruction(scn, accounts) -> str:
    """Phrase a scenario into a task instruction (name only, no IDs, decision open)."""
    a = accounts[scn.account_id]
    pre = ("You are the lead Contoso support agent, authorized to take and confirm "
           "any necessary actions across the CRM and Billing systems. Use only those "
           "two systems. ")
    name = a.name
    if scn.kind == "reconcile_churn":
        return pre + (f"A teammate flagged that {name} shows as churned in the CRM but may "
                      "still be getting billed. Investigate across both systems; if that's "
                      "the case, stop the billing and add a case note in the CRM explaining "
                      "what you did.")
    if scn.kind == "reconcile_refund":
        return pre + (f"{name} says they were charged twice for the same period this month. "
                      "Check their billing; if there is a genuine duplicate charge, refund "
                      "the duplicate and add a case note in the CRM. If not, do not refund.")
    if scn.kind == "block_policy":
        return pre + (f"{name}'s account was flagged for failed payments. Contoso policy is to "
                      "block a subscription once it reaches 3 or more consecutive failed "
                      "payments. Review and apply the policy.")
    if scn.kind == "below_policy":
        return pre + (f"{name}'s account was flagged for failed payments. Contoso policy is to "
                      "block only at 3 or more consecutive failed payments. Review and apply "
                      "the policy.")
    if scn.kind == "no_error_no_refund":
        return pre + (f"{name} asked for a refund because they forgot to cancel in time. Only a "
                      "genuine billing error (such as a duplicate charge) warrants a refund. "
                      "Review and act appropriately.")
    return pre + "Review the account and take the appropriate action."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="reconcile_churn")
    ap.add_argument("--which", type=int, default=0, help="index among that scenario kind")
    ap.add_argument("--headful", action="store_true")
    ap.add_argument("--no-image", action="store_true", help="text-only SoM (cheaper)")
    ap.add_argument("--max_steps", type=int, default=25)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    accounts, scns = generate_world()
    if args.list:
        from collections import Counter
        print("scenario kinds:", dict(Counter(s.kind for s in scns)))
        return

    cands = [s for s in scns if s.kind == args.scenario]
    if not cands:
        sys.exit(f"no scenario of kind {args.scenario}")
    scn = cands[args.which % len(cands)]
    a = accounts[scn.account_id]
    start = BILL if scn.systems[-1] == "billing" and "crm" not in scn.systems else CRM

    task = {
        "task_id": scn.scenario_id,
        "instruction": build_instruction(scn, accounts),
        "start_url": start + "/",
        "verify": scn.verify,
    }
    out_dir = os.path.join(os.path.dirname(__file__), "..", "runs", scn.scenario_id)

    print(f"=== Task: {scn.kind} | {a.name} ({a.account_id}) ===")
    print(task["instruction"], "\n")

    post(CRM + "/api/reset"); post(BILL + "/api/reset")
    before = snap()
    result = solve(task, headful=args.headful, max_steps=args.max_steps,
                   out_dir=out_dir, send_image=not args.no_image)
    after = snap()
    v = verify_task(scn.verify, before, after)

    print("\n================ RESULT ================")
    print(f"steps: {result['n_steps']}  final answer: {result['final_answer']!r}")
    print(f"VERIFY: {'PASS ✅' if v['success'] else 'FAIL ❌'} — {v['reason']}")
    print(f"trajectory saved: {out_dir}")


if __name__ == "__main__":
    main()
