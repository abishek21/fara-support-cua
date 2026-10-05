"""phrase.py — turn a world Scenario into a name-only task instruction (C1-C6)."""
from __future__ import annotations

_PRE = ("You are the lead Contoso support agent, authorized to take and confirm any "
        "necessary actions across the CRM and Billing systems. Use only those two "
        "systems; do not use web search. ")


def phrase(scn, accounts) -> str:
    name = accounts[scn.account_id].name
    k = scn.kind
    if k == "reconcile_churn":
        return _PRE + (f"A teammate flagged that {name} shows as churned in the CRM but may "
                       "still be getting billed. Investigate across both systems; if that's "
                       "the case, stop the billing and add a case note in the CRM explaining "
                       "what you did.")
    if k == "reconcile_refund":
        return _PRE + (f"{name} says they were charged twice for the same period this month. "
                       "Check their billing; if there is a genuine duplicate charge, refund the "
                       "duplicate and add a case note in the CRM. If not, do not refund.")
    if k == "block_policy":
        return _PRE + (f"{name}'s account was flagged for failed payments. Contoso policy is to "
                       "block a subscription once it reaches 3 or more consecutive failed "
                       "payments. Review and apply the policy.")
    if k == "below_policy":
        return _PRE + (f"{name}'s account was flagged for failed payments. Contoso policy is to "
                       "block only at 3 or more consecutive failed payments. Review and apply "
                       "the policy.")
    if k == "no_error_no_refund":
        return _PRE + (f"{name} asked for a refund because they forgot to cancel in time. Only a "
                       "genuine billing error such as a duplicate charge warrants a refund. "
                       "Review and act appropriately.")
    if k == "cancel_request":
        return _PRE + (f"{name} emailed asking to stop being billed going forward. Find their "
                       "account and handle the request appropriately.")
    return _PRE + f"Review {name}'s account and take the appropriate action."


# map scenario kind -> category label
CATEGORY = {
    "reconcile_refund": "C1_refund",
    "cancel_request": "C2_cancel",
    "block_policy": "C3_block",
    "reconcile_churn": "C4_churn",
    "below_policy": "C5_below",
    "no_error_no_refund": "C6_noerror",
}
