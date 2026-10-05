"""
metrics.py — ground-truth analytics computed directly from a Billing snapshot.

Used to (a) back the /api/report endpoint and (b) produce verifiable answers for
"analytical report" tasks (count defaulters, total MRR, paid-this-month, ...).
Everything is deterministic from the DB, so report-style tasks are auto-scorable
by exact-value match (no LLM judge).
"""
from __future__ import annotations


def filter_subscriptions(snapshot: dict, *, status="all", billing="all",
                         min_failed=0, plan="all") -> list[dict]:
    """Return subscription rows (joined with account name) matching the filters."""
    accts = {a["account_id"]: a for a in snapshot.get("accounts", [])}
    out = []
    for s in snapshot.get("subscriptions", []):
        if status != "all" and s["status"] != status:
            continue
        if billing != "all" and s["billing"] != billing:
            continue
        if plan != "all" and s["plan"] != plan:
            continue
        if s["failed_payments"] < int(min_failed):
            continue
        a = accts.get(s["account_id"], {})
        out.append({**s, "name": a.get("name", ""), "email": a.get("email", "")})
    return out


def aggregates(rows: list[dict]) -> dict:
    """Count + monthly recurring revenue (sum of price) over the given rows."""
    return {
        "count": len(rows),
        "mrr": round(sum(r["price"] for r in rows), 2),
    }


# ---- named ground-truth metrics for report tasks ----
def metric(snapshot: dict, name: str, **params) -> float | int:
    if name == "defaulters_count":              # subs with >=1 failed payment
        return len(filter_subscriptions(snapshot, min_failed=1))
    if name == "block_candidates_count":        # subs meeting the >=3 policy
        return len(filter_subscriptions(snapshot, min_failed=3))
    if name == "active_mrr":                     # MRR of billing=on subs
        return aggregates(filter_subscriptions(snapshot, billing="on"))["mrr"]
    if name == "blocked_count":
        return len(filter_subscriptions(snapshot, status="blocked"))
    if name == "paid_invoices_in_period":        # count of paid invoices in a period
        period = params["period"]
        return sum(1 for i in snapshot.get("invoices", [])
                   if i["period"] == period and i["status"] == "paid")
    if name == "revenue_in_period":              # sum of paid invoice amounts in a period
        period = params["period"]
        return round(sum(i["amount"] for i in snapshot.get("invoices", [])
                         if i["period"] == period and i["status"] == "paid"), 2)
    raise ValueError(f"unknown metric: {name}")
