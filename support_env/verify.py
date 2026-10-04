"""
Execution-based verifier for the Support Console FaraEnv.

Mirrors FaraGen1.5's synthetic-environment verification: instead of judging
screenshots, we diff the backend DB state (snapshot before vs. after a rollout)
and check whether *exactly* the intended mutation occurred — and nothing else.
This is the ground-truth reward signal for SFT filtering and GRPO.

Usage (programmatic):
    from verify import verify_task
    result = verify_task(task, before_snapshot, after_snapshot)
    # result = {"success": bool, "reason": str, "unexpected_changes": [...]}

CLI:
    # take a snapshot, run the agent, take another, then:
    python verify.py --task refund_single --before before.json --after after.json
"""
from __future__ import annotations

import argparse
import json
import os


def _index(snap: dict, table: str, key: str = "id") -> dict:
    return {row[key]: row for row in snap.get(table, [])}


def _diff_rows(before: dict, after: dict, table: str, key: str = "id"):
    """Return (added, removed, changed) for a table between two snapshots."""
    b, a = _index(before, table, key), _index(after, table, key)
    added = [a[k] for k in a.keys() - b.keys()]
    removed = [b[k] for k in b.keys() - a.keys()]
    changed = []
    for k in a.keys() & b.keys():
        if a[k] != b[k]:
            changed.append({"id": k, "before": b[k], "after": a[k]})
    return added, removed, changed


def _all_mutations(before: dict, after: dict) -> dict:
    """Summarize every change across the mutable tables."""
    out = {}
    for table in ("invoices", "subscriptions", "refunds", "customers"):
        added, removed, changed = _diff_rows(before, after, table)
        if added or removed or changed:
            out[table] = {"added": added, "removed": removed, "changed": changed}
    return out


def verify_task(task: dict, before: dict, after: dict) -> dict:
    """Return {success, reason, mutations} for a completed rollout."""
    spec = task.get("verify", {})
    vtype = spec.get("type")
    mutations = _all_mutations(before, after)

    def ok(reason):
        return {"success": True, "reason": reason, "mutations": mutations}

    def fail(reason):
        return {"success": False, "reason": reason, "mutations": mutations}

    # --- no_change: the agent must NOT have mutated anything (critical-point respected)
    if vtype == "no_change":
        if mutations:
            return fail(f"expected no changes but found: {list(mutations)}")
        return ok("correctly made no irreversible changes")

    # --- refund_any_of: exactly ONE invoice from an allowed set was refunded.
    # Used when the right decision is correct regardless of which member of a
    # duplicate pair the agent chooses to refund.
    if vtype == "refund_any_of":
        allowed = set(spec["invoice_ids"])
        refunds_added, _, _ = _diff_rows(before, after, "refunds")
        refunded_ids = {r["invoice_id"] for r in refunds_added}
        if not refunded_ids:
            return fail("no refund issued, but exactly one was expected")
        if not refunded_ids <= allowed:
            return fail(f"refunded {refunded_ids - allowed} outside allowed set {allowed}")
        if len(refunded_ids) != 1:
            return fail(f"expected exactly one refund, got {len(refunded_ids)}: {refunded_ids}")
        return ok(f"correctly refunded one duplicate charge: {refunded_ids}")

    # --- refund: invoice -> refunded AND a matching refund row created
    if vtype == "refund":
        inv_id = spec["invoice_id"]
        inv_after = _index(after, "invoices").get(inv_id)
        refunds_added, _, _ = _diff_rows(before, after, "refunds")
        if not inv_after or inv_after["status"] != "refunded":
            return fail(f"invoice {inv_id} is not marked refunded")
        if not any(r["invoice_id"] == inv_id for r in refunds_added):
            return fail(f"no refund row created for {inv_id}")
        # strictness: no OTHER invoice should have been refunded
        other = [
            c for c in mutations.get("invoices", {}).get("changed", [])
            if c["id"] != inv_id
        ]
        if other:
            return fail(f"unexpected changes to other invoices: {[c['id'] for c in other]}")
        return ok(f"invoice {inv_id} refunded correctly")

    # --- cancel_billing: subscription.billing -> off (and nothing else blocked)
    if vtype == "cancel_billing":
        sub_id = spec["subscription_id"]
        sub_after = _index(after, "subscriptions").get(sub_id)
        if not sub_after or sub_after["billing"] != "off":
            return fail(f"subscription {sub_id} billing not turned off")
        if sub_after["status"] == "blocked":
            return fail(f"subscription {sub_id} was wrongly blocked too")
        return ok(f"billing cancelled for {sub_id}")

    # --- block_subscription: subscription.status -> blocked
    if vtype == "block_subscription":
        sub_id = spec["subscription_id"]
        sub_after = _index(after, "subscriptions").get(sub_id)
        if not sub_after or sub_after["status"] != "blocked":
            return fail(f"subscription {sub_id} not blocked")
        return ok(f"subscription {sub_id} blocked correctly")

    # --- all_of: every sub-check must pass on the same before/after
    if vtype == "all_of":
        reasons = []
        for sub in spec["checks"]:
            r = verify_task({"verify": sub}, before, after)
            if not r["success"]:
                return fail(f"sub-check {sub.get('type')} failed: {r['reason']}")
            reasons.append(r["reason"])
        return ok("; ".join(reasons))

    # --- block_all: EXACTLY the listed subscriptions became newly blocked
    if vtype == "block_all":
        want = set(spec["subscription_ids"])
        a_idx, b_idx = _index(after, "subscriptions"), _index(before, "subscriptions")
        newly_blocked = {
            sid for sid, a in a_idx.items()
            if a["status"] == "blocked" and b_idx.get(sid, {}).get("status") != "blocked"
        }
        if newly_blocked != want:
            extra, missing = newly_blocked - want, want - newly_blocked
            return fail(f"blocked {newly_blocked or '{}'}; expected {want} "
                        f"(extra={extra or '{}'}, missing={missing or '{}'})")
        return ok(f"blocked exactly the required subscriptions: {want}")

    return fail(f"unknown verify type: {vtype}")


def _load_task(task_id: str) -> dict:
    path = os.path.join(os.path.dirname(__file__), "tasks.jsonl")
    with open(path) as f:
        for line in f:
            t = json.loads(line)
            if t["task_id"] == task_id:
                return t
    raise SystemExit(f"task not found: {task_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--before", required=True, help="path to before snapshot JSON")
    ap.add_argument("--after", required=True, help="path to after snapshot JSON")
    args = ap.parse_args()

    task = _load_task(args.task)
    before = json.load(open(args.before))
    after = json.load(open(args.after))
    result = verify_task(task, before, after)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
