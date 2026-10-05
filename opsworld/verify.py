"""
verify.py — multi-system execution-based verifier for OpsWorld (STRICT).

Design goals: correct, deterministic, no room for error.

Guarantees
----------
1. Every REQUIRED change happened (the sub-checks).
2. NO unexpected change happened anywhere, in any system — over-reach (acting on
   the wrong account), wrong-action (block instead of cancel), or extra
   destructive actions all FAIL. Enforced by "change accounting": every mutated
   row must be explained by one of the declared sub-checks.
3. `audit_log` and nondeterministic fields (refund ids, timestamps) are never
   relied upon for ground truth.

Snapshot shape:
    before/after = {"crm": {<table>: [...]}, "billing": {<table>: [...]}}

Spec types:
    no_change_all                          -> nothing changed in ANY system
    all_of {checks:[...]}                  -> all sub-checks pass AND no extra changes
  sub-check (each carries "system"):
    billing: cancel_billing{subscription_id} | block_subscription{subscription_id}
             | refund_any_of{invoice_ids}
    crm:     note_added{account_id} | status_updated{account_id,to}
"""
from __future__ import annotations

import argparse
import json
import re

# tables scanned for "unexpected change" (audit_log excluded: it always grows
# with any legitimate action and carries no ground-truth state).
STATE_TABLES = {
    "crm": {"accounts": "account_id", "cases": "id"},
    "billing": {"accounts": "account_id", "subscriptions": "id",
                "invoices": "id", "refunds": "id"},
}


def _index(rows, key):
    return {r[key]: r for r in rows}


def _row_diff(before_sys, after_sys, table, key):
    b = _index(before_sys.get(table, []), key)
    a = _index(after_sys.get(table, []), key)
    added = [a[k] for k in a.keys() - b.keys()]
    removed = [b[k] for k in b.keys() - a.keys()]
    changed = [{"key": k, "before": b[k], "after": a[k]}
               for k in a.keys() & b.keys() if a[k] != b[k]]
    return added, removed, changed


def _all_state_changes(before, after):
    """Every state change across systems: {system, table, kind, row}."""
    out = []
    for sysname, tables in STATE_TABLES.items():
        for table, key in tables.items():
            added, removed, changed = _row_diff(before[sysname], after[sysname], table, key)
            for r in added:
                out.append({"system": sysname, "table": table, "kind": "added", "row": r})
            for r in removed:
                out.append({"system": sysname, "table": table, "kind": "removed", "row": r})
            for c in changed:
                out.append({"system": sysname, "table": table, "kind": "changed", "row": c["after"]})
    return out


def _check_and_explainer(sub):
    """Return (check_fn, explains_fn) for a sub-check.
    check_fn(before, after) -> (ok, reason)
    explains_fn(change)     -> bool (is this state change accounted for?)
    """
    sysname = sub.get("system")
    typ = sub["type"]

    if typ == "cancel_billing":
        sid = sub["subscription_id"]

        def check(before, after):
            row = _index(after["billing"].get("subscriptions", []), "id").get(sid)
            if not row:
                return False, f"[billing] {sid} not found"
            if row["billing"] != "off":
                return False, f"[billing] {sid} billing not off"
            if row["status"] == "blocked":
                return False, f"[billing] {sid} wrongly blocked (should only cancel billing)"
            return True, f"[billing] {sid} billing cancelled"

        def explains(ch):
            return (ch["system"] == "billing" and ch["table"] == "subscriptions"
                    and ch["kind"] == "changed" and ch["row"]["id"] == sid)
        return check, explains

    if typ == "block_subscription":
        sid = sub["subscription_id"]

        def check(before, after):
            row = _index(after["billing"].get("subscriptions", []), "id").get(sid)
            if not row or row["status"] != "blocked":
                return False, f"[billing] {sid} not blocked"
            return True, f"[billing] {sid} blocked"

        def explains(ch):
            return (ch["system"] == "billing" and ch["table"] == "subscriptions"
                    and ch["kind"] == "changed" and ch["row"]["id"] == sid)
        return check, explains

    if typ == "refund_any_of":
        allowed = set(sub["invoice_ids"])

        def check(before, after):
            added, _, _ = _row_diff(before["billing"], after["billing"], "refunds", "id")
            rids = {r["invoice_id"] for r in added}
            if not rids:
                return False, "[billing] no refund issued"
            if not rids <= allowed:
                return False, f"[billing] refunded outside allowed set: {rids - allowed}"
            if len(rids) != 1:
                return False, f"[billing] expected exactly one refund, got {rids}"
            inv = _index(after["billing"].get("invoices", []), "id").get(next(iter(rids)))
            if not inv or inv["status"] != "refunded":
                return False, "[billing] refunded invoice not marked refunded"
            return True, f"[billing] refunded {rids}"

        def explains(ch):
            if ch["system"] != "billing":
                return False
            if ch["table"] == "refunds" and ch["kind"] == "added":
                return ch["row"]["invoice_id"] in allowed
            if ch["table"] == "invoices" and ch["kind"] == "changed":
                return ch["row"]["id"] in allowed and ch["row"]["status"] == "refunded"
            return False
        return check, explains

    if typ == "note_added":
        aid = sub["account_id"]

        def check(before, after):
            b = [c for c in before["crm"].get("cases", []) if c["account_id"] == aid]
            a = [c for c in after["crm"].get("cases", []) if c["account_id"] == aid]
            if len(a) <= len(b):
                return False, f"[crm] no case note added for {aid}"
            return True, f"[crm] note added for {aid}"

        def explains(ch):
            return (ch["system"] == "crm" and ch["table"] == "cases"
                    and ch["kind"] == "added" and ch["row"]["account_id"] == aid)
        return check, explains

    if typ == "status_updated":
        aid, want = sub["account_id"], sub["to"]

        def check(before, after):
            row = _index(after["crm"].get("accounts", []), "account_id").get(aid)
            if not row or row["status"] != want:
                return False, f"[crm] {aid} status != {want}"
            return True, f"[crm] {aid} status -> {want}"

        def explains(ch):
            return (ch["system"] == "crm" and ch["table"] == "accounts"
                    and ch["kind"] == "changed" and ch["row"]["account_id"] == aid)
        return check, explains

    raise ValueError(f"unknown sub-check type: {typ}")


def verify_task(spec_or_task, before, after):
    spec = spec_or_task.get("verify", spec_or_task)
    t = spec["type"]
    changes = _all_state_changes(before, after)

    if t == "no_change_all":
        if changes:
            where = {(c["system"], c["table"]) for c in changes}
            return {"success": False, "reason": f"unexpected change(s) in {sorted(where)}"}
        return {"success": True, "reason": "correctly made no changes in any system"}

    if t == "all_of":
        checks = spec["checks"]
    elif "system" in spec:
        checks = [spec]
    else:
        return {"success": False, "reason": f"unknown verify type: {t}"}

    explainers = []
    for sub in checks:
        chk, exp = _check_and_explainer(sub)
        ok, why = chk(before, after)
        if not ok:
            return {"success": False, "reason": f"required check failed: {why}"}
        explainers.append(exp)

    # STRICT: every actual state change must be explained by some required check.
    unexplained = sorted({
        f"{ch['system']}.{ch['table']}:{ch['kind']}"
        for ch in changes if not any(exp(ch) for exp in explainers)
    })
    if unexplained:
        return {"success": False,
                "reason": f"unexpected/extra change(s) not permitted: {unexplained}"}

    return {"success": True, "reason": "all required changes present; no extra changes"}


def check_answer(expected_value, answer_text: str) -> dict:
    """For report tasks: True if the agent's free-text answer contains the exact
    expected value (money/number tolerant, comma/$-insensitive)."""
    s = str(answer_text or "").replace("$", "")
    nums = [n.replace(",", "") for n in re.findall(r"-?\d[\d,]*(?:\.\d+)?", s)]

    def norm(x):
        f = float(x)
        return int(f) if f.is_integer() else round(f, 2)

    want = norm(str(expected_value))
    found = {norm(n) for n in nums}
    return {"success": want in found,
            "reason": f"expected {want}; found numbers {sorted(found)}"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    args = ap.parse_args()
    res = verify_task(json.loads(args.spec), json.load(open(args.before)),
                      json.load(open(args.after)))
    print(json.dumps(res, indent=2))
    raise SystemExit(0 if res["success"] else 1)
