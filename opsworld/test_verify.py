"""
test_verify.py — exhaustive correctness tests for the OpsWorld verifier.

Runs real actions against the live CRM (:8001) + Billing (:8002), snapshots
before/after, and asserts the verifier returns the EXACT expected pass/fail for
every path — including adversarial negatives (over-reach, wrong action, extra
action, partial completion, wrong account). Deterministic; resets between cases.

Prereq: ./run_all.sh  (both systems up)
Run:    python test_verify.py
"""
import json
import sys
import urllib.request

from verify import verify_task, check_answer
from common.world import generate_world
from common import metrics

CRM, BILL = "http://localhost:8001", "http://localhost:8002"


def get(u):
    return json.load(urllib.request.urlopen(u))


def post(u, b):
    r = urllib.request.Request(u, data=json.dumps(b).encode(),
                               headers={"Content-Type": "application/json"}, method="POST")
    return json.load(urllib.request.urlopen(r))


def reset():
    post(CRM + "/api/reset", {}); post(BILL + "/api/reset", {})


def snap():
    return {"crm": get(CRM + "/api/snapshot"), "billing": get(BILL + "/api/snapshot")}


def sid_of(aid):
    return "SUB-" + aid.split("-")[1]


_, SCNS = generate_world()
S = {s.kind: s for s in SCNS}  # first of each kind
churn = next(s for s in SCNS if s.kind == "reconcile_churn")
refund = next(s for s in SCNS if s.kind == "reconcile_refund")
block = next(s for s in SCNS if s.kind == "block_policy")
below = next(s for s in SCNS if s.kind == "below_policy")
noerr = next(s for s in SCNS if s.kind == "no_error_no_refund")
# a DIFFERENT account for over-reach tests
other = next(s for s in SCNS if s.account_id != churn.account_id)

results = []


def case(name, expect_pass, do, spec):
    reset()
    before = snap()
    do()
    after = snap()
    res = verify_task(spec, before, after)
    ok = (res["success"] == expect_pass)
    results.append((ok, name, "PASS" if res["success"] else "FAIL",
                    "PASS" if expect_pass else "FAIL", res["reason"]))


# ---------- churn reconciliation (cancel billing + CRM note) ----------
cs, caid = sid_of(churn.account_id), churn.account_id
case("churn: cancel+note (correct)", True,
     lambda: (post(BILL + "/api/cancel-billing", {"subscription_id": cs}),
              post(CRM + "/api/add-note", {"account_id": caid, "note": "reconciled churn"})),
     churn.verify)

case("churn: only cancel, no note (partial)", False,
     lambda: post(BILL + "/api/cancel-billing", {"subscription_id": cs}),
     churn.verify)

case("churn: only note, no cancel (partial)", False,
     lambda: post(CRM + "/api/add-note", {"account_id": caid, "note": "x"}),
     churn.verify)

case("churn: cancel+note BUT also wrongly block (extra destructive)", False,
     lambda: (post(BILL + "/api/cancel-billing", {"subscription_id": cs}),
              post(CRM + "/api/add-note", {"account_id": caid, "note": "x"}),
              post(BILL + "/api/block-subscription", {"subscription_id": cs})),
     churn.verify)

case("churn: cancel+note BUT note on WRONG account (over-reach)", False,
     lambda: (post(BILL + "/api/cancel-billing", {"subscription_id": cs}),
              post(CRM + "/api/add-note", {"account_id": other.account_id, "note": "x"})),
     churn.verify)

case("churn: BLOCK instead of cancel + note (wrong action)", False,
     lambda: (post(BILL + "/api/block-subscription", {"subscription_id": cs}),
              post(CRM + "/api/add-note", {"account_id": caid, "note": "x"})),
     churn.verify)

# ---------- refund reconciliation (refund duplicate + CRM note) ----------
tag = refund.account_id.split("-")[1]
raid = refund.account_id
case("refund: refund dup + note (correct)", True,
     lambda: (post(BILL + "/api/refund", {"invoice_id": f"INV-{tag}B"}),
              post(CRM + "/api/add-note", {"account_id": raid, "note": "refunded dup"})),
     refund.verify)

case("refund: refund the OTHER dup member + note (also correct)", True,
     lambda: (post(BILL + "/api/refund", {"invoice_id": f"INV-{tag}A"}),
              post(CRM + "/api/add-note", {"account_id": raid, "note": "ok"})),
     refund.verify)

case("refund: refund NON-duplicate invoice C (wrong invoice)", False,
     lambda: (post(BILL + "/api/refund", {"invoice_id": f"INV-{tag}C"}),
              post(CRM + "/api/add-note", {"account_id": raid, "note": "ok"})),
     refund.verify)

case("refund: refund both dup invoices (too many)", False,
     lambda: (post(BILL + "/api/refund", {"invoice_id": f"INV-{tag}A"}),
              post(BILL + "/api/refund", {"invoice_id": f"INV-{tag}B"}),
              post(CRM + "/api/add-note", {"account_id": raid, "note": "ok"})),
     refund.verify)

case("refund: refund dup but no note (partial)", False,
     lambda: post(BILL + "/api/refund", {"invoice_id": f"INV-{tag}B"}),
     refund.verify)

# ---------- block policy (>=3 failed) ----------
bs = sid_of(block.account_id)
case("block: block correct sub", True,
     lambda: post(BILL + "/api/block-subscription", {"subscription_id": bs}),
     block.verify)

case("block: cancel billing instead of block (wrong action)", False,
     lambda: post(BILL + "/api/cancel-billing", {"subscription_id": bs}),
     block.verify)

case("block: block correct + also block another (over-reach)", False,
     lambda: (post(BILL + "/api/block-subscription", {"subscription_id": bs}),
              post(BILL + "/api/block-subscription", {"subscription_id": sid_of(other.account_id)})),
     block.verify)

# ---------- below policy / no-error: must do NOTHING ----------
case("below_policy: do nothing (correct)", True, lambda: None, below.verify)
case("below_policy: wrongly block (fail)", False,
     lambda: post(BILL + "/api/block-subscription", {"subscription_id": sid_of(below.account_id)}),
     below.verify)
case("below_policy: wrongly add a note (fail)", False,
     lambda: post(CRM + "/api/add-note", {"account_id": below.account_id, "note": "x"}),
     below.verify)
case("no_error: do nothing (correct)", True, lambda: None, noerr.verify)
case("no_error: wrongly refund (fail)", False,
     lambda: post(BILL + "/api/refund",
                  {"invoice_id": "INV-" + noerr.account_id.split("-")[1] + "A"}),
     noerr.verify)

reset()

# ---------- report answer checks (exact value) ----------
sb = get(BILL + "/api/snapshot")
ans_cases = [
    ("defaulters count exact", metrics.metric(sb, "defaulters_count"),
     f"There are {metrics.metric(sb,'defaulters_count')} past-due accounts.", True),
    ("defaulters count wrong", metrics.metric(sb, "defaulters_count"), "I think it is 999.", False),
    ("active MRR with $ and commas", metrics.metric(sb, "active_mrr"),
     f"Total MRR is ${metrics.metric(sb,'active_mrr'):,.2f}.", True),
]
for name, exp, text, want in ans_cases:
    r = check_answer(exp, text)
    results.append((r["success"] == want, "ANSWER " + name,
                    "PASS" if r["success"] else "FAIL",
                    "PASS" if want else "FAIL", r["reason"]))

# ---------- report ----------
print("\n================= VERIFIER TEST SUITE =================")
passed = 0
for ok, name, got, exp, reason in results:
    mark = "✅" if ok else "❌ MISMATCH"
    passed += ok
    print(f"  {mark}  {name}")
    if not ok:
        print(f"        got={got} expected={exp} :: {reason}")
print("------------------------------------------------------")
print(f"  {passed}/{len(results)} verifier cases behaved correctly")
print("======================================================")
sys.exit(0 if passed == len(results) else 1)
