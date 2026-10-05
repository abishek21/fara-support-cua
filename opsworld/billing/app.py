"""
Contoso Billing — system #2 of the OpsWorld sandbox (port 8002).

Holds invoices, the subscription (billing on/off, status, failed payments), and
refunds. Actions: refund an invoice, cancel billing, block subscription.
Seeds billing.db from the shared world (common/world.py).
"""
from __future__ import annotations

import os
import sys
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import dbutil
from common import metrics
from common.world import generate_world

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "billing.db")
STATIC = os.path.join(HERE, "static")
TABLES = ["accounts", "subscriptions", "invoices", "refunds", "audit_log"]

SCHEMA = """
DROP TABLE IF EXISTS audit_log;
DROP TABLE IF EXISTS refunds;
DROP TABLE IF EXISTS invoices;
DROP TABLE IF EXISTS subscriptions;
DROP TABLE IF EXISTS accounts;
CREATE TABLE accounts (
    account_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL
);
CREATE TABLE subscriptions (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    plan TEXT NOT NULL,
    price REAL NOT NULL,
    status TEXT NOT NULL,          -- active | blocked
    billing TEXT NOT NULL,         -- on | off
    failed_payments INTEGER NOT NULL
);
CREATE TABLE invoices (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    subscription_id TEXT NOT NULL,
    amount REAL NOT NULL,
    status TEXT NOT NULL,          -- paid | refunded | failed
    period TEXT NOT NULL
);
CREATE TABLE refunds (
    id TEXT PRIMARY KEY, invoice_id TEXT, amount REAL, created_at TEXT
);
CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT, target TEXT, detail TEXT, created_at TEXT
);
"""

app = FastAPI(title="Contoso Billing")


def _sub_id(acc: str) -> str:
    return "SUB-" + acc.split("-")[1]


def seed():
    accounts, _ = generate_world()
    conn = dbutil.connect(DB)
    try:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO accounts (account_id,name,email) VALUES (?,?,?)",
                         [(a.account_id, a.name, a.email) for a in accounts.values()])
        subs, invs = [], []
        for a in accounts.values():
            sid = _sub_id(a.account_id)
            subs.append((sid, a.account_id, a.plan, a.price, "active",
                         "on" if a.billing_on else "off", a.failed_payments))
            for inv in a.invoices:
                invs.append((inv.id, a.account_id, sid, inv.amount, inv.status, inv.period))
        conn.executemany(
            "INSERT INTO subscriptions (id,account_id,plan,price,status,billing,failed_payments)"
            " VALUES (?,?,?,?,?,?,?)", subs)
        conn.executemany(
            "INSERT INTO invoices (id,account_id,subscription_id,amount,status,period)"
            " VALUES (?,?,?,?,?,?)", invs)
        conn.commit()
    finally:
        conn.close()


if not os.path.exists(DB):
    seed()


class RefundReq(BaseModel):
    invoice_id: str


class SubReq(BaseModel):
    subscription_id: str


@app.get("/api/search")
def search(q: str = ""):
    q = (q or "").strip().lower()
    if not q:
        return []
    conn = dbutil.connect(DB)
    try:
        like = f"%{q}%"
        rows = conn.execute(
            "SELECT account_id, name, email FROM accounts "
            "WHERE lower(name) LIKE ? OR lower(email) LIKE ? OR lower(account_id) LIKE ? "
            "ORDER BY RANDOM() LIMIT 50", (like, like, like)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.get("/api/accounts/{account_id}")
def account(account_id: str):
    conn = dbutil.connect(DB)
    try:
        a = conn.execute("SELECT * FROM accounts WHERE account_id=?", (account_id,)).fetchone()
        if not a:
            raise HTTPException(404, "account not found")
        subs = conn.execute("SELECT * FROM subscriptions WHERE account_id=?",
                            (account_id,)).fetchall()
        invs = conn.execute("SELECT * FROM invoices WHERE account_id=? ORDER BY id",
                            (account_id,)).fetchall()
        return {"account": dict(a), "subscriptions": [dict(s) for s in subs],
                "invoices": [dict(i) for i in invs]}
    finally:
        conn.close()


@app.post("/api/refund")
def refund(req: RefundReq):
    conn = dbutil.connect(DB)
    try:
        inv = conn.execute("SELECT * FROM invoices WHERE id=?", (req.invoice_id,)).fetchone()
        if not inv:
            raise HTTPException(404, "invoice not found")
        if inv["status"] == "refunded":
            raise HTTPException(409, "already refunded")
        conn.execute("UPDATE invoices SET status='refunded' WHERE id=?", (req.invoice_id,))
        conn.execute("INSERT INTO refunds (id,invoice_id,amount,created_at) VALUES (?,?,?,?)",
                     (f"REF-{uuid.uuid4().hex[:8]}", req.invoice_id, inv["amount"], dbutil.now()))
        conn.execute("INSERT INTO audit_log (action,target,detail,created_at) VALUES (?,?,?,?)",
                     ("refund", req.invoice_id, f"amount={inv['amount']}", dbutil.now()))
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


@app.post("/api/cancel-billing")
def cancel_billing(req: SubReq):
    conn = dbutil.connect(DB)
    try:
        s = conn.execute("SELECT 1 FROM subscriptions WHERE id=?", (req.subscription_id,)).fetchone()
        if not s:
            raise HTTPException(404, "subscription not found")
        conn.execute("UPDATE subscriptions SET billing='off' WHERE id=?", (req.subscription_id,))
        conn.execute("INSERT INTO audit_log (action,target,detail,created_at) VALUES (?,?,?,?)",
                     ("cancel_billing", req.subscription_id, "billing=off", dbutil.now()))
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


@app.post("/api/block-subscription")
def block(req: SubReq):
    conn = dbutil.connect(DB)
    try:
        s = conn.execute("SELECT 1 FROM subscriptions WHERE id=?", (req.subscription_id,)).fetchone()
        if not s:
            raise HTTPException(404, "subscription not found")
        conn.execute("UPDATE subscriptions SET status='blocked' WHERE id=?", (req.subscription_id,))
        conn.execute("INSERT INTO audit_log (action,target,detail,created_at) VALUES (?,?,?,?)",
                     ("block_subscription", req.subscription_id, "status=blocked", dbutil.now()))
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


@app.post("/api/reset")
def reset():
    seed()
    return {"ok": True}


@app.get("/api/report")
def report(status: str = "all", billing: str = "all",
           min_failed: int = 0, plan: str = "all"):
    """Filterable subscriptions report + live aggregates (count, MRR)."""
    snap = dbutil.snapshot(DB, TABLES)
    rows = metrics.filter_subscriptions(
        snap, status=status, billing=billing, min_failed=min_failed, plan=plan)
    return {"rows": rows, "aggregates": metrics.aggregates(rows)}


@app.get("/api/snapshot")
def snap():
    return dbutil.snapshot(DB, TABLES)


@app.get("/reports")
def reports_page():
    return FileResponse(os.path.join(STATIC, "reports.html"))


@app.get("/")
def console():
    return FileResponse(os.path.join(STATIC, "billing.html"))
