"""
Support Console FaraEnv — FastAPI backend + static console.

Run:
    cd support_env
    pip install -r requirements.txt
    python -m uvicorn app:app --port 8000
    open http://localhost:8000

Endpoints
---------
GET  /                         -> the clickable support console (HTML)
GET  /api/customers            -> list customers
GET  /api/customers/{id}       -> customer detail (subscriptions + invoices)
POST /api/refund               -> {invoice_id}             (irreversible: CRITICAL POINT)
POST /api/cancel-billing       -> {subscription_id}        (irreversible: CRITICAL POINT)
POST /api/block-subscription   -> {subscription_id}        (irreversible: CRITICAL POINT)
POST /api/reset                -> reseed DB to canonical state
GET  /api/snapshot             -> full DB state (used by verifier)

The three mutating endpoints are the "critical points": the UI forces an
explicit confirm step before calling them, and the agent is expected to stop
and ask for approval before confirming.
"""
from __future__ import annotations

import os
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import db

app = FastAPI(title="Support Console FaraEnv")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Reseed on startup so the environment always begins in a known state.
if not os.path.exists(db.DB_PATH):
    db.seed_db()


# ----------------------------- request models -----------------------------
class RefundReq(BaseModel):
    invoice_id: str


class SubReq(BaseModel):
    subscription_id: str


# ------------------------------- read APIs --------------------------------
@app.get("/api/customers")
def list_customers():
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT id, name, email, status FROM customers ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.get("/api/search")
def search_customers(q: str = ""):
    """Search customers by name / email / id (case-insensitive substring)."""
    q = (q or "").strip().lower()
    conn = db.connect()
    try:
        if not q:
            return []
        like = f"%{q}%"
        rows = conn.execute(
            "SELECT id, name, email, status FROM customers "
            "WHERE lower(name) LIKE ? OR lower(email) LIKE ? OR lower(id) LIKE ? "
            "ORDER BY RANDOM() LIMIT 50",
            (like, like, like),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.get("/api/customers/{customer_id}")
def customer_detail(customer_id: str):
    conn = db.connect()
    try:
        cust = conn.execute(
            "SELECT * FROM customers WHERE id = ?", (customer_id,)
        ).fetchone()
        if not cust:
            raise HTTPException(404, "customer not found")
        subs = conn.execute(
            "SELECT * FROM subscriptions WHERE customer_id = ? ORDER BY id",
            (customer_id,),
        ).fetchall()
        invs = conn.execute(
            "SELECT * FROM invoices WHERE customer_id = ? ORDER BY id",
            (customer_id,),
        ).fetchall()
        return {
            "customer": dict(cust),
            "subscriptions": [dict(s) for s in subs],
            "invoices": [dict(i) for i in invs],
        }
    finally:
        conn.close()


# ------------------------- mutating APIs (critical points) ----------------
@app.post("/api/refund")
def refund(req: RefundReq):
    conn = db.connect()
    try:
        inv = conn.execute(
            "SELECT * FROM invoices WHERE id = ?", (req.invoice_id,)
        ).fetchone()
        if not inv:
            raise HTTPException(404, "invoice not found")
        if inv["status"] == "refunded":
            raise HTTPException(409, "invoice already refunded")
        conn.execute(
            "UPDATE invoices SET status = 'refunded' WHERE id = ?", (req.invoice_id,)
        )
        conn.execute(
            "INSERT INTO refunds (id, invoice_id, amount, created_at) VALUES (?,?,?,?)",
            (f"REF-{uuid.uuid4().hex[:8]}", req.invoice_id, inv["amount"], db.now()),
        )
        conn.execute(
            "INSERT INTO audit_log (action, target, detail, created_at) VALUES (?,?,?,?)",
            ("refund", req.invoice_id, f"amount={inv['amount']}", db.now()),
        )
        conn.commit()
        return {"ok": True, "invoice_id": req.invoice_id, "amount": inv["amount"]}
    finally:
        conn.close()


@app.post("/api/cancel-billing")
def cancel_billing(req: SubReq):
    conn = db.connect()
    try:
        sub = conn.execute(
            "SELECT * FROM subscriptions WHERE id = ?", (req.subscription_id,)
        ).fetchone()
        if not sub:
            raise HTTPException(404, "subscription not found")
        conn.execute(
            "UPDATE subscriptions SET billing = 'off' WHERE id = ?",
            (req.subscription_id,),
        )
        conn.execute(
            "INSERT INTO audit_log (action, target, detail, created_at) VALUES (?,?,?,?)",
            ("cancel_billing", req.subscription_id, "billing=off", db.now()),
        )
        conn.commit()
        return {"ok": True, "subscription_id": req.subscription_id, "billing": "off"}
    finally:
        conn.close()


@app.post("/api/block-subscription")
def block_subscription(req: SubReq):
    conn = db.connect()
    try:
        sub = conn.execute(
            "SELECT * FROM subscriptions WHERE id = ?", (req.subscription_id,)
        ).fetchone()
        if not sub:
            raise HTTPException(404, "subscription not found")
        conn.execute(
            "UPDATE subscriptions SET status = 'blocked' WHERE id = ?",
            (req.subscription_id,),
        )
        conn.execute(
            "INSERT INTO audit_log (action, target, detail, created_at) VALUES (?,?,?,?)",
            ("block_subscription", req.subscription_id, "status=blocked", db.now()),
        )
        conn.commit()
        return {"ok": True, "subscription_id": req.subscription_id, "status": "blocked"}
    finally:
        conn.close()


# ------------------------------ control APIs ------------------------------
@app.post("/api/reset")
def reset():
    db.seed_db()
    return {"ok": True, "message": "database reset to canonical seed state"}


@app.get("/api/snapshot")
def get_snapshot():
    return db.snapshot()


# ------------------------------- the console ------------------------------
@app.get("/")
def console():
    return FileResponse(os.path.join(STATIC_DIR, "console.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
