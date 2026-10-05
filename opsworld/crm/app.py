"""
Contoso CRM — system #1 of the OpsWorld sandbox (port 8001).

Holds the customer relationship view: account, status (active/churned/suspended),
plan, and CASE NOTES. Actions: add a case note, update account status.
Seeds crm.db from the shared world (common/world.py).
"""
from __future__ import annotations

import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import dbutil
from common.world import generate_world

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "crm.db")
STATIC = os.path.join(HERE, "static")
TABLES = ["accounts", "cases", "audit_log"]

SCHEMA = """
DROP TABLE IF EXISTS audit_log;
DROP TABLE IF EXISTS cases;
DROP TABLE IF EXISTS accounts;
CREATE TABLE accounts (
    account_id TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    email      TEXT NOT NULL,
    plan       TEXT NOT NULL,
    status     TEXT NOT NULL        -- active | churned | suspended
);
CREATE TABLE cases (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id  TEXT NOT NULL,
    note        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT, target TEXT, detail TEXT, created_at TEXT
);
"""

app = FastAPI(title="Contoso CRM")


def seed():
    accounts, _ = generate_world()
    conn = dbutil.connect(DB)
    try:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO accounts (account_id,name,email,plan,status) VALUES (?,?,?,?,?)",
            [(a.account_id, a.name, a.email, a.plan, a.crm_status)
             for a in accounts.values()],
        )
        conn.commit()
    finally:
        conn.close()


if not os.path.exists(DB):
    seed()


class NoteReq(BaseModel):
    account_id: str
    note: str


class StatusReq(BaseModel):
    account_id: str
    status: str   # active | churned | suspended


@app.get("/api/search")
def search(q: str = ""):
    q = (q or "").strip().lower()
    if not q:
        return []
    conn = dbutil.connect(DB)
    try:
        like = f"%{q}%"
        rows = conn.execute(
            "SELECT account_id, name, email, status FROM accounts "
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
        cases = conn.execute(
            "SELECT * FROM cases WHERE account_id=? ORDER BY id", (account_id,)).fetchall()
        return {"account": dict(a), "cases": [dict(c) for c in cases]}
    finally:
        conn.close()


@app.post("/api/add-note")
def add_note(req: NoteReq):
    conn = dbutil.connect(DB)
    try:
        a = conn.execute("SELECT 1 FROM accounts WHERE account_id=?", (req.account_id,)).fetchone()
        if not a:
            raise HTTPException(404, "account not found")
        conn.execute("INSERT INTO cases (account_id,note,created_at) VALUES (?,?,?)",
                     (req.account_id, req.note, dbutil.now()))
        conn.execute("INSERT INTO audit_log (action,target,detail,created_at) VALUES (?,?,?,?)",
                     ("add_note", req.account_id, req.note[:120], dbutil.now()))
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


@app.post("/api/update-status")
def update_status(req: StatusReq):
    if req.status not in ("active", "churned", "suspended"):
        raise HTTPException(400, "invalid status")
    conn = dbutil.connect(DB)
    try:
        a = conn.execute("SELECT 1 FROM accounts WHERE account_id=?", (req.account_id,)).fetchone()
        if not a:
            raise HTTPException(404, "account not found")
        conn.execute("UPDATE accounts SET status=? WHERE account_id=?",
                     (req.status, req.account_id))
        conn.execute("INSERT INTO audit_log (action,target,detail,created_at) VALUES (?,?,?,?)",
                     ("update_status", req.account_id, req.status, dbutil.now()))
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


@app.post("/api/reset")
def reset():
    seed()
    return {"ok": True}


@app.get("/api/snapshot")
def snap():
    return dbutil.snapshot(DB, TABLES)


@app.get("/")
def console():
    return FileResponse(os.path.join(STATIC, "crm.html"))
