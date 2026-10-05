"""Shared SQLite helpers for OpsWorld system apps."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def snapshot(db_path: str, tables: list[str]) -> dict:
    conn = connect(db_path)
    try:
        out = {}
        for t in tables:
            rows = conn.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall()
            out[t] = [dict(r) for r in rows]
        return out
    finally:
        conn.close()


def ensure_dir(db_path: str) -> None:
    d = os.path.dirname(db_path)
    if d and not os.path.exists(d):
        os.makedirs(d)
