"""
Database layer for the Support Console FaraEnv (v2 — hard mode).

Changes vs v1:
- 100+ procedurally generated customers (distractors) so the agent must
  SEARCH / SCROLL to find the right account — raising navigation difficulty.
- A handful of PLANTED scenario customers with known ground truth, used by tasks.
- Consistent billing: a subscription's `failed_payments` equals the number of
  recent invoices with status='failed' (no more "3 failed but all paid" bug).

Invoice status: paid | refunded | failed.
Every reset reseeds to an identical ground-truth state (sandboxed + resettable,
mirroring FaraGen1.5 synthetic environments).
"""
from __future__ import annotations

import os
import random
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "support.db")
SEED = int(os.environ.get("DB_SEED", "20261004"))  # override to reshuffle everything

SCHEMA = """
DROP TABLE IF EXISTS refunds;
DROP TABLE IF EXISTS audit_log;
DROP TABLE IF EXISTS invoices;
DROP TABLE IF EXISTS subscriptions;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    id        TEXT PRIMARY KEY,
    name      TEXT NOT NULL,
    email     TEXT NOT NULL,
    status    TEXT NOT NULL DEFAULT 'active'    -- active | suspended
);

CREATE TABLE subscriptions (
    id              TEXT PRIMARY KEY,
    customer_id     TEXT NOT NULL REFERENCES customers(id),
    plan            TEXT NOT NULL,
    price           REAL NOT NULL,
    status          TEXT NOT NULL DEFAULT 'active', -- active | cancelled | blocked
    billing         TEXT NOT NULL DEFAULT 'on',     -- on | off
    failed_payments INTEGER NOT NULL DEFAULT 0      -- = count of 'failed' invoices
);

CREATE TABLE invoices (
    id               TEXT PRIMARY KEY,
    customer_id      TEXT NOT NULL REFERENCES customers(id),
    subscription_id  TEXT NOT NULL REFERENCES subscriptions(id),
    amount           REAL NOT NULL,
    status           TEXT NOT NULL DEFAULT 'paid', -- paid | refunded | failed
    issued_on        TEXT NOT NULL,
    period           TEXT NOT NULL,
    description      TEXT NOT NULL
);

CREATE TABLE refunds (
    id           TEXT PRIMARY KEY,
    invoice_id   TEXT NOT NULL REFERENCES invoices(id),
    amount       REAL NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE TABLE audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    action      TEXT NOT NULL,
    target      TEXT NOT NULL,
    detail      TEXT,
    created_at  TEXT NOT NULL
);
"""

PLANS = [("Starter Monthly", 19.0), ("Pro Monthly", 49.0),
         ("Team Monthly", 99.0), ("Business Annual", 990.0)]

FIRST = ["Olivia", "Raj", "Mei", "Liam", "Nina", "Carlos", "Aisha", "Tom",
         "Sofia", "Hiro", "Elena", "Noah", "Priya", "Lucas", "Fatima", "Jack",
         "Yuki", "Omar", "Clara", "Ben", "Ava", "Diego", "Sara", "Ivan",
         "Lena", "Max", "Zara", "Leo", "Nora", "Sam"]
LAST = ["Martins", "Patel", "Chen", "O'Connor", "Alvarez", "Rossi", "Khan",
        "Nguyen", "Garcia", "Tanaka", "Novak", "Silva", "Haddad", "Meyer",
        "Costa", "Dubois", "Park", "Ferreira", "Walsh", "Ibrahim", "Kowalski",
        "Reyes", "Andersen", "Blum", "Fischer", "Moreau", "Lopez", "Singh"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


# --------------------------------------------------------------------------
# Data generation
# --------------------------------------------------------------------------
def _build_dataset():
    """Return (customers, subscriptions, invoices) as lists of tuples.

    Planted scenario accounts have FIXED ids/names so tasks + verifier agree.
    The rest are random distractors.
    """
    rng = random.Random(SEED)
    customers, subs, invoices = [], [], []
    roles = {}  # scenario -> human label, for transparency

    DOMAINS = ["contoso", "fabrikam", "northwind", "adventure"]
    used_names: set[str] = set()

    def rand_name() -> str:
        while True:
            n = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
            if n not in used_names:
                used_names.add(n)
                return n

    def email_for(name: str) -> str:
        return (name.lower().replace(" ", ".").replace("'", "")
                + f"@{rng.choice(DOMAINS)}.com")

    def add_customer(cid: str, name: str):
        customers.append((cid, name, email_for(name), "active"))

    # ---- PLANTED scenario accounts (random NAMES, fixed roles/IDs) -------
    # The *person* for each scenario is randomized per seed so the model cannot
    # rely on a name prior (e.g. "Olivia = duplicate"). Ground truth is the ROLE,
    # which task generators read from the DB.

    # R1 — genuine duplicate charge -> refund one of the pair
    n1 = rand_name(); roles["duplicate_refund"] = f"{n1} (CUST-1001)"
    add_customer("CUST-1001", n1)
    subs.append(["SUB-1001", "CUST-1001", "Pro Monthly", 49.0, "active", "on", 0])
    invoices += [
        ("INV-1001A", "CUST-1001", "SUB-1001", 49.0, "paid", "2026-09-01", "Sep 2026", "Pro Monthly - Sep 2026"),
        ("INV-1001B", "CUST-1001", "SUB-1001", 49.0, "paid", "2026-09-01", "Sep 2026", "Pro Monthly - Sep 2026 (duplicate)"),
        ("INV-1001C", "CUST-1001", "SUB-1001", 49.0, "paid", "2026-08-01", "Aug 2026", "Pro Monthly - Aug 2026"),
    ]

    # R2 — normal account; will request billing cancellation
    n2 = rand_name(); roles["cancel_request"] = f"{n2} (CUST-1002)"
    add_customer("CUST-1002", n2)
    subs.append(["SUB-1002", "CUST-1002", "Business Annual", 990.0, "active", "on", 0])
    invoices += [
        ("INV-1002A", "CUST-1002", "SUB-1002", 990.0, "paid", "2026-01-15", "2026 (annual)", "Business Annual - 2026"),
    ]

    # R3 — single legitimate charge; refund NOT warranted
    n3 = rand_name(); roles["no_refund"] = f"{n3} (CUST-1003)"
    add_customer("CUST-1003", n3)
    subs.append(["SUB-1003", "CUST-1003", "Starter Monthly", 19.0, "active", "on", 0])
    invoices += [
        ("INV-1003A", "CUST-1003", "SUB-1003", 19.0, "paid", "2026-09-10", "Sep 2026", "Starter Monthly - Sep 2026"),
    ]

    # G1 — disambiguation collision group: THREE customers sharing a RANDOM first
    # name; exactly ONE qualifies for block (>=3 failed). Qualifier chosen randomly.
    coll_first = rng.choice(FIRST)
    coll_lasts = rng.sample(LAST, 3)
    coll_ids = ["CUST-1004", "CUST-1201", "CUST-1202"]
    coll_subs = ["SUB-1004", "SUB-1201", "SUB-1202"]
    coll_qual = rng.randrange(3)
    for k in range(3):
        name = f"{coll_first} {coll_lasts[k]}"
        used_names.add(name)
        cid, sid = coll_ids[k], coll_subs[k]
        add_customer(cid, name)
        tag = cid.split("-")[1]
        if k == coll_qual:
            subs.append([sid, cid, "Pro Monthly", 49.0, "active", "on", 3])
            invoices += [
                (f"INV-{tag}A", cid, sid, 49.0, "paid",   "2026-06-05", "Jun 2026", "Pro Monthly - Jun 2026"),
                (f"INV-{tag}B", cid, sid, 49.0, "failed", "2026-07-05", "Jul 2026", "Pro Monthly - Jul 2026 (payment failed)"),
                (f"INV-{tag}C", cid, sid, 49.0, "failed", "2026-08-05", "Aug 2026", "Pro Monthly - Aug 2026 (payment failed)"),
                (f"INV-{tag}D", cid, sid, 49.0, "failed", "2026-09-05", "Sep 2026", "Pro Monthly - Sep 2026 (payment failed)"),
            ]
        else:
            subs.append([sid, cid, "Pro Monthly", 49.0, "active", "on", 0])
            invoices += [
                (f"INV-{tag}A", cid, sid, 49.0, "paid", "2026-09-05", "Sep 2026", "Pro Monthly - Sep 2026"),
            ]
    roles["disambiguate_block"] = f"{coll_first} {coll_lasts[coll_qual]} ({coll_ids[coll_qual]})"
    globals()["QUALIFYING_LIAM"] = roles["disambiguate_block"]  # back-compat name

    # R4 — only 2 FAILED payments -> policy NOT met, do NOT block (edge negative)
    n5 = rand_name(); roles["below_policy_no_block"] = f"{n5} (CUST-1005)"
    add_customer("CUST-1005", n5)
    subs.append(["SUB-1005", "CUST-1005", "Team Monthly", 99.0, "active", "on", 2])
    invoices += [
        ("INV-1005A", "CUST-1005", "SUB-1005", 99.0, "paid",   "2026-07-20", "Jul 2026", "Team Monthly - Jul 2026"),
        ("INV-1005B", "CUST-1005", "SUB-1005", 99.0, "failed", "2026-08-20", "Aug 2026", "Team Monthly - Aug 2026 (payment failed)"),
        ("INV-1005C", "CUST-1005", "SUB-1005", 99.0, "failed", "2026-09-20", "Sep 2026", "Team Monthly - Sep 2026 (payment failed)"),
    ]

    # R5 — 3 FAILED payments -> qualifies for block (extra L6 target)
    n6 = rand_name(); roles["block_extra_a"] = f"{n6} (CUST-1006)"
    add_customer("CUST-1006", n6)
    subs.append(["SUB-1006", "CUST-1006", "Starter Monthly", 19.0, "active", "on", 3])
    invoices += [
        ("INV-1006A", "CUST-1006", "SUB-1006", 19.0, "failed", "2026-07-12", "Jul 2026", "Starter Monthly - Jul 2026 (payment failed)"),
        ("INV-1006B", "CUST-1006", "SUB-1006", 19.0, "failed", "2026-08-12", "Aug 2026", "Starter Monthly - Aug 2026 (payment failed)"),
        ("INV-1006C", "CUST-1006", "SUB-1006", 19.0, "failed", "2026-09-12", "Sep 2026", "Starter Monthly - Sep 2026 (payment failed)"),
    ]

    # R6 — 4 FAILED payments -> qualifies for block (extra L6 target)
    n7 = rand_name(); roles["block_extra_b"] = f"{n7} (CUST-1007)"
    add_customer("CUST-1007", n7)
    subs.append(["SUB-1007", "CUST-1007", "Business Annual", 990.0, "active", "on", 4])
    invoices += [
        ("INV-1007A", "CUST-1007", "SUB-1007", 990.0, "failed", "2026-06-01", "Jun 2026", "Business Annual - Jun 2026 (payment failed)"),
        ("INV-1007B", "CUST-1007", "SUB-1007", 990.0, "failed", "2026-07-01", "Jul 2026", "Business Annual - Jul 2026 (payment failed)"),
        ("INV-1007C", "CUST-1007", "SUB-1007", 990.0, "failed", "2026-08-01", "Aug 2026", "Business Annual - Aug 2026 (payment failed)"),
        ("INV-1007D", "CUST-1007", "SUB-1007", 990.0, "failed", "2026-09-01", "Sep 2026", "Business Annual - Sep 2026 (payment failed)"),
    ]
    globals()["ROLES"] = roles

    planted_names = {c[1] for c in customers}

    # ---- RANDOM distractor accounts --------------------------------------
    used = set(planted_names)
    idx = 1008
    target_total = 110  # ~103 distractors + 7 planted
    while len(customers) < target_total:
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if name in used:
            continue
        used.add(name)
        cid = f"CUST-{idx}"
        sid = f"SUB-{idx}"
        plan, price = rng.choice(PLANS)
        email = (name.lower().replace(" ", ".").replace("'", "")
                 + f"@{rng.choice(['contoso', 'fabrikam', 'northwind', 'adventure'])}.com")
        customers.append((cid, name, email, "active"))
        # a few distractors have 1 failed payment (noise, below the block policy)
        nfail = rng.choices([0, 0, 0, 0, 1], k=1)[0]
        subs.append([sid, cid, plan, price, "active", "on", nfail])
        n_inv = rng.randint(1, 3)
        for k in range(n_inv):
            month = 9 - k
            status = "failed" if (nfail and k == 0) else "paid"
            invoices.append((
                f"INV-{idx}{chr(65 + k)}", cid, sid, price, status,
                f"2026-{month:02d}-10", f"{month:02d} 2026",
                f"{plan} - 2026-{month:02d}" + (" (payment failed)" if status == "failed" else ""),
            ))
        idx += 1

    subs = [tuple(s) for s in subs]
    return customers, subs, invoices


def seed_db() -> None:
    customers, subs, invoices = _build_dataset()
    conn = connect()
    try:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO customers (id,name,email,status) VALUES (?,?,?,?)", customers)
        conn.executemany(
            "INSERT INTO subscriptions (id,customer_id,plan,price,status,billing,"
            "failed_payments) VALUES (?,?,?,?,?,?,?)", subs)
        conn.executemany(
            "INSERT INTO invoices (id,customer_id,subscription_id,amount,status,"
            "issued_on,period,description) VALUES (?,?,?,?,?,?,?,?)", invoices)
        conn.commit()
    finally:
        conn.close()


def snapshot() -> dict:
    conn = connect()
    try:
        out = {}
        for table in ("customers", "subscriptions", "invoices", "refunds", "audit_log"):
            rows = conn.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            out[table] = [dict(r) for r in rows]
        return out
    finally:
        conn.close()


if __name__ == "__main__":
    seed_db()
    s = snapshot()
    print(f"Seeded {DB_PATH} (SEED={SEED}): {len(s['customers'])} customers, "
          f"{len(s['subscriptions'])} subscriptions, {len(s['invoices'])} invoices")
    print("  Randomized scenario roles:")
    for role, who in (globals().get("ROLES") or {}).items():
        print(f"    {role:<22} -> {who}")
