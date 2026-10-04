# Support Console FaraEnv

A self-contained, **resettable, DB-verifiable** sandbox simulating a SaaS tier-1
support workspace — the environment layer of our Fara data pipeline
(modeled on FaraGen1.5's synthetic environments).

A tier-1 agent (human, GPT, or Fara) works real support flows:

- **Refund** an invoice
- **Cancel billing** for a subscription
- **Block** a subscription

All three are **irreversible → critical points**: the console forces an explicit
confirm step, and the agent is expected to pause for approval before confirming.

## Why this design
- **Full-stack control** (FastAPI + SQLite + HTML) → reset between rollouts and
  read ground truth from the DB.
- **Execution-based verification** (`verify.py`) → diff the DB snapshot before/after
  and check *exactly* the intended mutation happened and nothing else. This is the
  reward signal for SFT filtering and later GRPO.

## Run
```bash
cd support_env
pip install -r requirements.txt           # or reuse ../fara/.venv
python db.py                               # seed support.db
python -m uvicorn app:app --port 8000
open http://localhost:8000
```

## Point Fara at it
```bash
# with the RunPod tunnel up (localhost:5000) and the server on :8000
cd ..
./run_task.sh "Issue a refund for invoice INV-3002 for Olivia Martins and confirm." support_refund "http://localhost:8000/"
```

## Verify a rollout
```bash
curl -s localhost:8000/api/snapshot > before.json
# ... agent acts ...
curl -s localhost:8000/api/snapshot > after.json
python verify.py --task refund_single --before before.json --after after.json
```

## Files
| File | Purpose |
|---|---|
| `db.py` | schema + deterministic seed + `snapshot()` |
| `app.py` | FastAPI API + serves the console; `/api/reset`, `/api/snapshot` |
| `static/console.html` | the clickable console (critical-point confirm modal) |
| `tasks.jsonl` | seed support tasks w/ critical-point tags + success predicates |
| `verify.py` | execution-based verifier (DB diff) |

## Pipeline position
`[THIS: environment] → tasks → solver → trajectories → verify → SFT → GRPO`
