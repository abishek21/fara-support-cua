#!/bin/bash
# Launch the OpsWorld multi-system sandbox: CRM (:8001) + Billing (:8002).
# Both seed from the same world (common/world.py); set DB_SEED to reshuffle.
set -e
cd "$(dirname "$0")"
VENV="${VENV:-../fara/.venv/bin/activate}"
source "$VENV"

export DB_SEED="${DB_SEED:-20261005}"

echo "Seeding + launching OpsWorld (DB_SEED=$DB_SEED)..."
# fresh seed so a changed DB_SEED takes effect
rm -f crm/crm.db billing/billing.db
python -c "import sys; sys.path.insert(0,'.'); from crm.app import seed; seed()"
python -c "import sys; sys.path.insert(0,'.'); from billing.app import seed; seed()"

pkill -f "uvicorn crm.app:app" 2>/dev/null || true
pkill -f "uvicorn billing.app:app" 2>/dev/null || true
sleep 1
nohup python -m uvicorn crm.app:app --port 8001 > crm/server.log 2>&1 &
nohup python -m uvicorn billing.app:app --port 8002 > billing/server.log 2>&1 &
sleep 3
echo "CRM:     http://localhost:8001  ($(curl -s -o /dev/null -w '%{http_code}' localhost:8001/api/search?q=a))"
echo "Billing: http://localhost:8002  ($(curl -s -o /dev/null -w '%{http_code}' localhost:8002/api/search?q=a))"
echo "Inspect the world:  python inspect_world.py"
