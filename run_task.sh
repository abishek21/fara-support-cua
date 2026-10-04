#!/bin/bash
# Run the Fara-7B agent against the tunneled RunPod vLLM endpoint.
# Prereq: ./tunnel.sh running in another terminal (localhost:5000 -> pod).
# Usage: ./run_task.sh "your task here" [output_subfolder] [start_page_url]
set -e
cd "$(dirname "$0")"
source fara/.venv/bin/activate

TASK="${1:-how many pages does wikipedia have}"
OUT="runs/${2:-$(date +%Y%m%d_%H%M%S)}"
START_PAGE="${3:-}"
mkdir -p "$OUT"

echo "Task: $TASK"
echo "Output: $OUT"
[ -n "$START_PAGE" ] && echo "Start page: $START_PAGE"

# Optional: drop Fara directly on a site to avoid the Bing search gate.
START_ARG=()
[ -n "$START_PAGE" ] && START_ARG=(--start_page "$START_PAGE")

# Feed empty stdin so the interactive loop exits after the task completes.
# PYTHONUNBUFFERED=1 keeps output unbuffered so you see thoughts LIVE *and*
# they get captured to console.log even if you Ctrl+C mid-run.
printf '\n\n' | PYTHONUNBUFFERED=1 fara-cli \
  --fara-7b \
  --task "$TASK" \
  --endpoint_config runpod_config.json \
  --headful \
  --save_screenshots \
  --max_rounds 20 \
  "${START_ARG[@]}" \
  --output_folder "$OUT" 2>&1 | tee "$OUT/console.log"

echo "Done. Trajectory saved to $OUT"
python view_trajectory.py "$OUT" && open "$OUT/trajectory.html"
