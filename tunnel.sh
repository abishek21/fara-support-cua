#!/bin/bash
# Opens an SSH tunnel from local :5000 to the RunPod pod's vLLM server (:5000).
# Keep this terminal open while running fara-cli. Ctrl+C to close the tunnel.
set -e
POD_HOST="root@69.30.85.117"
POD_PORT="22162"
KEY="$HOME/.ssh/id_ed25519"

echo "Opening SSH tunnel: localhost:5000 -> pod:5000  (Ctrl+C to stop)"
ssh -N -L 5000:localhost:5000 -p "$POD_PORT" -i "$KEY" "$POD_HOST"
