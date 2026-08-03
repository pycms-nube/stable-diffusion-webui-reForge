#!/usr/bin/env bash
#################################################################
# Combined backend+frontend launcher (BFISO Phase 20)
#
# Single-command entry point for the two-process split: starts
# webui-backend.sh, waits for it to report healthy, then starts
# webui-frontend.sh against it. Ctrl-C (or any exit) tears both down.
#
# This does NOT replace webui.sh -- that script is still the
# single-process, full-feature launch (every tab: Settings, Extras, PNG
# Info, Train, every extension's own UI). This is the one-command way to
# run the split frontend/backend pair proven in PHASE8-19.md, whose
# feature set is still txt2img + img2img only. See BFISO.md's Phase 20
# entry for the current scope of "replacing" webui.sh as the primary
# launch path.
#
# Env vars (same names webui-backend.sh / webui-frontend.sh already use):
#   BACKEND_VENV_DIR, BACKEND_PORT, BACKEND_CKPT   -- forwarded to webui-backend.sh
#   FRONTEND_VENV_DIR, FRONTEND_PORT               -- forwarded to webui-frontend.sh
#   FRONTEND_BACKEND_URL   -- defaults to http://127.0.0.1:${BACKEND_PORT}, so it
#                             tracks a custom BACKEND_PORT automatically unless you
#                             override it yourself
#   BACKEND_HEALTH_TIMEOUT -- seconds to wait for the backend before giving up
#                             (default 120 -- model loading can be slow)
#
# Extra args passed to this script are forwarded to webui-frontend.sh (and from
# there to webui_frontend.py), same convention as the two individual scripts.
#################################################################

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

BACKEND_PORT="${BACKEND_PORT:-7860}"
FRONTEND_PORT="${FRONTEND_PORT:-7870}"
FRONTEND_BACKEND_URL="${FRONTEND_BACKEND_URL:-http://127.0.0.1:${BACKEND_PORT}}"
BACKEND_HEALTH_TIMEOUT="${BACKEND_HEALTH_TIMEOUT:-120}"
export FRONTEND_BACKEND_URL FRONTEND_PORT

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    trap - EXIT INT TERM
    echo
    echo "Shutting down..."
    if [[ -n "$FRONTEND_PID" ]]; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    if [[ -n "$BACKEND_PID" ]]; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "=== Starting backend (port ${BACKEND_PORT}) ==="
bash webui-backend.sh &
BACKEND_PID=$!

echo "Waiting for the backend to report healthy (timeout ${BACKEND_HEALTH_TIMEOUT}s)..."
elapsed=0
until curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${BACKEND_PORT}/sdapi/v1/samplers" \
        --max-time 2 2>/dev/null | grep -q 200; do
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "ERROR: backend process exited before becoming healthy -- check its output above." >&2
        exit 1
    fi
    if (( elapsed >= BACKEND_HEALTH_TIMEOUT )); then
        echo "ERROR: backend did not become healthy within ${BACKEND_HEALTH_TIMEOUT}s (set BACKEND_HEALTH_TIMEOUT to wait longer)." >&2
        exit 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
done
echo "Backend is up."

echo "=== Starting frontend (port ${FRONTEND_PORT}, backend ${FRONTEND_BACKEND_URL}) ==="
bash webui-frontend.sh "$@" &
FRONTEND_PID=$!

wait "$FRONTEND_PID"
