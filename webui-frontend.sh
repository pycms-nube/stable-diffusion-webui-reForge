#!/usr/bin/env bash
#################################################################
# Standalone frontend launcher (BFISO Phase 8)
#
# Launches modules_frontend/txt2img_ui.py via webui_frontend.py -- a Gradio app
# that imports no torch/ldm_patched code at all, driving generation entirely over
# HTTP against a separately-running backend (start one first with webui-backend.sh).
#
# This is a proof-of-concept launcher, not a replacement for webui.sh: txt2img only,
# core params only. See PHASE8.md for exact scope and verification.
#################################################################

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

VENV_DIR="${FRONTEND_VENV_DIR:-venv-gr3}"
PORT="${FRONTEND_PORT:-7870}"
BACKEND_URL="${FRONTEND_BACKEND_URL:-http://127.0.0.1:7860}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "ERROR: ${VENV_DIR}/bin/python not found. Set FRONTEND_VENV_DIR to point at a venv with gradio+requests+pillow installed (torch NOT required)." >&2
    exit 1
fi

echo "Launching standalone frontend (BFISO Phase 8):"
echo "  backend:  ${BACKEND_URL}"
echo "  port:     ${PORT}"
echo "  venv:     ${VENV_DIR}"
echo "Start a backend first if you haven't:  bash webui-backend.sh"
echo

exec "${VENV_DIR}/bin/python" -u webui_frontend.py --backend-url "${BACKEND_URL}" --port "${PORT}" "$@"
