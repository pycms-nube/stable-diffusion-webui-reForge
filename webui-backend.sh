#!/usr/bin/env bash
#################################################################
# Debugger-attachable, API-only backend launcher (BFISO Phase 2)
#
# This is a NEW, separate script — webui.sh itself is intentionally left
# untouched (its own header says not to modify it), and webui-user.sh holds
# unrelated frontend flags (--listen, --forge-jax-pipeline, ...) that don't
# belong in a clean backend-only launch.
#
# What this does: launches the existing pipeline (processing.py, scripts,
# extensions, ldm_patched) as a standalone FastAPI process via the
# already-existing `--nowebui --api` mode, with no Gradio UI attached. It's
# a single plain foreground `python launch.py` process — no re-exec, no
# multiprocessing workers — so a debugger (debugpy, an IDE's "attach to
# process", gdb, py-spy, ...) can attach to its PID normally at any point
# after launch.
#
# See PHASE2.md for what was proven, exact commands, and test results.
#################################################################

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

VENV_DIR="${BACKEND_VENV_DIR:-venv-gr3}"
PORT="${BACKEND_PORT:-7860}"
CKPT="${BACKEND_CKPT:-test/test_files/empty.pt}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "ERROR: ${VENV_DIR}/bin/python not found. Set BACKEND_VENV_DIR to point at a venv with the app's dependencies installed." >&2
    exit 1
fi

CMD=(
    "${VENV_DIR}/bin/python" -u launch.py
    --skip-prepare-environment
    --skip-torch-cuda-test
    --nowebui --api --api-server-stop
    --disable-nan-check
    --always-low-vram
    --ckpt "${CKPT}"
    --port "${PORT}"
)

echo "Launching backend-only API server (PID will be printed by the shell):"
printf '  %s\n' "${CMD[*]}"
echo "Health check once up:  curl http://127.0.0.1:${PORT}/sdapi/v1/options"
echo "Stop it gracefully:    curl -X POST http://127.0.0.1:${PORT}/sdapi/v1/server-stop"
echo "Override checkpoint/port/venv via BACKEND_CKPT / BACKEND_PORT / BACKEND_VENV_DIR env vars."
echo "Extra flags passed to this script are forwarded to launch.py unchanged."
echo

exec "${CMD[@]}" "$@"
