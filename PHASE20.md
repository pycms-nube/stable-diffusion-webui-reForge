# PHASE20 — Combined Launch Script

New `webui-split.sh`: one command starts the backend, waits for it to become
healthy, then starts the frontend against it, and tears both down together on
exit. This is the first step toward the user's stated longer-term goal —
"see if we can replace current project implementation so future Gradio and
backend upgrades are easier" — specifically the piece they picked first when
asked to prioritize: **launch integration**, not new feature coverage.

## Explicit scope decision (asked, not assumed)

The full "replace the current implementation" goal is large — the shipped
`modules/ui.py` covers Settings, Extras, PNG Info, Train, and every extension's own
UI (ControlNet's tab, etc.), none of which the split frontend (Phases 8-19) touches.
Rather than guess which slice to build next, the user was asked to choose between
(a) launch integration, (b) a Phase-0-style investigation cataloging the full gap
before writing code, or (c) building the next-highest-value missing tab (Settings).
They picked (a) — this phase is scoped to exactly that, not a step toward feature
parity.

## What changed

- `webui-split.sh`: backgrounds `webui-backend.sh`, polls
  `GET /sdapi/v1/samplers` until `200` (or the process exits early, or
  `BACKEND_HEALTH_TIMEOUT` — default 120s — elapses), then backgrounds
  `webui-frontend.sh` pointed at it. A `trap ... EXIT INT TERM` kills both children
  on any exit path. `FRONTEND_BACKEND_URL` defaults to
  `http://127.0.0.1:${BACKEND_PORT}`, so overriding `BACKEND_PORT` alone (without
  also having to separately set `FRONTEND_BACKEND_URL`) still points the frontend at
  the right place.
- `.gitignore`: added `!webui-split.sh` to the existing allowlist section (the repo
  blanket-ignores `*.sh`, allowlisting specific launcher scripts by name — the same
  pattern `webui-backend.sh`/`webui-frontend.sh` already needed).

## Verification

**Real end-to-end run**: launched `webui-split.sh` with `BACKEND_CKPT`/
`BACKEND_VENV_DIR`/`FRONTEND_VENV_DIR` set, watched the log show the backend load a
real SDXL checkpoint, the health-check loop poll and succeed
(`GET /sdapi/v1/samplers 200 OK` → `Backend is up.`), then the frontend start
automatically against the right backend URL and become reachable
(`curl` returned `200` on both `:7860` and `:7870`) — a genuine cold start of both
processes from one command, not a mocked test.

**Shutdown — a real finding, not glossed over**: sent `SIGTERM` to the script's PID
— confirmed via the log (`"Shutting down..."` printed) and `ps` (both child
processes, launch.py and webui_frontend.py, gone) that cleanup fired correctly and
tore down both processes. Also tried `SIGINT` first and found it did **not** trigger
cleanup — `/proc/<pid>/status` showed `SigIgn` including signal 2 (SIGINT) with no
corresponding entry in `SigCgt`, meaning the trap genuinely wasn't registered for
that signal in this run. Traced this to how the test itself was launched, not a bug
in the script: `webui-split.sh` was started as `nohup bash webui-split.sh ... &` (to
run it non-interactively from this session's sandboxed shell, which has no TTY) —
and bash's documented behavior for **asynchronous list commands** in a non-interactive
shell is to force SIGINT/SIGQUIT to be ignored for that command, which is exactly
what an async `&`-launched process is. A real user running `bash webui-split.sh` (or
`./webui-split.sh`) directly in their own interactive terminal runs it in the
**foreground**, not as an asynchronous list command, so this rule doesn't apply
there — foreground scripts receive and can trap SIGINT normally, the same mechanism
that made SIGTERM's trap fire correctly here. This distinction is stated explicitly
rather than either quietly assuming Ctrl-C works or overclaiming it was verified:
SIGTERM cleanup is directly proven; Ctrl-C-in-a-real-terminal is expected to work via
the same trap but wasn't independently re-provable from this sandboxed, non-TTY
environment.

## What this phase did NOT do

- **Not a step toward feature parity** — this only changes *how* the existing
  txt2img/img2img split frontend is started, not what it can do. Settings, Extras,
  PNG Info, Train, and extension UI are all still exclusively on the original
  `webui.sh` path.
- **No browser auto-open** — matches the existing two scripts' behavior (they print
  the URL, they don't launch a browser), kept consistent rather than adding new
  behavior neither predecessor script has.
- **Ctrl-C in a real interactive terminal session wasn't independently re-verified**
  beyond the reasoning above (this environment has no TTY to test that exact path) —
  worth a quick manual check if the user wants full confidence, though the mechanism
  is standard and SIGTERM's success through the identical trap is strong evidence it
  will behave the same way.
- **`webui.sh` itself is untouched**, per the same convention every prior phase's
  scripts have followed (it explicitly says not to modify it).
