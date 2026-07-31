# PHASE1 — Venv-Split Proof

Executes [BFISO.md](BFISO.md) §6 Phase 1: "prove the venv split... just to prove numpy
versions can diverge without touching the wire protocol yet." Spike code lives in the
session scratchpad (throwaway, not committed to the repo) — this file is the durable
record of what was done and what it proved.

## Setup

The repo's actual pinned venv is `venv-gr3/` (not `venv/` as CLAUDE.md's tooling section
names it — worth a doc fix later, noted but out of scope here). Confirmed baseline:

```
venv-gr3/bin/python --version   → Python 3.13.13
venv-gr3/bin/python -c "import numpy; print(numpy.__version__)"  → 1.26.4
venv-gr3/bin/python -c "import gradio; print(gradio.__version__)" → 3.41.2
```

Created a second, throwaway venv with a newer Python and an unpinned numpy:

```bash
python3.12 -m venv venv-backend-spike
venv-backend-spike/bin/pip install "numpy>=2"
# → numpy 2.5.1
```

## Spike

Two scripts (scratchpad `bfiso_phase1/`):

- `worker.py` — runs under the backend venv's interpreter. Reads a JSON payload from
  stdin, does a trivial numpy computation, and reports `np.__version__` plus
  `hasattr(np, "float_")` — `float_` was removed in numpy 2.0, so this is proof the
  *behavior* differs, not just the version string.
- `launcher.py` — run under `venv-gr3/bin/python` (simulating the pinned frontend
  process). Reports its own numpy/Python version, then spawns
  `venv-backend-spike/bin/python worker.py` via `subprocess.run(..., input=..., text=True)`,
  passing a JSON payload on stdin and reading the JSON result from stdout.

## Result

```bash
venv-gr3/bin/python launcher.py
```

```json
{
  "process": "frontend (this process, simulating the Gradio venv)",
  "numpy_version": "1.26.4",
  "python_version": "3.13.13",
  "has_np_float_alias": true
}
{
  "numpy_version": "2.5.1",
  "python_version": "3.12.12",
  "has_np_float_alias": false,
  "sum": 10.0,
  "mean": 2.5,
  "process": "backend (subprocess, separate venv)"
}
```

**Proven:**
- The frontend process (`venv-gr3`) keeps Gradio's required numpy 1.26.4 (`float_` alias
  still present) throughout.
- The backend subprocess independently runs numpy 2.5.1 (`float_` alias correctly absent
  — confirms it's genuinely running the newer runtime, not just reporting a different
  string) on a different Python minor version (3.12 vs 3.13) entirely.
- A JSON-over-stdin/stdout round trip across the process boundary works and produces a
  correct computation (`sum`/`mean`) from data the frontend process sent.

This confirms the core premise of the whole BFISO effort: nothing prevents the backend
from running a different numpy/Python than the Gradio process once they're separate
processes with separate venvs. The mechanism (subprocess + JSON on stdio) is a stand-in
for Phase 1 only — Phase 2 replaces it with the HTTP+JSON service decided in BFISO.md §4.

## What this does NOT resolve

This proof is orthogonal to the blocker found in [PHASE0.md](PHASE0.md) Risk 2: script
modules under `extensions-builtin/` still eagerly import torch/`ldm_patched` at module
scope via `load_scripts()`. That's a code-structure problem inside the backend's own
import graph, not a venv-boundary problem — proving venv independence here doesn't touch
it. It remains required work before Phase 3.

## Next steps (Phase 2 — extract the backend service)

Per BFISO.md §6, Phase 2 is: move `processing.py`, `scripts.py`, `script_callbacks.py`,
`extensions-builtin/`, `extensions/`, `ldm_patched/`, `modules_forge/`, and model loading
behind a standalone FastAPI process built by generalizing `modules/api/api.py`; the
Gradio frontend becomes an HTTP client using the JSON contract from BFISO.md §4.

Concrete starting points identified so far, to pick up next session:
- Generalize `modules/api/api.py`'s FastAPI app so it can run standalone (own process,
  own `uvicorn`/venv) rather than mounted inside the Gradio app's process — check how
  `modules/api/api.py` is currently instantiated/mounted (likely in `webui.py`/`launch.py`)
  before assuming it can just be lifted out unchanged.
- Decide, before writing the client side, whether to reuse the `subprocess`+stdio pattern
  proven here anywhere (e.g. short-lived tooling) or go straight to long-lived HTTP for
  everything — BFISO.md §4 already argues for HTTP+JSON, so the subprocess pattern from
  this phase should not be carried forward as the real mechanism.
- Still open: port vs. rebuild the SSE/progress work from `feat/user-session-jobqueue`
  (PHASE0.md Risk 1) — needs a decision before Phase 2's progress-endpoint design is final.
- Still open: the script-loading refactor from PHASE0.md Risk 2 — likely needs to happen
  *during* Phase 2 (since it's part of moving `scripts.py` backend-side cleanly), not
  deferred to Phase 3 as BFISO.md originally sequenced it. Worth revisiting the phase
  ordering itself next session.

*(This file records Phase 1 as executed. If a later session re-runs or extends the venv
proof — e.g. adding a torch/`ldm_patched` import to the backend worker — append a new
"Phase 1b" section below rather than rewriting the result above; overwrite only the
"Next steps" section as it's superseded by actual Phase 2 progress.)*
