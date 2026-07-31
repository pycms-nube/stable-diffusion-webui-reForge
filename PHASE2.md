# PHASE2 — Backend-Only Extraction (Debugger-Attachable, Launcher-Started)

Executes the scoped-down version of [BFISO.md](BFISO.md) §6 Phase 2 that was agreed on:
launch the existing pipeline as a standalone process, launchable via a webui-style
launcher script and attachable with a debugger. The Gradio frontend is **unchanged** —
it still calls `processing.py` in-process exactly as it does today. This phase proves
the backend can run standalone and generate real images over HTTP; it does not yet wire
the Gradio UI to call it (that's a larger follow-up, tracked in "Next steps" below).

## What already existed (no code needed)

`--nowebui`/`--api` (`modules/cmd_args.py:104,107`) already boot this codebase as an
API-only FastAPI process with no Gradio UI — confirmed via `webui.py:245` /
`modules/launch_utils.py:577` (`webui.api_only()` vs `webui.webui()`). The test suite
(`test/conftest.py`) already launches exactly this mode for its own HTTP tests. So the
backend-extraction mechanism itself required zero new backend code — only a convenience
launcher and verification that it actually works end-to-end for real generation.

## Launching it

New file: [webui-backend.sh](webui-backend.sh). A separate script, not a modification of
`webui.sh` — that file's own header says not to change it, and `webui-user.sh` holds the
user's real frontend flags (`--listen`, `--forge-jax-pipeline`, etc.) that don't belong
in a backend-only launch.

```bash
./webui-backend.sh
# or, to point at a real checkpoint:
BACKEND_CKPT=/path/to/model.safetensors ./webui-backend.sh
```

Env overrides: `BACKEND_VENV_DIR` (default `venv-gr3`), `BACKEND_PORT` (default `7860`,
matching `pyproject.toml`'s test `base_url` ini — see gotcha below), `BACKEND_CKPT`
(default `test/test_files/empty.pt`, but see gotcha below — that fixture doesn't
actually work on this branch). Extra args are forwarded to `launch.py` unchanged.

The script `exec`s `python -u launch.py ...` directly (no subshell left running), so the
shell-reported PID is the actual server process. `launch.py` never re-execs or forks a
worker after that (confirmed by reading `launch.py`/`modules/launch_utils.py:start()` —
`prepare_environment()` may run `pip` as subprocesses, but `--skip-prepare-environment`
avoids that entirely) — it's one plain foreground process from boot to shutdown, so any
standard debugger (`debugpy` attach, an IDE's "attach to process", `gdb python`,
`py-spy`) attaches to it normally at any point.

## What was actually run and proven

1. **Manual generation test, real checkpoint, real GPU.** Launched via the script above
   pointed at `waiIllustriousSDXL_v170.safetensors` (a real SDXL checkpoint found under
   the shared StabilityMatrix `Data/Models/StableDiffusion/` directory), on the machine's
   actual RTX 2080 (Max-Q, 8GB) with `--always-low-vram`. Sent a real request:
   ```
   POST /sdapi/v1/txt2img
   {"prompt":"a red apple on a wooden table...","steps":10,"width":512,"height":512,...}
   ```
   Got back a valid 512×512 PNG (decoded and verified with PIL) in ~13s. This is a real
   image generated entirely through the standalone `--nowebui --api` process, proving the
   backend-only extraction produces correct output end-to-end.

2. **Automated test suite, same live instance.** Installed `requirements-test.txt` into
   `venv-gr3` (pytest wasn't previously installed there) and ran the project's existing
   test suite against the running backend:
   ```
   venv-gr3/bin/python -m pytest test/test_txt2img.py --no-server -q
   → 11 passed, 1 skipped (restore_faces — no face-restore model installed, expected)
   ```
   Re-ran after stopping and relaunching via `webui-backend.sh` itself (not a manual
   `launch.py` invocation) — same result, confirming the launcher script boots a fully
   test-passing instance, not just a superficially-responding one.
   `test/test_img2img.py` was also run out of curiosity: 2/4 passed (both inpainting
   variants); `test_img2img_simple_performed` and `test_img2img_sd_upscale_performed`
   failed with HTTP 422 (request schema validation, not a server error) — this is a
   pre-existing mismatch between that test file's request fixtures and the current
   Pydantic API models, unrelated to backend extraction. Not fixed here; out of scope for
   this phase.

## Gotchas discovered along the way (pre-existing, not introduced by this work)

- **`test/test_files/empty.pt` no longer works as a fixture on this branch.** It's a
  torch-pickled `{}` (confirmed by loading it directly) — a genuinely empty state dict.
  `modules_forge/forge_loader.py`'s current `load_checkpoint_guess_config` requires
  actual key patterns to detect architecture and raises `RuntimeError: Could not detect
  model type` on it. This means `conftest.py`'s own server-bootstrap fixture (which uses
  `empty.pt` by default) currently cannot serve real generation requests on this branch —
  a pre-existing gap, not something this phase broke. Worked around here by pointing at a
  real checkpoint instead; someone should regenerate `empty.pt` against the current
  `forge_loader` detection logic (or replace the "empty" concept with a minimal-but-valid
  fake state dict) if the fixture is meant to support real generation tests.
- **`conftest.py`'s `base_url` fixture ignores the `--base-url` CLI flag.** It calls
  `request.config.getini("base_url")` first and returns that unconditionally if truthy —
  `pyproject.toml` hardcodes `base_url = "http://127.0.0.1:7860"`, so `pytest --base-url
  http://127.0.0.1:7861` is silently ignored. Worked around here by launching the backend
  on port 7860 instead of fighting the fixture. Also pre-existing, not fixed here.
- **Some checkpoints in the shared model directory have no baked-in CLIP** (e.g.
  `miaomiaoHarem_anima13.safetensors`, `novaFurryAM_v20.safetensors` both failed with
  `RuntimeError: No CLIP model available for SD15 conditioner setup.`, expecting a
  separate text-encoder file this environment doesn't have configured) — a property of
  those specific checkpoint files, not a bug. `waiIllustriousSDXL_v170.safetensors`
  worked cleanly.
- `extensions-builtin/soft-inpainting/scripts/soft_inpainting.py` fails to import
  (`ModuleNotFoundError: No module named 'joblib'`) in `venv-gr3` — pre-existing missing
  dependency, logged as a warning at script-load time, doesn't block server startup.

## What this phase deliberately did NOT do

- **Did not rewire the Gradio frontend.** Per the scoping decision at the start of this
  phase, the frontend still calls `processing.py` in-process, unchanged. Nothing here
  lets you press Generate in the actual web UI and have it hit the new backend process —
  that's the "full split" option that was explicitly deferred.
- **Did not attempt real numpy/venv divergence for the backend's actual dependencies.**
  [PHASE1.md](PHASE1.md) already proved venv independence is *possible* with a throwaway
  numpy-only spike. Standing up a second full venv with `torch`==2.11.0 + the rest of
  `requirements_versions.txt` at a different numpy version was out of scope here (heavy,
  and blocked from mattering yet by [PHASE0.md](PHASE0.md) Risk 2 — the frontend still
  needs to import script modules that import `ldm_patched`/torch, so it can't yet be in a
  differently-pinned venv anyway). This phase's backend ran in `venv-gr3`, same as the
  frontend, deliberately — the point here was proving process separation and the wire
  contract, not re-proving venv independence.

## Next steps

- **Frontend rewrite** (the deferred "full split"): rewire `modules/txt2img.py`/
  `img2img.py` to call the backend over HTTP. Still blocked on PHASE0.md Risk 2 (script
  UI can't render without torch) unless extensions are disabled for a first cut, or the
  script-loading refactor happens first.
- **Fix or replace `test/test_files/empty.pt`** so `conftest.py`'s own fixture works for
  real generation tests without needing a real multi-GB checkpoint — otherwise every
  future contributor hits the same `Could not detect model type` wall this phase did.
- **Fix `conftest.py`'s `base_url` fixture** to actually respect `--base-url` (use the
  `pytest-base-url` plugin's own resolution order instead of reading the ini directly),
  so tests can target a non-default port without renaming/moving servers.
- **Real venv/numpy divergence for the backend**, once Risk 2 is resolved and there's a
  reason the frontend and backend actually need different numpy versions in practice —
  install `requirements_versions.txt` (minus the Gradio/numpy ceiling) into a fresh venv
  and repeat this phase's generation + test-suite proof against it.
