# PHASE8 — The Actual Cutover: a Torch-Free Frontend Driving a Real Backend

This is the phase every prior phase (0–7) was building toward: a Gradio process with
**no torch installed at all**, successfully driving real Stable Diffusion generation
against a separately-running backend process, entirely over HTTP.

## Why this needed a new entry point, not `modules/ui.py`

Checked first rather than assuming: `modules/ui.py` itself imports `sd_hijack`,
`sd_models`, `sd_samplers`, `sd_schedulers`, `processing`, `prompt_parser`,
`hypernetworks.ui` — all confirmed torch-heavy by PHASE3.md's scan. `webui.py`'s own
module scope runs `initialize_forge()` / `initialize.imports()` before anything else,
also torch-heavy. Neither can be reused or imported by a genuinely torch-free process.
This phase adds new, separate files instead:

- **`modules_frontend/txt2img_ui.py`** — the actual UI: prompt/negative prompt,
  steps/cfg/width/height/seed, a sampler dropdown fetched live from
  `GET /sdapi/v1/samplers`, script controls fetched live from
  `GET /sdapi/v1/script-info` and rendered via PHASE4.md's `build_controls_from_schema`,
  a Generate button that `POST`s to `/sdapi/v1/txt2img`, and a gallery + info textbox for
  the result.
- **`webui_frontend.py`** — the entry point (`argparse` for `--backend-url`/`--port`),
  deliberately independent of `webui.py`.
- **`webui-frontend.sh`** — launcher, mirroring `webui-backend.sh`'s style.

**Scope, stated honestly up front**: txt2img only, core params only. Script controls
render live against real backend data (proving PHASE4.md's builder works in a real app,
not just a verification harness) but their values are **not yet sent** with the
generation request — wiring them into `alwayson_scripts` JSON is separate follow-up work,
not bundled into this first cutover slice. No live progress bar yet either (Phase 5/6's
SSE work isn't wired into this new UI) — Generate just blocks until the result returns.

## Bugs found and fixed along the way (not pre-planned, discovered by actually running this)

1. **Argparse collision.** `InputAccordion.__init__` (via `modules.ui_script_schema`,
   see #3 below) lazily imports `modules.script_callbacks` → ... → `modules.shared` →
   `modules.shared_cmd_options`, which runs `parser.parse_args()` on the **real**
   `sys.argv` at import time — including this process's own `--backend-url` flag, which
   the backend's parser doesn't recognize, crashing with "unrecognized arguments." Fixed
   by setting `IGNORE_CMD_ARGS_ERRORS=1` (an escape hatch `modules/shared_cmd_options.py`
   already had for exactly this: `parse_known_args()` instead of `parse_args()`) before
   any import that could trigger the chain — used the existing mechanism rather than
   inventing a workaround.
2. **Gradio-on-Starlette TemplateResponse crash.** Gradio 3.41.2 calls Starlette's
   `TemplateResponse` with its old `(name, context)` signature; Starlette 0.36+ requires
   `(request, name, context)`, surfacing as `TypeError: unhashable type: 'dict'` deep in
   Jinja2 on first page load. The app's existing fix
   (`modules/ui_gradio_extensions.py`'s `reload_javascript()`) imports `modules.shared` →
   `modules.shared_items` → `modules.scripts` (loads every extension, most import torch)
   — exactly what this process must not need. Reimplemented the compatibility shim
   standalone in `webui_frontend.py`, without the JS/CSS-injection half (this frontend
   doesn't use `progressbar.js`), with zero `modules.*` dependency.
3. **`SamplerItem.options: dict[str, str]` — a real, pre-existing backend bug**,
   unrelated to this phase's own code, just never exercised before because nothing had
   called `/sdapi/v1/samplers` in this session until this frontend did. Real sampler
   configs include booleans (`second_order`, `brownian_noise`, `uses_ensd`,
   `discard_next_to_last_sigma`), which failed FastAPI's response validation against the
   too-strict `dict[str, str]` type, returning 500 on every call. Fixed:
   `dict[str, Any]` in `modules/api/models.py`.
4. **The actual `modules/safe.py` blocker from PHASE3.md, manifesting for real.** Running
   the frontend in a genuinely torch-free venv (not just reasoning about it) crashed with
   `ModuleNotFoundError: No module named 'torch'` — traced to
   `modules/ui_script_schema.py`'s `InputAccordion` builder (added in PHASE4.md for real
   fidelity over a plain Checkbox substitute), whose real class's `__init__` lazily
   imports `modules.script_callbacks` → `modules.extensions` → `modules.shared` →
   `modules.shared_items` → `modules.scripts` (the full `load_scripts()` machinery) →
   `modules.paths` → `modules.safe` → `import torch`. PHASE4.md's fidelity improvement
   assumed same-process reuse; it silently broke `ui_script_schema.py`'s own documented
   promise ("this module only imports gradio, deliberately") once actually run
   cross-process. **Fixed by reverting to a plain `gr.Checkbox`** for `InputAccordion`-typed
   args — same value semantics (open/closed as bool), just without the paired Accordion's
   visual open/close behavior. Re-verified zero mismatches don't apply here (this is a
   deliberate, documented fidelity reduction, not a bug), but confirmed the module is
   genuinely torch-free again by the successful run below.

## Verification

**Stage 1 — quick iteration in `venv-gr3`** (still has torch, fast to debug): launched a
real backend (`webui-backend.sh`), launched the new frontend against it, drove a real
generation through the browser tool. Found and fixed bugs #1–#3 here. Result: real image
generated, `Generation info` correctly populated (seed, sampler, model hash, version), no
console errors.

**Stage 2 — the actual proof.** Built a fresh venv (`python3.12 -m venv`, since the
system's `python3.11` had broken venv path resolution) with exactly
`gradio==3.41.2 pillow==10.4.0 numpy==1.26.4 requests setuptools` (the `requirements_versions.txt`
frontend pins, plus `setuptools` — Python 3.12 removed stdlib `distutils`, which old
Gradio still imports; `setuptools` provides the compatibility shim). **No torch package
in this venv at all** — confirmed via `pip list` and `find -iname "*torch*"` (the only
filesystem hits were `huggingface_hub`'s internal torch-serialization helper file, dead
code without torch actually importable). Found and fixed bug #4 here — the first run
crashed with a real `ModuleNotFoundError: No module named 'torch'`.

After the fix, re-ran: clean startup, zero errors in the launch log. Opened in the
browser, typed a real prompt, clicked Generate:

```
imgCount: 3
infoText: "a blue ceramic mug on a marble countertop, studio lighting
Steps: 20, Sampler: DPM++ 2M, Schedule type: Karras, CFG scale: 7.0, Seed: 556726566,
Size: 512x512, Model hash: f116b0c78f, Model: waiIllustriousSDXL_v170, Clip skip: 2,
Version: f1.0.0v2-v1.10.1RC-latest-2661-gc500bdb0"
```

Real image, real generation info, sampler dropdown correctly populated from the
backend's live sampler list, script controls rendered from the backend's live schema —
all from a process with zero torch installed, talking only over HTTP to a separate
backend process. No console errors.

**Regression check**: re-ran `pytest test/test_txt2img.py test/test_img2img.py
--no-server` against the backend used throughout — same 13 passed / 2 pre-existing
failures (PHASE2.md) / 1 skipped as every prior phase's baseline. No new failures from
the `SamplerItem`/`ui_script_schema.py` changes.

## Staging (per request: revertible if something's wrong)

Committed as its own isolated commit, separate from the Phase 0–7 baseline
(`c500bdb0`) already committed before this phase started:
- New files: `modules_frontend/__init__.py`, `modules_frontend/txt2img_ui.py`,
  `webui_frontend.py`, `webui-frontend.sh`, `PHASE8.md`.
- Modified: `modules/api/models.py` (`SamplerItem.options` type fix — bug #3),
  `modules/ui_script_schema.py` (InputAccordion revert — bug #4), `.gitignore`
  (`!webui-frontend.sh` allowlist entry, matching the earlier `webui-backend.sh` pattern).
- `git revert` this one commit to return to the Phase 0–7 state without touching any of
  that work, if something about this phase turns out wrong later.

## What this phase did NOT do (real remaining work, named specifically)

- **Script args aren't sent with the request.** The controls render and are interactive,
  but `run_txt2img()` doesn't read their values into `alwayson_scripts` JSON. Needs a
  mapping from the schema's flat arg list back to each script's expected JSON key/order —
  a real, separate piece of work.
- **No live progress.** Generate blocks synchronously until the backend responds. Phase
  5/6's SSE endpoint and JS aren't wired into this new UI at all.
- **txt2img only.** img2img, Extras, PNG Info, and every other tab don't exist in this
  frontend yet.
- **Layout is flat**, not matching the shipped UI's category/accordion sectioning
  (PHASE4.md already flagged this as out of scope for the schema mechanism itself).
- **Still same-machine, same-checkpoint testing.** Didn't test against a *remote* backend
  (different host) or verify behavior when the backend restarts/reloads a different
  checkpoint mid-session.
