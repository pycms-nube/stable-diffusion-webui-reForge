# PHASE9 — Closing Phase 8's Two Named Gaps: Script Args + Live Progress

PHASE8.md ended with an explicit, honest "what this phase did NOT do" list. The two
biggest items on it were: script control values render but aren't sent with the
generation request, and there's no live progress in the new frontend. This phase closes
both, in the same `modules_frontend/txt2img_ui.py` the Phase 8 cutover proved is
genuinely torch-free.

## 1. Script args wired into the request

`build_alwayson_script_controls()` now returns `[(script_name, [gr.Component, ...]),
...]` instead of discarding the built components. `create_ui()` flattens this into the
click handler's `inputs` list and keeps `script_specs = [(name, count), ...]` alongside
it. `run_txt2img()` slices the flat trailing `*script_arg_values` back into per-script
chunks using those counts and builds the `alwayson_scripts` payload key exactly the
shape `modules/api/api.py`'s `text2imgapi` expects: `{name: {"args": [values...]}}`.

**Verified it actually reaches the backend**, not just that the request doesn't error:
toggled "Enable Mahiro CFG" on in the rendered script controls, generated, and the
returned `info` string included `mahiro_cfg_enabled: True` — that key only appears when
the script itself received and processed the arg, so this is the script's own code
confirming receipt, not just an assumption about payload shape.

## 2. Live progress + preview

Uses the request's existing `force_task_id` field (`modules/api/models.py` — already
present, unrelated to this session) so the frontend can mint its own task id before
submitting, then poll Phase 5's `GET /internal/progress-stream?id_task=...` for that
exact id while the (blocking) `POST /sdapi/v1/txt2img` runs. Structure:

- `_post_txt2img()` runs the blocking POST in a background `threading.Thread`, writing
  the result (or exception) into a shared dict once done.
- `_stream_progress()` opens the SSE stream with `requests.get(..., stream=True)` and
  yields `(progress_text, preview_image)` per event until the POST thread signals done
  or the stream reports `completed`.
- `run_txt2img()` is now a generator: it starts the thread, drains
  `_stream_progress()`'s yields into the UI (leaving the gallery/infotext outputs
  untouched via `gr.update()` while streaming), then joins the thread and yields the
  final images + infotext.

## Bugs found fixing this (not pre-planned — found by actually clicking Generate)

1. **A lambda wrapping a generator function isn't itself a generator function.**
   `fn=lambda *args: run_txt2img(backend_url, script_specs, *args)` crashed with
   `ValueError: An event handler didn't receive enough output values (needed: 4,
   received: 1)` — Gradio checks `inspect.isgeneratorfunction(fn)` on the object it was
   given, and a lambda that calls a generator function just returns a generator object
   when invoked; it has no `yield` of its own, so the check is `False` and Gradio
   treats the single generator object as one static return value instead of iterating
   it. Confirmed the fix in a throwaway interpreter check before touching the file:
   `functools.partial(gen_fn, arg)` preserves `isgeneratorfunction() == True` where a
   `lambda` wrapper does not. Fixed: `fn=functools.partial(run_txt2img, backend_url,
   script_specs)`.
2. **Gradio 3.x needs an explicit queue to stream generator outputs at all.** Even after
   fix #1, clicking Generate crashed with `ValueError: Need to enable queue to use
   generators.` — multi-yield event handlers require `demo.queue()`, which nothing in
   Phase 8's single-yield `create_ui()` had ever needed. Fixed: `demo.queue()` before
   `create_ui()` returns.

## Verification

**Script args**: real browser session (via the `Claude_Browser` tool driving the
already-running `venv-gr3` frontend against a real `webui-backend.sh`), confirmed by the
`mahiro_cfg_enabled: True` round-trip above, plus a full render of all 55 scripts'
controls at real values (same set Phase 4 verified, now larger — new script categories
like `clpc sampler settings` picked up automatically since the schema is backend-served,
not hand-maintained).

**Progress**: confirmed via the same session — polling the Progress textbox mid-generation
returned real intermediate values (`57%`, `15%`, `77%` across separate runs), not a
single jump straight to `"done"`.

**Live preview — methodology change mid-phase.** Automated verification of the preview
image hit a real snag: the host OS hard-crashed during heavy automated browser-driven
generation testing (unclear whether caused by GPU/memory pressure from repeated SDXL
runs or unrelated — not conclusively diagnosed, and not chased further since it wasn't
reproducible on demand). Rather than keep re-running the same heavy automated loop that
preceded the crash, verification switched to the user manually opening
`http://127.0.0.1:7870` in their own browser and driving a real generation by hand.
**Confirmed by the user**: live preview images visibly updating mid-generation, progress
bar moving through real intermediate values (0% → 20% → 60%). Before this, a curl-only
probe of the raw SSE stream had shown `live_preview` as `null` across an entire manual
`curl`-triggered generation — that turned out to be an artifact of the ad hoc curl/bash
backgrounding test harness (subshell/timing quirks under the sandboxed tool, not
reproduced or explained further), not a real backend defect; the user's real-browser
confirmation is the trustworthy signal here and supersedes it.

**Regression check**: `pytest test/test_txt2img.py test/test_img2img.py --no-server`
against the same `venv-gr3` backend — `2 failed, 13 passed, 1 skipped`, identical to
every prior phase's baseline (the 2 failures are `test_img2img_simple_performed` /
`test_img2img_sd_upscale_performed`, pre-existing per PHASE2.md, unrelated to this
session's changes). No new failures from the script-arg or progress-streaming code.

**Torch-freedom**: not re-run as a full from-scratch venv proof this time (that heavy
verification loop is what preceded the OS crash, and Stage 1's fast-iteration bugs were
real and worth fixing before spending that cost again). Verified instead by inspection:
the only new imports added this phase are `functools`, `threading`, `time`, `uuid` — all
stdlib — plus `demo.queue()`, a plain Gradio API call. `modules_frontend/txt2img_ui.py`'s
import list remains exactly `gradio`, `requests`, `PIL`, and
`modules.ui_script_schema` (itself torch-free per PHASE4.md). Re-running the genuine
torch-free venv proof from PHASE8.md's Stage 2 is reasonable follow-up if that
reassurance is wanted, but isn't required to trust this phase's claims given the import
list didn't change in kind.

## What this phase did NOT do

- **No img2img/other tabs** — still txt2img only, same as Phase 8.
- **Preview cadence is whatever the backend's `show_progress_every_n_steps` /
  `live_preview_refresh_period` settings already dictate** — nothing added to the
  frontend to control or improve that cadence.
- **No cancel/interrupt button** — Generate blocks (with live feedback now) but there's
  no way to stop a running generation from this frontend.
- **Layout still flat** — same PHASE4.md-flagged gap, unaddressed.
- **The OS-crash root cause is not diagnosed.** Noted honestly above rather than
  silently worked around; if it recurs, it needs its own investigation rather than being
  assumed as this phase's automated-testing pattern being inherently unsafe.
