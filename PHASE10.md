# PHASE10 — Interrupt/Skip + Missing Core Params

Small, self-contained follow-up to Phase 9: two more gaps in
`modules_frontend/txt2img_ui.py` that were real but hadn't been named yet — no way to
stop a running generation, and several standard txt2img params (batch count/size,
restore faces, tiling) that the UI never exposed or sent at all.

## What changed

- **Batch count (`n_iter`), batch size (`batch_size`), Restore faces (`restore_faces`),
  Tiling (`tiling`)** — added as controls (two new rows under Width/Height) and wired
  into `run_txt2img()`'s payload. These are plain `StableDiffusionProcessingTxt2Img`
  fields already accepted by the backend; the gap was purely that the frontend never
  built or sent them.
- **Skip / Interrupt buttons** — thin wrappers over the existing
  `POST /sdapi/v1/interrupt` and `POST /sdapi/v1/skip` endpoints (`modules/api/api.py`),
  both already used by the shipped Gradio UI's own equivalent buttons. No backend
  changes needed — `shared.state` lives entirely backend-side, so these are simple
  fire-and-forget POSTs, not something this frontend needs to track state for.

## The bug that made this phase two round trips instead of one

First pass: both buttons appeared to do nothing — the running generation continued
uninterrupted and the "Interrupt requested." / "Skip requested." message only appeared
*after* the generation had already finished on its own. The user caught this by
actually clicking the buttons mid-generation (exactly the kind of dynamic behavior that
needs a live click, not something inferable from reading the code).

**Root cause**: `demo.queue()` (added in Phase 9, needed for `run_txt2img`'s generator
yields) defaults to `concurrency_count=1` in Gradio 3.41.2. That serializes *all*
queued events through one slot — `run_txt2img`'s generator holds that slot for the
entire generation, so a Skip/Interrupt click queues up behind it and only executes once
Generate's handler has already released the slot by finishing naturally. The buttons
weren't broken; they were correct, just stuck waiting in line behind the very run they
were supposed to interrupt.

**Fix**: `demo.queue(concurrency_count=3)` — lets Skip/Interrupt's short-lived handlers
run concurrently with Generate's long-lived one instead of queueing behind it.

## Verification

Given the OS crash in the PHASE9.md session (during heavy automated browser-driven
generation testing, root cause undiagnosed), this phase deliberately used a lighter
verification split:

- **Light, one-shot automated checks** (not loops, not rapid polling): a single
  `curl` txt2img request with `restore_faces=true, tiling=true` confirmed
  `Face restoration: CodeFormer` and `Tiling: True` both appear in the real returned
  infotext — proving the new fields reach the actual processing pipeline, not just that
  the request doesn't 422. Separately, one `curl` each to `/sdapi/v1/interrupt` and
  `/sdapi/v1/skip` confirmed both return `200` with no generation running.
- **Dynamic/live behavior handed to the user for manual review**, per their explicit
  request after the earlier crash: does clicking Skip/Interrupt during a real running
  generation actually take effect. First manual check caught the concurrency bug above;
  second manual check confirmed the fix — buttons now behave normally, with the
  "requested" message appearing immediately rather than only after completion.

This is a deliberate methodology shift from Phase 8/9: automated testing for anything
that's a single request/response, manual testing for anything that depends on live,
concurrent, multi-second UI state — the exact class of check that stressed the user's
hardware before.

**Regression check**: not re-run this phase — no backend code changed (both new params
and both new endpoints were already exposed and already used elsewhere by the shipped
UI), so there's nothing in `modules/` for `pytest test/test_txt2img.py
test/test_img2img.py` to newly cover.

## What this phase did NOT do

- **No hires-fix, no img2img/other tabs** — same gaps as Phase 8/9, still open.
- **No cancel confirmation / disabled-state UX** — Skip/Interrupt fire immediately with
  no "are you sure" or visual indication the run is being torn down beyond the progress
  message.
- **`concurrency_count=3` is a round number, not tuned** — enough to unblock
  Skip/Interrupt racing a single Generate; not evaluated under multiple concurrent
  Generate clicks or multiple browser tabs against the same frontend process.
