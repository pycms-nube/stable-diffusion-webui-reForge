# PHASE18 — Batch (Multiple Images)

Closes the "no Batch sub-tab" gap: a "Batch (multiple images)" accordion on the
img2img tab uploads several images at once and runs the current
prompt/steps/sampler/etc. settings against each one in turn.

## Why this looks different from the shipped UI's Batch sub-tab

The shipped UI's Batch sub-tab points at a local `input_dir`/`output_dir` on the
**backend's own filesystem** — that only works because the Gradio UI and the
inference backend are the same process on the same machine. This whole BFISO effort
exists to let them be different processes, potentially on different machines, so a
directory path typed into this frontend wouldn't necessarily mean anything on the
backend's disk. Rather than build a fragile "hope the paths line up" feature, this
phase uploads the images through the browser (`gr.File(file_count="multiple")`) and
loops real `POST /sdapi/v1/img2img` calls over them client-side instead — works
regardless of whether frontend and backend share a filesystem, which is the point.

## What changed

- `run_batch_img2img()`: takes the uploaded file list plus the *same* live
  prompt/negative_prompt/steps/sampler_name/cfg_scale/width/height/seed/
  restore_faces/tiling/denoising_strength/resize_mode/script-arg components already
  on the tab (passed as extra inputs to a second click handler — not duplicated into
  a separate form), opens each uploaded file as a PIL image, and POSTs one
  `/sdapi/v1/img2img` request per image sequentially. Yields a "Processing N/M..."
  status after each completed image and accumulates results into one gallery.
  Deliberately no per-image SSE live progress (unlike Generate) — that's this phase's
  one real simplification, not a bug: the batch just isn't going to look at
  `/internal/progress-stream` for each image in the loop, only reports before/after
  each request.
- New UI: `gr.File(file_count="multiple")`, "Run Batch" button, its own progress
  textbox and output gallery, separate from the single-image Generate flow's gallery.

## Verification

**Direct call against the real backend, no browser needed for a first check**:
constructed two synthetic in-memory PNGs, wrapped them in objects shaped like what
`gr.File` actually hands the callback (`SimpleNamespace(name=<path>)`, matching
Gradio's own file-object `.name` attribute), and called `run_batch_img2img()` as a
plain generator. Observed real progress ticks `Processing 0/2...` →
`Processing 1/2...` → `Processing 2/2...` → `Batch done: 2 image(s) processed.`, with
2 real decoded result images returned — a genuine sequential run against the live
backend, not a mock.

**Build check**: `create_ui()` builds cleanly with the new Batch accordion/button/
gallery, no exceptions.

**Live UI**: bundled into the Phase 16-19 consolidated review, per the user's
instruction.

## What this phase did NOT do

- **No per-image live progress bar or preview** — only a before/after "Processing
  N/M" message, unlike Generate's SSE-driven progress+preview.
- **No partial-failure recovery** — if one image in the batch fails, the whole batch
  raises and stops; already-completed images before the failure aren't returned to
  the gallery. A real gap for large batches, acceptable for a first pass.
- **No mask/inpainting support in batch mode** — batch always runs plain img2img
  (init image only), doesn't expose the Inpainting accordion's controls.
- Still **no color-difference "Inpaint sketch" masking** (unchanged from PHASE17.md).
