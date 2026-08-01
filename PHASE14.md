# PHASE14 — Sketch-on-Canvas Mask Drawing

Closes PHASE13.md's named remaining gap: the shipped UI's "Inpaint" sub-tab (draw a
mask directly over the image) alongside Phase 13's "Inpaint upload" sub-tab
equivalent (separate mask image).

## What changed

- New **"Mask input"** `gr.Radio` toggle inside the Inpainting accordion:
  `["Upload mask separately", "Draw mask on image"]`, `type="index"`.
- New `sketch_canvas = gr.Image(type="pil", tool="sketch", source="upload",
  visible=False)` — a single component that's both the base image and the drawing
  surface. Toggling the radio flips visibility three ways via one `.change()` handler:
  the Phase 13 mask-upload group, this canvas, and (importantly) the main "Init image"
  field itself, since in draw mode the canvas *is* the init image — showing both would
  be confusing and wrong (two different "which image is actually used" answers).
- `run_img2img()`: when `mask_source == 1`, `init_image`/`mask_image` are overridden
  from `sketch_value["image"]` / `sketch_value["mask"]` before building the payload.
  No manual canvas/base64 decoding needed — confirmed by reading
  `gradio/components/image.py`'s own `Image.preprocess()`: when `tool="sketch"` and
  `type="pil"`, Gradio's own preprocessing already splits the frontend's drawn-canvas
  payload into `{"image": PIL, "mask": PIL}` (with the mask's alpha channel already
  converted to a plain white/black RGB image) before the value ever reaches this
  module's function. Reading the source here was substituted for read-then-implement
  since actually drawing on a canvas isn't automatable through this session's tools —
  the format claim is a direct code citation, not a guess.

## Verification

**Backend contract + dict-unpacking logic, called directly (not via curl this time,
since there's no HTTP boundary to synthesize for a canvas draw)**: called
`run_img2img()` itself as a plain Python generator with `mask_source=1` and a
synthetic `sketch_value = {"image": <PIL>, "mask": <PIL>}` matching exactly what
Gradio's own preprocessing produces — consumed the generator to its final yield and
got back a real image plus infotext showing `Mask blur: 4`, `Inpaint area: Only
masked`, `Masked area padding: 32`, confirming the extraction and payload-building
logic is correct end-to-end against the real backend, with `init_image`/`mask_image`
both passed as `None` (as they would be with the upload fields hidden) to prove the
override actually happens rather than silently falling through.

**Import/build check**: `create_ui()` builds cleanly with the new toggle/canvas/
`.change()` wiring, no exceptions.

**Live UI — the one thing that genuinely needed a human**: drawing on a canvas isn't
something this session's tools can automate or fake. User manually confirmed: toggled
to "Draw mask on image," drew a mask over an uploaded image, generated, and the drawn
area was repainted while the rest of the image stayed close to the original.

## What this phase did NOT do

- **No "Sketch" sub-tab** (color-sketch used as the img2img *source* image itself,
  distinct from inpaint-sketch's mask-only drawing) — still out of scope.
- **No Batch sub-tab.**
- **No brush size / color controls exposed** — Gradio's sketch tool has its own
  built-in brush UI; this phase didn't add any `brush_radius`/`brush_color` overrides
  beyond gr.Image's defaults.
- Still **flat layout**, same tradeoff as every prior phase.
