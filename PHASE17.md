# PHASE17 — Paint (Color-Sketch) Image Source

Closes the "no Sketch sub-tab" gap: an "Image source" toggle (Upload / Paint) lets
img2img's init image come from a painted canvas instead of an uploaded file,
matching the shipped UI's "Sketch" sub-tab.

## Design note: color-sketch is simpler than sketch

`tool="color-sketch"` was confirmed (by reading `gradio/components/image.py`) to
**not** trigger Gradio's image/mask dict-split the way `tool="sketch"` does — that
split is gated specifically on `self.tool == "sketch"`, not color-sketch. So Paint
mode's `color_sketch_canvas` behaves like a completely ordinary `gr.Image(type="pil")`
from this module's perspective: whatever's painted arrives as a plain PIL image, no
manual unpacking needed, same as a regular upload. This makes it independent of the
Phase 14 mask-drawing toggle rather than needing to compose with it.

## What changed

- New `image_source = gr.Radio(["Upload", "Paint"], type="index")` and
  `color_sketch_canvas = gr.Image(type="pil", tool="color-sketch", source="canvas",
  visible=False)`.
- `_compute_image_control_visibility(mask_mode, image_mode)` — both the "Mask input"
  (Phase 14) and "Image source" (this phase) toggles now recompute the same five
  components' visibility together (`init_image`, `color_sketch_canvas`,
  `sketch_canvas`, `mask_upload_group`, and `image_source` itself), instead of each
  toggle only touching its own components. Necessary because the two toggles are
  orthogonal but share `init_image`'s visibility: "Draw mask on image" mode already
  supplies its own base image via the mask-drawing canvas, so in that mode the
  Image-source toggle is irrelevant and hides itself entirely (set `visible=False`) so
  there's no dangling "Upload / Paint" choice that does nothing.
- `run_img2img()`: when `mask_source != 1` (not in draw-mask mode) and
  `image_source == 1`, `init_image` is overridden from `color_sketch_canvas`'s plain
  PIL value before building the payload.

## Verification

**Direct call against the real backend**: called `run_img2img()` with `init_image=None`
and `image_source=1` plus a synthetic painted PIL image — completed successfully
(if the override hadn't fired, the function's own `init_image is None` guard would
have raised "Upload or paint an init image first."), confirming the override path
genuinely supplies the image rather than silently falling through to the missing
`init_image`.

**Build check**: `create_ui()` builds cleanly with the new toggle/canvas/visibility
wiring, no exceptions.

**Live UI**: bundled into the Phase 16-19 consolidated review, per the user's
instruction to implement this batch together and check once at the end.

## What this phase did NOT do

- **No color-difference "Inpaint sketch" masking** — the shipped UI has a distinct
  mode that derives a mask from where you've drawn colored strokes over an existing
  image (comparing original vs. sketched pixels). This phase's Paint mode is purely
  "paint a fresh image from scratch as the img2img source," not that derived-mask
  workflow — genuinely a different feature, not implemented here.
- **Combining "Paint" source with mask-upload inpainting untested in combination**
  — the visibility logic supports it (Paint + "Upload mask separately" should both be
  visible together), but this specific combination wasn't separately exercised beyond
  the isolated direct-call test above.
