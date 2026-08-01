# PHASE12 — Basic img2img Tab

Adds a second pipeline to the standalone frontend: img2img, alongside txt2img in a
`gr.Tabs()` layout. The biggest remaining gap named across PHASE8-11.md.

## Refactor: extracting modules_frontend/common.py

img2img needs almost everything txt2img already had — sampler/upscaler/script-info
fetching, progress SSE streaming, interrupt/skip, image decoding. Rather than
duplicate all of it into a second file, extracted the backend-agnostic helpers into
`modules_frontend/common.py`:

- `fetch_samplers`, `fetch_hr_upscalers`, `fetch_script_info`
- `decode_images`, `encode_image_to_base64` (new — img2img needs to go the other
  direction, PIL image the user uploaded → base64 for the request payload),
  `_decode_data_uri`
- `build_alwayson_script_controls(backend_url, is_img2img)` — generalized with an
  `is_img2img` flag instead of the hardcoded `not s.get("is_img2img")` filter, since
  scripts.py reports each script's applicable pipeline(s) and img2img has its own
  distinct alwayson script set (e.g. no Hires-fix-adjacent scripts, but its own
  inpainting-adjacent ones)
- `interrupt_generation`, `skip_current_image`, `post_generate`, `stream_progress`

`modules_frontend/txt2img_ui.py`'s `create_ui()` became `create_txt2img_tab()` — no
longer owns its own `gr.Blocks()`/`.queue()`, since those are now shared with img2img.
New `modules_frontend/app.py` assembles both tabs into one `gr.Blocks()` +
`gr.Tabs()` and owns the single `demo.queue(concurrency_count=3)` call (PHASE10.md's
fix, now needs to cover both tabs' Skip/Interrupt buttons, not just one). This is a
pure structural refactor — txt2img's own behavior is unchanged, verified below.

## New: modules_frontend/img2img_ui.py

- `gr.Image(type="pil")` for the init image upload, `gr.Radio(..., type="index")` for
  resize mode (matching the shipped UI's own `modules/ui.py:629` — `type="index"`
  means Gradio hands back the choice's list position as an int, exactly the
  `resize_mode` field's expected 0-3 range, not a string needing translation).
- Same core params as txt2img (steps/sampler/cfg/size/batch/seed/restore
  faces/tiling), plus `denoising_strength` (img2img's own, not shared with the
  Hires. fix one on the txt2img tab).
- `run_img2img()` mirrors `run_txt2img()`'s generator/thread/SSE structure exactly,
  posting to `/sdapi/v1/img2img` instead of `/sdapi/v1/txt2img`.

**Scope, honest**: basic img2img only. No inpainting (no mask, no mask blur, no
inpaint_full_res) and no Sketch/Inpaint/Inpaint-upload/Batch sub-tabs the shipped UI
has under img2img — just the plain "transform this whole image" case.

## Verification

**Backend contract, single one-shot `curl` check** (not a loop): built a tiny 64x64
red-square PNG, base64-encoded it exactly as `encode_image_to_base64()` does, POSTed
to `/sdapi/v1/img2img` with the same field names/shapes `run_img2img()` sends —
`200`, real infotext returned (`Denoising strength: 0.5`, correct size/steps),
confirming the payload shape this frontend builds is exactly what the backend expects
before ever touching a browser.

**Import/build check**: `create_ui()` from the new `modules_frontend/app.py` actually
builds a `gr.Blocks` against the live backend (fetches real sampler/upscaler/script
lists for both tabs) with no exceptions. Torch-freedom re-confirmed by import
inspection across all four `modules_frontend/*.py` files plus `webui_frontend.py` --
only stdlib, `gradio`, `requests`, `PIL`, and `modules.ui_script_schema` (itself
torch-free per PHASE4.md).

**Live UI, both tabs**: handed to the user for manual confirmation per the Phase
10-established methodology. Confirmed: txt2img tab unaffected by the refactor,
img2img tab's upload → Generate flow works end to end.

## What this phase did NOT do

- **No inpainting** — no mask upload, no mask blur, no inpaint_full_res/padding, no
  masked-content fill mode. The single biggest remaining img2img gap.
- **No Sketch / Inpaint sketch / Inpaint upload / Batch sub-tabs.**
- **No resize-by vs resize-to toggle** — img2img's width/height sliders always act as
  an absolute target size (like `resize_mode` implies), not the shipped UI's separate
  "Resize by scale factor" convenience tab.
- Still **flat layout**, same tradeoff as every prior phase.
