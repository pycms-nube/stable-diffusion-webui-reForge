# PHASE13 — Mask-Upload Inpainting

Closes PHASE12.md's named "single biggest remaining img2img gap": inpainting.
`modules_frontend/img2img_ui.py` gets a separate mask-image upload plus the five
standard inpainting sub-params, matching the shipped UI's "Inpaint upload" sub-tab
semantics.

## What changed

- New "Inpainting" `gr.Accordion` on the img2img tab: a second `gr.Image(type="pil",
  image_mode="L")` for the mask (white = inpaint, black = keep — a plain second
  upload, not a sketch canvas drawn over the init image), plus:
  - `mask_blur` (0-64, default 4)
  - `inpainting_mask_invert` — Radio `["Inpaint masked", "Inpaint not masked"]`,
    `type="index"` (matches the shipped UI's own `modules/ui.py:710` exactly)
  - `inpainting_fill` — Radio `["fill", "original", "latent noise", "latent
    nothing"]`, `type="index"`, default `"original"` (`modules/ui.py:713`)
  - `inpaint_full_res` — Radio `["Whole picture", "Only masked"]`, `type="index"`
    (`modules/ui.py:717`); backend field is typed `bool` but the shipped UI itself
    feeds it a Radio index (0/1), which Python accepts fine as truthy/falsy — matched
    that exact pattern rather than inventing a different one
  - `inpaint_full_res_padding` (0-256, default 32)
- `run_img2img()`: only adds `"mask"` to the payload when a mask image was actually
  uploaded (mirrors `modules/api/api.py::img2imgapi`'s own `if mask:` check) — a plain
  img2img request without a mask is unaffected by this phase.

## Verification

**Backend contract, single one-shot `curl` check**: built a synthetic 64x64 init
image and a mask with a filled rectangle, base64-encoded both exactly as
`encode_image_to_base64()` does, POSTed to `/sdapi/v1/img2img` with the same field
names/shapes `run_img2img()` sends — `200`, and the returned infotext included
`Mask blur: 4`, `Inpaint area: Only masked`, `Masked area padding: 32`, confirming
inpainting genuinely activated (these fields only appear in infotext when a mask is
present and processed) rather than the mask being silently ignored.

**Import/build check**: `create_ui()` builds cleanly against the live backend with the
new controls added, no exceptions.

**Live UI**: user manually confirmed — uploaded an init image and a separate mask,
generated, and the masked region changed while the rest of the image stayed close to
the original.

## What this phase did NOT do

- **No sketch-on-canvas mask drawing** — the shipped UI's "Inpaint" and "Inpaint
  sketch" sub-tabs let you paint a mask directly over the init image in the browser.
  This phase only supports uploading a pre-made separate mask image (the "Inpaint
  upload" sub-tab's mechanism), which is enough to exercise the real inpainting
  pipeline but not the most convenient workflow for freehand masking.
- **No Batch sub-tab** (processing a folder of images).
- **`mask_alpha` / color-sketch-specific fields not exposed** — the shipped UI has a
  few sketch-canvas-only controls (`mask_alpha`, hidden by default even there) that
  don't apply to plain mask-upload inpainting.
- Still **flat layout**, same tradeoff as every prior phase.
