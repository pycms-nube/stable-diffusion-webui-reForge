"""
BFISO Phase 12 -- a genuinely torch-free img2img tab.

Mirrors modules_frontend/txt2img_ui.py's structure, sharing its backend-fetch/
progress-streaming/script-control helpers via modules_frontend/common.py. Torch-free
for the same reason as every other modules_frontend file: only gradio and
modules_frontend.common.

Phase 13 (PHASE13.md) added inpainting: an optional separate mask-image upload plus
mask_blur/inpainting_fill/inpaint_full_res/inpaint_full_res_padding/
inpainting_mask_invert, matching the shipped UI's own "Inpaint upload" sub-tab
semantics.

Phase 14 (PHASE14.md) added the other half: sketch-on-canvas mask drawing, matching
the shipped UI's "Inpaint" sub-tab. A "Mask input" toggle switches between Phase 13's
separate-upload mode and a single gr.Image(tool="sketch") where the mask is drawn
directly over the uploaded image -- Gradio's own preprocessing (gradio/components/
image.py Image.preprocess) already splits that into {"image": ..., "mask": ...} PIL
images before it reaches this module, so no manual canvas decoding was needed.

Phase 17 (PHASE17.md) added the shipped UI's "Sketch" sub-tab equivalent: an "Image
source" toggle (Upload / Paint) alongside the existing "Mask input" toggle. Paint uses
gr.Image(tool="color-sketch"), which -- unlike tool="sketch" -- is NOT split into an
{"image", "mask"} dict by Gradio's preprocessing (confirmed by reading
gradio/components/image.py: the dict-split branch is gated on `self.tool == "sketch"`
specifically, not color-sketch), so it's just a plain painted image used as init_image,
independent of mask handling. Only relevant when "Mask input" is "Upload mask
separately" -- when "Draw mask on image" is selected, that mode already supplies its
own base image via the mask-drawing canvas, so the Image-source toggle hides itself.

Phase 18 (PHASE18.md) added a Batch sub-section: upload multiple images, run the same
generation settings (prompt/steps/sampler/etc. -- the same live components already on
this tab, reused as extra inputs to a second click handler rather than duplicated)
against each one in turn via real /sdapi/v1/img2img calls. The shipped UI's own
"Batch" sub-tab processes a local input_dir/output_dir on the backend's filesystem --
architecturally awkward for a frontend/backend that might not share a filesystem
(the whole point of BFISO), so this uploads and loops instead, with no live progress
mid-image (a plain "Processing N/M" message between calls, not the SSE preview
Generate gets).

Phase 19 (PHASE19.md) added lightweight section headers (Sampling/Size/Batch/
Options/Sampler & Seed), matching txt2img_ui.py's equivalent change -- a purely
visual grouping of the existing controls, no rewiring.

Scope, honest: img2img with mask-upload or sketch-drawn inpainting, Upload/Paint init
image source, upload-multiple-images batch processing, and section headers only (not
the shipped UI's full nested Accordion-per-category layout). No color-difference-based
"Inpaint sketch" masking (distinct from plain color-sketch-as-source). See
PHASE12.md / PHASE13.md / PHASE14.md / PHASE17.md / PHASE18.md for what's deferred.
"""
import functools
import json
import threading
import time
import uuid

import gradio as gr
import requests
from PIL import Image

from modules_frontend.common import (
    build_alwayson_script_controls,
    build_confirm_action_button,
    decode_images,
    encode_image_to_base64,
    fetch_samplers,
    interrupt_generation,
    post_generate,
    skip_current_image,
    stream_progress,
)

RESIZE_MODE_CHOICES = ["Just resize", "Crop and resize", "Resize and fill", "Just resize (latent upscale)"]
MASKED_CONTENT_CHOICES = ["fill", "original", "latent noise", "latent nothing"]
INPAINT_AREA_CHOICES = ["Whole picture", "Only masked"]
MASK_MODE_CHOICES = ["Inpaint masked", "Inpaint not masked"]
MASK_SOURCE_CHOICES = ["Upload mask separately", "Draw mask on image"]
IMAGE_SOURCE_CHOICES = ["Upload", "Paint"]


def _compute_image_control_visibility(mask_mode, image_mode):
    """Both the "Mask input" and "Image source" toggles affect which of
    init_image/color_sketch_canvas/sketch_canvas is actually the live init-image
    control, so both toggles' .change() handlers recompute all of them together
    (rather than each only touching its own component) to keep the two orthogonal
    choices from fighting over the same components' visibility."""
    upload_mask_mode = mask_mode == 0
    return (
        gr.update(visible=upload_mask_mode and image_mode == 0),  # init_image
        gr.update(visible=upload_mask_mode and image_mode == 1),  # color_sketch_canvas
        gr.update(visible=not upload_mask_mode),                  # sketch_canvas (mask draw mode)
        gr.update(visible=upload_mask_mode),                      # mask_upload_group
        gr.update(visible=upload_mask_mode),                      # image_source radio itself
    )


def run_img2img(backend_url, script_specs, init_image, mask_image, mask_source, sketch_value,
                 image_source, color_sketch_image, prompt, negative_prompt, steps, sampler_name,
                 cfg_scale, width, height, seed, batch_count, batch_size, restore_faces, tiling,
                 denoising_strength, resize_mode, mask_blur, inpainting_mask_invert, inpainting_fill,
                 inpaint_full_res, inpaint_full_res_padding, *script_arg_values):
    if mask_source == 1:  # "Draw mask on image"
        if sketch_value is None:
            raise gr.Error("Draw a mask on the image first.")
        init_image, mask_image = sketch_value["image"], sketch_value["mask"]
    elif image_source == 1:  # "Paint" (color-sketch, no mask -- plain image)
        init_image = color_sketch_image

    if init_image is None:
        raise gr.Error("Upload or paint an init image first.")

    alwayson_scripts = {}
    idx = 0
    for name, count in script_specs:
        alwayson_scripts[name] = {"args": list(script_arg_values[idx:idx + count])}
        idx += count

    id_task = f"task(frontend-{uuid.uuid4().hex[:12]})"
    payload = {
        "init_images": [encode_image_to_base64(init_image)],
        "resize_mode": int(resize_mode),
        "denoising_strength": float(denoising_strength),
        "mask_blur": int(mask_blur),
        "inpainting_mask_invert": int(inpainting_mask_invert),
        "inpainting_fill": int(inpainting_fill),
        "inpaint_full_res": bool(inpaint_full_res),
        "inpaint_full_res_padding": int(inpaint_full_res_padding),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": int(steps),
        "sampler_name": sampler_name,
        "cfg_scale": float(cfg_scale),
        "width": int(width),
        "height": int(height),
        "seed": int(seed),
        "n_iter": int(batch_count),
        "batch_size": int(batch_size),
        "restore_faces": bool(restore_faces),
        "tiling": bool(tiling),
        "force_task_id": id_task,
    }
    if mask_image is not None:
        payload["mask"] = encode_image_to_base64(mask_image)
    if alwayson_scripts:
        payload["alwayson_scripts"] = alwayson_scripts

    result_box = {}
    thread = threading.Thread(target=post_generate, args=(backend_url, "/sdapi/v1/img2img", payload, result_box),
                               daemon=True)
    thread.start()

    time.sleep(0.2)
    for progress_text, preview in stream_progress(backend_url, id_task, result_box):
        yield progress_text, preview, gr.update(), gr.update()

    thread.join(timeout=600)

    if result_box.get("error"):
        raise gr.Error(f"Generation request to {backend_url} failed: {result_box['error']}")

    data = result_box.get("response") or {}
    images = decode_images(data.get("images", []))
    info = json.loads(data.get("info", "{}"))
    infotext = (info.get("infotexts") or [""])[0]
    yield "done", None, images, infotext


def run_batch_img2img(backend_url, script_specs, files, prompt, negative_prompt, steps, sampler_name,
                      cfg_scale, width, height, seed, restore_faces, tiling, denoising_strength,
                      resize_mode, *script_arg_values):
    """Sequential, synchronous per-image /sdapi/v1/img2img calls -- no per-image live
    progress (that's what Generate's SSE streaming is for), just a "Processing N/M"
    message between calls. Reuses the same prompt/steps/sampler/etc. components
    already on this tab as extra inputs rather than duplicating a whole second form."""
    if not files:
        raise gr.Error("Upload at least one image for batch processing.")

    alwayson_scripts = {}
    idx = 0
    for name, count in script_specs:
        alwayson_scripts[name] = {"args": list(script_arg_values[idx:idx + count])}
        idx += count

    total = len(files)
    results = []
    yield f"Processing 0/{total}...", results
    for i, file_obj in enumerate(files, start=1):
        image = Image.open(file_obj.name).convert("RGB")
        payload = {
            "init_images": [encode_image_to_base64(image)],
            "resize_mode": int(resize_mode),
            "denoising_strength": float(denoising_strength),
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": int(steps),
            "sampler_name": sampler_name,
            "cfg_scale": float(cfg_scale),
            "width": int(width),
            "height": int(height),
            "seed": int(seed),
            "restore_faces": bool(restore_faces),
            "tiling": bool(tiling),
        }
        if alwayson_scripts:
            payload["alwayson_scripts"] = alwayson_scripts

        try:
            r = requests.post(f"{backend_url}/sdapi/v1/img2img", json=payload, timeout=600)
            r.raise_for_status()
        except requests.RequestException as e:
            raise gr.Error(f"Batch image {i}/{total} failed: {e}") from e

        results.extend(decode_images(r.json().get("images", [])))
        yield f"Processing {i}/{total}...", results

    yield f"Batch done: {total} image(s) processed.", results


def create_img2img_tab(backend_url):
    """Builds the img2img tab's contents into the enclosing gr.Blocks/gr.Tab context.
    Does not create its own Blocks or call .queue() -- modules_frontend/app.py owns
    both, since they're shared with the txt2img tab."""
    with gr.Row():
        with gr.Column(scale=4):
            image_source = gr.Radio(label="Image source", choices=IMAGE_SOURCE_CHOICES,
                                     type="index", value=IMAGE_SOURCE_CHOICES[0])
            init_image = gr.Image(label="Init image", type="pil", source="upload")
            color_sketch_canvas = gr.Image(label="Paint the init image", type="pil",
                                            source="canvas", tool="color-sketch", visible=False)
            resize_mode = gr.Radio(label="Resize mode", choices=RESIZE_MODE_CHOICES,
                                    type="index", value=RESIZE_MODE_CHOICES[0])

            with gr.Accordion("Inpainting", open=False):
                mask_source = gr.Radio(label="Mask input", choices=MASK_SOURCE_CHOICES,
                                        type="index", value=MASK_SOURCE_CHOICES[0])
                with gr.Group(visible=True) as mask_upload_group:
                    mask_image = gr.Image(label="Mask (white = inpaint)", type="pil",
                                           source="upload", image_mode="L")
                sketch_canvas = gr.Image(label="Draw mask directly on the image", type="pil",
                                          source="upload", tool="sketch", visible=False)
                mask_source.change(
                    fn=_compute_image_control_visibility,
                    inputs=[mask_source, image_source],
                    outputs=[init_image, color_sketch_canvas, sketch_canvas, mask_upload_group, image_source],
                )
                image_source.change(
                    fn=_compute_image_control_visibility,
                    inputs=[mask_source, image_source],
                    outputs=[init_image, color_sketch_canvas, sketch_canvas, mask_upload_group, image_source],
                )
                mask_blur = gr.Slider(label="Mask blur", minimum=0, maximum=64, step=1, value=4)
                inpainting_mask_invert = gr.Radio(label="Mask mode", choices=MASK_MODE_CHOICES,
                                                   type="index", value=MASK_MODE_CHOICES[0])
                inpainting_fill = gr.Radio(label="Masked content", choices=MASKED_CONTENT_CHOICES,
                                            type="index", value="original")
                inpaint_full_res = gr.Radio(label="Inpaint area", choices=INPAINT_AREA_CHOICES,
                                             type="index", value=INPAINT_AREA_CHOICES[0])
                inpaint_full_res_padding = gr.Slider(label="Only masked padding, pixels", minimum=0,
                                                      maximum=256, step=4, value=32)

            prompt = gr.Textbox(label="Prompt", lines=3, placeholder="a photo of...")
            negative_prompt = gr.Textbox(label="Negative prompt", lines=2)

            gr.Markdown("#### Sampling")
            with gr.Row():
                steps = gr.Slider(label="Steps", minimum=1, maximum=150, step=1, value=20)
                cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=30, step=0.5, value=7)

            with gr.Row():
                denoising_strength = gr.Slider(label="Denoising strength", minimum=0.0,
                                                maximum=1.0, step=0.01, value=0.75)

            gr.Markdown("#### Size")
            with gr.Row():
                width = gr.Slider(label="Width", minimum=64, maximum=2048, step=8, value=512)
                height = gr.Slider(label="Height", minimum=64, maximum=2048, step=8, value=512)

            gr.Markdown("#### Batch")
            with gr.Row():
                batch_count = gr.Slider(label="Batch count", minimum=1, maximum=50, step=1, value=1)
                batch_size = gr.Slider(label="Batch size", minimum=1, maximum=8, step=1, value=1)

            gr.Markdown("#### Options")
            with gr.Row():
                restore_faces = gr.Checkbox(label="Restore faces", value=False)
                tiling = gr.Checkbox(label="Tiling", value=False)

            gr.Markdown("#### Sampler & Seed")
            with gr.Row():
                try:
                    sampler_choices = fetch_samplers(backend_url)
                except RuntimeError as e:
                    sampler_choices = []
                    gr.Markdown(f"⚠️ {e}")
                sampler_name = gr.Dropdown(label="Sampler", choices=sampler_choices,
                                            value=sampler_choices[0] if sampler_choices else None)
                seed = gr.Number(label="Seed", value=-1, precision=0)

            with gr.Accordion("Scripts", open=False):
                script_controls = build_alwayson_script_controls(backend_url, is_img2img=True)
            script_specs = [(name, len(controls)) for name, controls in script_controls]
            flat_script_inputs = [c for _name, controls in script_controls for c in controls]

            with gr.Accordion("Batch (multiple images)", open=False):
                gr.Markdown(
                    "Runs the current prompt/steps/sampler/etc. settings above against every "
                    "uploaded image in turn via real /sdapi/v1/img2img calls -- no shared-filesystem "
                    "input_dir/output_dir like the shipped UI's Batch sub-tab, since frontend and "
                    "backend may not be on the same machine."
                )
                batch_files = gr.File(label="Images to process", file_count="multiple")
                batch_run_btn = gr.Button("Run Batch")
                batch_progress_box = gr.Textbox(label="Batch progress", interactive=False)
                batch_gallery = gr.Gallery(label="Batch output", show_label=True, columns=2)

            progress_box = gr.Textbox(label="Progress", interactive=False)
            with gr.Row():
                generate_btn = gr.Button("Generate", variant="primary")
                # build_confirm_action_button() builds AND wires Skip/Interrupt itself
                # (two-click confirm, PHASE16.md) -- progress_box must exist before
                # this call since it's used as the confirm message's output target.
                skip_btn = build_confirm_action_button(backend_url, "Skip", skip_current_image, progress_box)
                interrupt_btn = build_confirm_action_button(backend_url, "Interrupt", interrupt_generation,
                                                             progress_box, variant="stop")

        with gr.Column(scale=5):
            preview_image = gr.Image(label="Live preview", interactive=False)
            gallery = gr.Gallery(label="Output", show_label=True, columns=2)
            infotext_box = gr.Textbox(label="Generation info", lines=4, interactive=False)

    generate_btn.click(
        # See txt2img_ui.py's identical comment: functools.partial, not a lambda, so
        # Gradio still detects run_img2img's `yield` through the wrapper (PHASE9.md).
        fn=functools.partial(run_img2img, backend_url, script_specs),
        inputs=[init_image, mask_image, mask_source, sketch_canvas, image_source, color_sketch_canvas,
                prompt, negative_prompt, steps, sampler_name, cfg_scale, width, height, seed,
                batch_count, batch_size, restore_faces, tiling, denoising_strength, resize_mode,
                mask_blur, inpainting_mask_invert, inpainting_fill, inpaint_full_res,
                inpaint_full_res_padding, *flat_script_inputs],
        outputs=[progress_box, preview_image, gallery, infotext_box],
    )
    batch_run_btn.click(
        # Same functools.partial-not-lambda reasoning as generate_btn.click above --
        # run_batch_img2img is a generator too (PHASE9.md).
        fn=functools.partial(run_batch_img2img, backend_url, script_specs),
        inputs=[batch_files, prompt, negative_prompt, steps, sampler_name, cfg_scale, width, height,
                seed, restore_faces, tiling, denoising_strength, resize_mode, *flat_script_inputs],
        outputs=[batch_progress_box, batch_gallery],
    )
