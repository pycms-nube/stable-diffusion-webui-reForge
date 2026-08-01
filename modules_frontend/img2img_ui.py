"""
BFISO Phase 12 -- a genuinely torch-free img2img tab.

Mirrors modules_frontend/txt2img_ui.py's structure, sharing its backend-fetch/
progress-streaming/script-control helpers via modules_frontend/common.py. Torch-free
for the same reason as every other modules_frontend file: only gradio and
modules_frontend.common.

Scope, honest: basic img2img only -- a single init image, no inpainting (no mask, no
mask blur, no inpaint_full_res). No Sketch/Inpaint/Inpaint upload/Batch sub-tabs the
shipped UI has; just the plain img2img case. See PHASE12.md for what's deferred.
"""
import functools
import json
import threading
import time
import uuid

import gradio as gr

from modules_frontend.common import (
    build_alwayson_script_controls,
    decode_images,
    encode_image_to_base64,
    fetch_samplers,
    interrupt_generation,
    post_generate,
    skip_current_image,
    stream_progress,
)

RESIZE_MODE_CHOICES = ["Just resize", "Crop and resize", "Resize and fill", "Just resize (latent upscale)"]


def run_img2img(backend_url, script_specs, init_image, prompt, negative_prompt, steps, sampler_name,
                 cfg_scale, width, height, seed, batch_count, batch_size, restore_faces, tiling,
                 denoising_strength, resize_mode, *script_arg_values):
    if init_image is None:
        raise gr.Error("Upload an init image first.")

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


def create_img2img_tab(backend_url):
    """Builds the img2img tab's contents into the enclosing gr.Blocks/gr.Tab context.
    Does not create its own Blocks or call .queue() -- modules_frontend/app.py owns
    both, since they're shared with the txt2img tab."""
    with gr.Row():
        with gr.Column(scale=4):
            init_image = gr.Image(label="Init image", type="pil", source="upload")
            resize_mode = gr.Radio(label="Resize mode", choices=RESIZE_MODE_CHOICES,
                                    type="index", value=RESIZE_MODE_CHOICES[0])

            prompt = gr.Textbox(label="Prompt", lines=3, placeholder="a photo of...")
            negative_prompt = gr.Textbox(label="Negative prompt", lines=2)

            with gr.Row():
                steps = gr.Slider(label="Steps", minimum=1, maximum=150, step=1, value=20)
                cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=30, step=0.5, value=7)

            with gr.Row():
                denoising_strength = gr.Slider(label="Denoising strength", minimum=0.0,
                                                maximum=1.0, step=0.01, value=0.75)

            with gr.Row():
                width = gr.Slider(label="Width", minimum=64, maximum=2048, step=8, value=512)
                height = gr.Slider(label="Height", minimum=64, maximum=2048, step=8, value=512)

            with gr.Row():
                batch_count = gr.Slider(label="Batch count", minimum=1, maximum=50, step=1, value=1)
                batch_size = gr.Slider(label="Batch size", minimum=1, maximum=8, step=1, value=1)

            with gr.Row():
                restore_faces = gr.Checkbox(label="Restore faces", value=False)
                tiling = gr.Checkbox(label="Tiling", value=False)

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

            with gr.Row():
                generate_btn = gr.Button("Generate", variant="primary")
                skip_btn = gr.Button("Skip")
                interrupt_btn = gr.Button("Interrupt", variant="stop")
            progress_box = gr.Textbox(label="Progress", interactive=False)

        with gr.Column(scale=5):
            preview_image = gr.Image(label="Live preview", interactive=False)
            gallery = gr.Gallery(label="Output", show_label=True, columns=2)
            infotext_box = gr.Textbox(label="Generation info", lines=4, interactive=False)

    generate_btn.click(
        # See txt2img_ui.py's identical comment: functools.partial, not a lambda, so
        # Gradio still detects run_img2img's `yield` through the wrapper (PHASE9.md).
        fn=functools.partial(run_img2img, backend_url, script_specs),
        inputs=[init_image, prompt, negative_prompt, steps, sampler_name, cfg_scale, width, height,
                seed, batch_count, batch_size, restore_faces, tiling, denoising_strength, resize_mode,
                *flat_script_inputs],
        outputs=[progress_box, preview_image, gallery, infotext_box],
    )
    skip_btn.click(fn=functools.partial(skip_current_image, backend_url), outputs=[progress_box])
    interrupt_btn.click(fn=functools.partial(interrupt_generation, backend_url), outputs=[progress_box])
