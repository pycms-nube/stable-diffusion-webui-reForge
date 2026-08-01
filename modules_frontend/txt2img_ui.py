"""
BFISO Phase 8/9/10/11 -- a genuinely torch-free txt2img tab.

This module (and modules_frontend/app.py + webui_frontend.py, which assemble and
launch it) is part of the actual cutover proof: everything up to now (PHASE0-7.md)
built and verified the pieces, but the shipped Gradio app (modules/ui.py) still runs
fully in-process and still needs torch, because it imports
modules.scripts/modules.processing/modules.sd_* directly (PHASE3.md). This module
deliberately imports NONE of that -- only gradio and modules_frontend.common (itself
torch-free) -- and drives generation entirely over HTTP against a separately-running
backend (webui-backend.sh or the full app's --api mode).

Phase 9 (PHASE9.md): script control values sent with the request, live progress+preview
via Phase 5's SSE stream and the request's force_task_id field.
Phase 10 (PHASE10.md): Interrupt/Skip, batch count/size, restore faces, tiling.
Phase 11 (PHASE11.md): Hires. fix.
Phase 12 (PHASE12.md): extracted shared helpers into modules_frontend/common.py so
modules_frontend/img2img_ui.py could reuse them; create_ui() split into
create_txt2img_tab() (this file, no longer owns its own gr.Blocks) assembled by the new
modules_frontend/app.py alongside the new img2img tab.

Phase 15 (PHASE15.md): the six advanced Hires. fix overrides (hr_checkpoint_name/
hr_sampler_name/hr_scheduler/hr_prompt/hr_negative_prompt/hr_cfg). Unlike the REST
API's *main* sampler_name/scheduler fields, hr_sampler_name/hr_scheduler get no
"Use same X" sentinel-to-None normalization inside the API path (modules/api/api.py's
text2imgapi) -- only the Gradio-only modules/txt2img.py::txt2img_create_processing()
does that translation, and this module never goes through it. run_txt2img() replicates
that translation client-side (see its own comment) so picking the sentinel default
behaves identically to leaving these fields unset, rather than sending a literal
"Use same sampler" string a sampler lookup would reject.

Scope, still honest: flat layout (no accordion/category sectioning matching the
shipped UI).
"""
import functools
import json
import threading
import time
import uuid

import gradio as gr

from modules_frontend.common import (
    build_alwayson_script_controls,
    build_confirm_action_button,
    decode_images,
    fetch_hr_upscalers,
    fetch_samplers,
    fetch_sd_models,
    fetch_schedulers,
    interrupt_generation,
    post_generate,
    skip_current_image,
    stream_progress,
)

HR_CHECKPOINT_SENTINEL = "Use same checkpoint"
HR_SAMPLER_SENTINEL = "Use same sampler"
HR_SCHEDULER_SENTINEL = "Use same scheduler"


def run_txt2img(backend_url, script_specs, prompt, negative_prompt, steps, sampler_name,
                 cfg_scale, width, height, seed, batch_count, batch_size, restore_faces,
                 tiling, enable_hr, hr_scale, hr_upscaler, hr_second_pass_steps,
                 denoising_strength, hr_resize_x, hr_resize_y, hr_checkpoint_name,
                 hr_sampler_name, hr_scheduler, hr_prompt, hr_negative_prompt, hr_cfg,
                 *script_arg_values):
    alwayson_scripts = {}
    idx = 0
    for name, count in script_specs:
        alwayson_scripts[name] = {"args": list(script_arg_values[idx:idx + count])}
        idx += count

    id_task = f"task(frontend-{uuid.uuid4().hex[:12]})"
    payload = {
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
        "enable_hr": bool(enable_hr),
        "hr_scale": float(hr_scale),
        "hr_upscaler": hr_upscaler,
        "hr_second_pass_steps": int(hr_second_pass_steps),
        "denoising_strength": float(denoising_strength),
        "hr_resize_x": int(hr_resize_x),
        "hr_resize_y": int(hr_resize_y),
        # The REST API path (modules/api/api.py::text2imgapi) does NOT translate the
        # "Use same X" dropdown sentinels to None the way the Gradio-only
        # modules/txt2img.py::txt2img_create_processing() does -- only that
        # Gradio-path wrapper performs this conversion, and this frontend never goes
        # through it. Replicate the translation here so the default (unchanged)
        # dropdown value behaves as "use the base pass's value" instead of a literal
        # sampler/scheduler lookup for a name that doesn't exist.
        "hr_checkpoint_name": None if hr_checkpoint_name == HR_CHECKPOINT_SENTINEL else hr_checkpoint_name,
        "hr_sampler_name": None if hr_sampler_name == HR_SAMPLER_SENTINEL else hr_sampler_name,
        "hr_scheduler": None if hr_scheduler == HR_SCHEDULER_SENTINEL else hr_scheduler,
        "hr_prompt": hr_prompt,
        "hr_negative_prompt": hr_negative_prompt,
        "hr_cfg": float(hr_cfg),
        "force_task_id": id_task,
    }
    if alwayson_scripts:
        payload["alwayson_scripts"] = alwayson_scripts

    result_box = {}
    thread = threading.Thread(target=post_generate, args=(backend_url, "/sdapi/v1/txt2img", payload, result_box),
                               daemon=True)
    thread.start()

    # Give the backend a moment to register the task before we start polling for it,
    # then stream progress until the POST thread reports done.
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


def create_txt2img_tab(backend_url):
    """Builds the txt2img tab's contents into the enclosing gr.Blocks/gr.Tab context.
    Does not create its own Blocks or call .queue() -- modules_frontend/app.py owns
    both, since they're shared with the img2img tab."""
    with gr.Row():
        with gr.Column(scale=4):
            prompt = gr.Textbox(label="Prompt", lines=3, placeholder="a photo of...")
            negative_prompt = gr.Textbox(label="Negative prompt", lines=2)

            with gr.Row():
                steps = gr.Slider(label="Steps", minimum=1, maximum=150, step=1, value=20)
                cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=30, step=0.5, value=7)

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

            with gr.Accordion("Hires. fix", open=False):
                enable_hr = gr.Checkbox(label="Enable Hires. fix", value=False)
                with gr.Row():
                    try:
                        hr_upscaler_choices = fetch_hr_upscalers(backend_url)
                    except RuntimeError as e:
                        hr_upscaler_choices = []
                        gr.Markdown(f"⚠️ {e}")
                    hr_upscaler = gr.Dropdown(label="Upscaler", choices=hr_upscaler_choices,
                                               value=hr_upscaler_choices[0] if hr_upscaler_choices else None)
                    hr_second_pass_steps = gr.Slider(label="Hires steps", minimum=0, maximum=150,
                                                      step=1, value=0)
                with gr.Row():
                    hr_scale = gr.Slider(label="Upscale by", minimum=1.0, maximum=4.0, step=0.05, value=2.0)
                    denoising_strength = gr.Slider(label="Denoising strength", minimum=0.0,
                                                    maximum=1.0, step=0.01, value=0.75)
                with gr.Row():
                    hr_resize_x = gr.Number(label="Resize width to (0 = use Upscale by)", value=0, precision=0)
                    hr_resize_y = gr.Number(label="Resize height to (0 = use Upscale by)", value=0, precision=0)

                with gr.Accordion("Advanced Hires. fix overrides", open=False):
                    with gr.Row():
                        try:
                            hr_checkpoint_choices = [HR_CHECKPOINT_SENTINEL] + fetch_sd_models(backend_url)
                        except RuntimeError as e:
                            hr_checkpoint_choices = [HR_CHECKPOINT_SENTINEL]
                            gr.Markdown(f"⚠️ {e}")
                        hr_checkpoint_name = gr.Dropdown(label="Hires checkpoint", choices=hr_checkpoint_choices,
                                                          value=HR_CHECKPOINT_SENTINEL)
                        hr_sampler_name = gr.Dropdown(label="Hires sampling method",
                                                       choices=[HR_SAMPLER_SENTINEL] + sampler_choices,
                                                       value=HR_SAMPLER_SENTINEL)
                    with gr.Row():
                        try:
                            hr_scheduler_choices = [HR_SCHEDULER_SENTINEL] + fetch_schedulers(backend_url)
                        except RuntimeError as e:
                            hr_scheduler_choices = [HR_SCHEDULER_SENTINEL]
                            gr.Markdown(f"⚠️ {e}")
                        hr_scheduler = gr.Dropdown(label="Hires schedule type", choices=hr_scheduler_choices,
                                                    value=HR_SCHEDULER_SENTINEL)
                        hr_cfg = gr.Slider(label="Hires CFG Scale", minimum=0.0, maximum=30.0, step=0.1, value=0.0)
                    hr_prompt = gr.Textbox(label="Hires prompt", lines=2,
                                            placeholder="Leave empty to use the same prompt as the base pass.")
                    hr_negative_prompt = gr.Textbox(label="Hires negative prompt", lines=2,
                                                     placeholder="Leave empty to use the same negative "
                                                                  "prompt as the base pass.")

            with gr.Accordion("Scripts", open=False):
                script_controls = build_alwayson_script_controls(backend_url, is_img2img=False)
            script_specs = [(name, len(controls)) for name, controls in script_controls]
            flat_script_inputs = [c for _name, controls in script_controls for c in controls]

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
        # functools.partial (not a lambda) so Gradio's inspect.isgeneratorfunction(fn)
        # check still sees run_txt2img's `yield` through the wrapper -- a lambda
        # wrapping a generator function is itself NOT a generator function, since
        # calling it just returns a generator object rather than yielding, and Gradio
        # silently expects a single return value in that case (PHASE9.md).
        fn=functools.partial(run_txt2img, backend_url, script_specs),
        inputs=[prompt, negative_prompt, steps, sampler_name, cfg_scale, width, height, seed,
                batch_count, batch_size, restore_faces, tiling, enable_hr, hr_scale, hr_upscaler,
                hr_second_pass_steps, denoising_strength, hr_resize_x, hr_resize_y,
                hr_checkpoint_name, hr_sampler_name, hr_scheduler, hr_prompt, hr_negative_prompt,
                hr_cfg, *flat_script_inputs],
        outputs=[progress_box, preview_image, gallery, infotext_box],
    )
