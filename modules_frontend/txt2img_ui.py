"""
BFISO Phase 8/9 -- a genuinely torch-free txt2img UI.

This module (and webui_frontend.py, which launches it) is the actual cutover proof:
everything up to now (PHASE0-7.md) built and verified the pieces, but the shipped
Gradio app (modules/ui.py) still runs fully in-process and still needs torch, because
it imports modules.scripts/modules.processing/modules.sd_* directly (PHASE3.md). This
module deliberately imports NONE of that -- only gradio, requests, PIL, and
modules.ui_script_schema (itself torch-free, see PHASE4.md) -- and drives generation
entirely over HTTP against a separately-running backend (webui-backend.sh or the full
app's --api mode).

Phase 9 (see PHASE9.md) closed the two biggest gaps Phase 8 named explicitly: script
control values are now sent with the request (alwayson_scripts), and generation shows
live progress + preview via Phase 5's SSE stream, using the existing force_task_id
field on the txt2img request so the frontend can pick its own task id up front and
poll for it while the (blocking) POST runs in a background thread.

Phase 10 (see PHASE10.md) added Interrupt/Skip controls (thin wrappers over
/sdapi/v1/interrupt and /sdapi/v1/skip -- both fire-and-forget, no generation-state
tracking needed here since shared.state lives entirely backend-side) and the batch
count/size + restore faces/tiling params that were missing from the request payload
even though the UI never exposed them.

Phase 11 (see PHASE11.md) added Hires. fix: enable_hr plus its five core sub-params
(hr_scale, hr_upscaler, hr_second_pass_steps, denoising_strength, hr_resize_x/y),
tucked into their own Accordion so the base form doesn't grow when unused. The
upscaler dropdown is populated from the backend's own /sdapi/v1/latent-upscale-modes
+ /sdapi/v1/upscalers, the same two lists the real hr_upscaler validation in
processing.py checks against.

Scope, still honest: txt2img only, core params only. No img2img/other tabs, layout is
flat (no conditional visibility -- Hires params always render, matching them being
sent unconditionally too; see PHASE11.md), no hr_checkpoint_name/hr_sampler_name/
hr_scheduler/hr_prompt/hr_negative_prompt/hr_cfg (advanced hires overrides, left at
their None/same-as-base defaults).
"""
import base64
import functools
import io
import json
import threading
import time
import uuid

import gradio as gr
import requests
from PIL import Image

from modules.ui_script_schema import build_controls_from_schema

DEFAULT_BACKEND_URL = "http://127.0.0.1:7860"


def _get(backend_url, path, timeout=10):
    r = requests.get(f"{backend_url}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_samplers(backend_url):
    try:
        return [s["name"] for s in _get(backend_url, "/sdapi/v1/samplers")]
    except requests.RequestException as e:
        raise RuntimeError(f"Could not reach backend at {backend_url} for /sdapi/v1/samplers: {e}") from e


def fetch_hr_upscalers(backend_url):
    """hr_upscaler is validated backend-side (modules/processing.py) against the union
    of shared.latent_upscale_modes and shared.sd_upscalers -- so the dropdown has to
    offer exactly that union, not just one or the other, or a legal choice would 422."""
    try:
        latent_modes = [m["name"] for m in _get(backend_url, "/sdapi/v1/latent-upscale-modes")]
        upscalers = [u["name"] for u in _get(backend_url, "/sdapi/v1/upscalers")]
    except requests.RequestException as e:
        raise RuntimeError(f"Could not reach backend at {backend_url} for upscaler lists: {e}") from e
    return latent_modes + upscalers


def fetch_script_info(backend_url):
    try:
        return _get(backend_url, "/sdapi/v1/script-info")
    except requests.RequestException as e:
        raise RuntimeError(f"Could not reach backend at {backend_url} for /sdapi/v1/script-info: {e}") from e


def decode_images(b64_list):
    return [Image.open(io.BytesIO(base64.b64decode(b64))) for b64 in b64_list]


def _decode_data_uri(data_uri):
    _header, b64data = data_uri.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(b64data)))


def build_alwayson_script_controls(backend_url):
    """Renders every alwayson script's controls, schema-built (PHASE4.md), inside its
    own Accordion. Returns [(script_name, [gr.Component, ...]), ...] in the same order
    the backend's own script list expects args in -- required so run_txt2img can
    rebuild the alwayson_scripts payload on submit (see module docstring)."""
    try:
        script_info = fetch_script_info(backend_url)
    except RuntimeError as e:
        gr.Markdown(f"⚠️ Could not load script list: {e}")
        return []

    alwayson = [s for s in script_info if s.get("is_alwayson") and not s.get("is_img2img") and s.get("args")]
    if not alwayson:
        gr.Markdown("(no always-on txt2img scripts reported by the backend)")
        return []

    script_controls = []
    for script in alwayson:
        with gr.Accordion(script["name"], open=False):
            controls = build_controls_from_schema(script["args"])
        script_controls.append((script["name"], controls))
    return script_controls


def _post(backend_url, path, timeout=10):
    r = requests.post(f"{backend_url}{path}", timeout=timeout)
    r.raise_for_status()
    return r


def interrupt_generation(backend_url):
    try:
        _post(backend_url, "/sdapi/v1/interrupt")
        return "Interrupt requested."
    except requests.RequestException as e:
        raise gr.Error(f"Interrupt request to {backend_url} failed: {e}") from e


def skip_current_image(backend_url):
    try:
        _post(backend_url, "/sdapi/v1/skip")
        return "Skip requested."
    except requests.RequestException as e:
        raise gr.Error(f"Skip request to {backend_url} failed: {e}") from e


def _post_txt2img(backend_url, payload, result_box):
    try:
        r = requests.post(f"{backend_url}/sdapi/v1/txt2img", json=payload, timeout=600)
        r.raise_for_status()
        result_box["response"] = r.json()
    except requests.RequestException as e:
        result_box["error"] = e
    finally:
        result_box["done"] = True


def _stream_progress(backend_url, id_task, result_box):
    """Yields (progress_text, preview_image_or_None) until result_box['done'] is set
    by the background POST thread. A best-effort display, not the source of truth for
    the actual result -- the POST response is."""
    url = f"{backend_url}/internal/progress-stream"
    params = {"id_task": id_task, "live_preview": "true"}
    try:
        with requests.get(url, params=params, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if result_box.get("done"):
                    break
                if not line or not line.startswith("data:"):
                    continue
                event = json.loads(line[len("data:"):].strip())
                pct = event.get("progress")
                pct_text = f"{pct * 100:.0f}%" if pct is not None else "..."
                textinfo = event.get("textinfo") or ""
                preview = None
                if event.get("live_preview"):
                    try:
                        preview = _decode_data_uri(event["live_preview"])
                    except (ValueError, OSError):
                        preview = None
                yield f"{pct_text} {textinfo}".strip(), preview
                if event.get("completed"):
                    break
    except requests.RequestException:
        # SSE is a progress nicety; the POST thread (and its own error handling)
        # remains the actual source of truth for success/failure.
        return


def run_txt2img(backend_url, script_specs, prompt, negative_prompt, steps, sampler_name,
                 cfg_scale, width, height, seed, batch_count, batch_size, restore_faces,
                 tiling, enable_hr, hr_scale, hr_upscaler, hr_second_pass_steps,
                 denoising_strength, hr_resize_x, hr_resize_y, *script_arg_values):
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
        "force_task_id": id_task,
    }
    if alwayson_scripts:
        payload["alwayson_scripts"] = alwayson_scripts

    result_box = {}
    thread = threading.Thread(target=_post_txt2img, args=(backend_url, payload, result_box), daemon=True)
    thread.start()

    # Give the backend a moment to register the task before we start polling for it,
    # then stream progress until the POST thread reports done.
    time.sleep(0.2)
    for progress_text, preview in _stream_progress(backend_url, id_task, result_box):
        yield progress_text, preview, gr.update(), gr.update()

    thread.join(timeout=600)

    if result_box.get("error"):
        raise gr.Error(f"Generation request to {backend_url} failed: {result_box['error']}")

    data = result_box.get("response") or {}
    images = decode_images(data.get("images", []))
    info = json.loads(data.get("info", "{}"))
    infotext = (info.get("infotexts") or [""])[0]
    yield "done", None, images, infotext


def create_ui(backend_url=DEFAULT_BACKEND_URL):
    with gr.Blocks(title="reForge -- frontend (torch-free proof, BFISO Phase 8/9/10/11)") as demo:
        gr.Markdown(
            f"### Standalone frontend -- backend: `{backend_url}`\n"
            "BFISO Phase 8/9/10/11 proof: this process has no torch installed. txt2img only; "
            "script control values are sent with the request, progress streams live, "
            "Interrupt/Skip are wired up, and Hires. fix is supported -- see "
            "PHASE9.md / PHASE10.md / PHASE11.md."
        )

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

                with gr.Accordion("Scripts", open=False):
                    script_controls = build_alwayson_script_controls(backend_url)
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
            # functools.partial (not a lambda) so Gradio's
            # inspect.isgeneratorfunction(fn) check still sees run_txt2img's `yield`
            # through the wrapper -- a lambda wrapping a generator function is itself
            # NOT a generator function, since calling it just returns a generator
            # object rather than yielding, and Gradio silently expects a single
            # return value in that case (found by actually clicking Generate).
            fn=functools.partial(run_txt2img, backend_url, script_specs),
            inputs=[prompt, negative_prompt, steps, sampler_name, cfg_scale, width, height, seed,
                    batch_count, batch_size, restore_faces, tiling, enable_hr, hr_scale, hr_upscaler,
                    hr_second_pass_steps, denoising_strength, hr_resize_x, hr_resize_y,
                    *flat_script_inputs],
            outputs=[progress_box, preview_image, gallery, infotext_box],
        )
        skip_btn.click(fn=functools.partial(skip_current_image, backend_url), outputs=[progress_box])
        interrupt_btn.click(fn=functools.partial(interrupt_generation, backend_url), outputs=[progress_box])

    # Gradio 3.x requires an explicit queue to stream multiple yields from a
    # generator-based click handler back to the client -- without this,
    # run_txt2img's progress/preview yields fail with "Need to enable queue to use
    # generators" (found by actually clicking Generate). concurrency_count defaults to
    # 1, which serializes ALL queued events -- Skip/Interrupt clicks would queue up
    # behind the still-running Generate generator and only fire once it already
    # finished on its own, making both buttons silently no-op until the user found
    # this by actually clicking them mid-generation. >=2 lets Skip/Interrupt jump the
    # queue while Generate's handler is still yielding.
    demo.queue(concurrency_count=3)
    return demo
