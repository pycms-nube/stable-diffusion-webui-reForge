"""
BFISO Phase 8 -- a genuinely torch-free txt2img UI.

This module (and webui_frontend.py, which launches it) is the actual cutover proof:
everything up to now (PHASE0-7.md) built and verified the pieces, but the shipped
Gradio app (modules/ui.py) still runs fully in-process and still needs torch, because
it imports modules.scripts/modules.processing/modules.sd_* directly (PHASE3.md). This
module deliberately imports NONE of that -- only gradio, requests, PIL, and
modules.ui_script_schema (itself torch-free, see PHASE4.md) -- and drives generation
entirely over HTTP against a separately-running backend (webui-backend.sh or the full
app's --api mode).

Scope, honestly: txt2img only, core params only. Script controls are fetched and
rendered live from the backend's real /sdapi/v1/script-info schema (proving Phase 4's
builder works in a real app, not just a verification harness), but their values are not
yet wired back into the generation request -- see PHASE8.md for why that's a separate,
larger follow-up (matching alwayson_scripts argument order/shape) rather than bundled
into this first cutover slice.
"""
import base64
import io
import json

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


def fetch_script_info(backend_url):
    try:
        return _get(backend_url, "/sdapi/v1/script-info")
    except requests.RequestException as e:
        raise RuntimeError(f"Could not reach backend at {backend_url} for /sdapi/v1/script-info: {e}") from e


def decode_images(b64_list):
    images = []
    for b64 in b64_list:
        images.append(Image.open(io.BytesIO(base64.b64decode(b64))))
    return images


def run_txt2img(backend_url, prompt, negative_prompt, steps, sampler_name, cfg_scale, width, height, seed):
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": int(steps),
        "sampler_name": sampler_name,
        "cfg_scale": float(cfg_scale),
        "width": int(width),
        "height": int(height),
        "seed": int(seed),
    }
    try:
        r = requests.post(f"{backend_url}/sdapi/v1/txt2img", json=payload, timeout=600)
        r.raise_for_status()
    except requests.RequestException as e:
        raise gr.Error(f"Generation request to {backend_url} failed: {e}") from e

    data = r.json()
    images = decode_images(data["images"])
    info = json.loads(data.get("info", "{}"))
    infotext = (info.get("infotexts") or [""])[0]
    return images, infotext


def build_alwayson_script_controls(backend_url):
    """Renders every alwayson script's controls, schema-built (PHASE4.md), inside its
    own Accordion. Live and interactive against real backend data -- not yet wired
    back into the generation request (see module docstring / PHASE8.md)."""
    try:
        script_info = fetch_script_info(backend_url)
    except RuntimeError as e:
        gr.Markdown(f"⚠️ Could not load script list: {e}")
        return

    alwayson = [s for s in script_info if s.get("is_alwayson") and not s.get("is_img2img") and s.get("args")]
    if not alwayson:
        gr.Markdown("(no always-on txt2img scripts reported by the backend)")
        return

    for script in alwayson:
        with gr.Accordion(script["name"], open=False):
            build_controls_from_schema(script["args"])


def create_ui(backend_url=DEFAULT_BACKEND_URL):
    with gr.Blocks(title="reForge -- frontend (torch-free proof, BFISO Phase 8)") as demo:
        gr.Markdown(
            f"### Standalone frontend -- backend: `{backend_url}`\n"
            "BFISO Phase 8 proof: this process has no torch installed. txt2img only; "
            "script controls are live but not yet sent with the request -- see PHASE8.md."
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
                    try:
                        sampler_choices = fetch_samplers(backend_url)
                    except RuntimeError as e:
                        sampler_choices = []
                        gr.Markdown(f"⚠️ {e}")
                    sampler_name = gr.Dropdown(label="Sampler", choices=sampler_choices,
                                                value=sampler_choices[0] if sampler_choices else None)
                    seed = gr.Number(label="Seed", value=-1, precision=0)

                with gr.Accordion("Scripts (preview -- not yet sent with request)", open=False):
                    build_alwayson_script_controls(backend_url)

                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=5):
                gallery = gr.Gallery(label="Output", show_label=True, columns=2)
                infotext_box = gr.Textbox(label="Generation info", lines=4, interactive=False)

        generate_btn.click(
            fn=lambda *args: run_txt2img(backend_url, *args),
            inputs=[prompt, negative_prompt, steps, sampler_name, cfg_scale, width, height, seed],
            outputs=[gallery, infotext_box],
        )

    return demo
