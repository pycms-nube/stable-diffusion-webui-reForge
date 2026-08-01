"""
BFISO Phase 12 -- helpers shared by txt2img_ui.py and img2img_ui.py.

Extracted from txt2img_ui.py (Phase 8-11) when img2img_ui.py needed the same
backend-fetch/progress-streaming/script-control machinery. Torch-free, same as every
other module_frontend file -- only gradio, requests, PIL.
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


def _post(backend_url, path, timeout=10):
    r = requests.post(f"{backend_url}{path}", timeout=timeout)
    r.raise_for_status()
    return r


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


def encode_image_to_base64(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_data_uri(data_uri):
    _header, b64data = data_uri.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(b64data)))


def build_alwayson_script_controls(backend_url, is_img2img):
    """Renders every alwayson script's controls for the given pipeline (txt2img or
    img2img -- scripts.py reports is_img2img per script since the two pipelines have
    separate script_callbacks.on_ui_settings registries), schema-built (PHASE4.md),
    inside its own Accordion. Returns [(script_name, [gr.Component, ...]), ...] in the
    same order the backend's own script list expects args in -- required so the caller
    can rebuild the alwayson_scripts payload on submit."""
    try:
        script_info = fetch_script_info(backend_url)
    except RuntimeError as e:
        gr.Markdown(f"⚠️ Could not load script list: {e}")
        return []

    alwayson = [s for s in script_info
                if s.get("is_alwayson") and bool(s.get("is_img2img")) == is_img2img and s.get("args")]
    if not alwayson:
        gr.Markdown("(no always-on scripts reported by the backend for this pipeline)")
        return []

    script_controls = []
    for script in alwayson:
        with gr.Accordion(script["name"], open=False):
            controls = build_controls_from_schema(script["args"])
        script_controls.append((script["name"], controls))
    return script_controls


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


def post_generate(backend_url, path, payload, result_box):
    try:
        r = requests.post(f"{backend_url}{path}", json=payload, timeout=600)
        r.raise_for_status()
        result_box["response"] = r.json()
    except requests.RequestException as e:
        result_box["error"] = e
    finally:
        result_box["done"] = True


def stream_progress(backend_url, id_task, result_box):
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
