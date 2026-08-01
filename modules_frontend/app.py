"""
BFISO Phase 12 -- top-level assembly of the torch-free frontend's tabs.

webui_frontend.py imports create_ui from here (previously from txt2img_ui.py directly,
back when txt2img was the only tab -- see PHASE8-11.md). Owns the single gr.Blocks and
the single demo.queue() call, since both tabs share them.
"""
import gradio as gr

from modules_frontend.common import DEFAULT_BACKEND_URL
from modules_frontend.img2img_ui import create_img2img_tab
from modules_frontend.txt2img_ui import create_txt2img_tab


def create_ui(backend_url=DEFAULT_BACKEND_URL):
    with gr.Blocks(title="reForge -- frontend (torch-free proof, BFISO Phase 8-12)") as demo:
        gr.Markdown(
            f"### Standalone frontend -- backend: `{backend_url}`\n"
            "BFISO Phase 8-12 proof: this process has no torch installed. txt2img + "
            "basic img2img, script control values are sent with the request, progress "
            "streams live, Interrupt/Skip are wired up, and txt2img supports Hires. fix "
            "-- see PHASE9-12.md."
        )
        with gr.Tabs():
            with gr.Tab("txt2img"):
                create_txt2img_tab(backend_url)
            with gr.Tab("img2img"):
                create_img2img_tab(backend_url)

    # Gradio 3.x requires an explicit queue to stream multiple yields from a
    # generator-based click handler back to the client (PHASE9.md). concurrency_count
    # defaults to 1, which serializes ALL queued events across BOTH tabs -- Skip/
    # Interrupt clicks (on either tab) would queue up behind a still-running Generate
    # generator and only fire once it already finished on its own (PHASE10.md).
    demo.queue(concurrency_count=3)
    return demo
