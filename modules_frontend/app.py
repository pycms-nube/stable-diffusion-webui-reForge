"""
BFISO Phase 12 -- top-level assembly of the torch-free frontend's tabs.

webui_frontend.py imports create_ui from here (previously from txt2img_ui.py directly,
back when txt2img was the only tab -- see PHASE8-11.md). Owns the single gr.Blocks and
the single demo.queue() call, since both tabs share them.

Phase 21 (PHASE21.md) applied the shipped UI's actual theme: gr.themes.Default() with
the same font/font_mono args as modules/shared_gradio_themes.py::reload_gradio_theme()
(a plain Python object construction, no torch/backend call needed -- confirmed by
reading that function directly), plus the same style.css file the shipped UI serves,
read from this repo checkout's own local disk (this frontend lives in the same
checkout as style.css; reading a CSS file isn't a torch import). Non-Default named
themes (gr.themes.gradio_theme option other than "Default") are NOT reproduced --
only the theme's name is exposed over /sdapi/v1/options as JSON, not its actual CSS
variables, so reproducing an arbitrary Hub theme would need direct backend
filesystem/hub access this frontend deliberately doesn't have.
"""
from pathlib import Path

import gradio as gr

from modules_frontend.common import DEFAULT_BACKEND_URL
from modules_frontend.img2img_ui import create_img2img_tab
from modules_frontend.txt2img_ui import create_txt2img_tab

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STYLE_CSS_PATH = _REPO_ROOT / "style.css"

# Matches modules/shared_gradio_themes.py::reload_gradio_theme()'s default_theme_args
# exactly -- the shipped UI's actual theme when gradio_theme="Default" (the setting's
# own default value). Non-Default Hub themes are out of scope, see module docstring.
THEME = gr.themes.Default(
    font=["Source Sans Pro", "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=["IBM Plex Mono", "ui-monospace", "Consolas", "monospace"],
)


def _load_style_css():
    try:
        return _STYLE_CSS_PATH.read_text(encoding="utf-8")
    except OSError:
        return None


def create_ui(backend_url=DEFAULT_BACKEND_URL):
    with gr.Blocks(title="reForge -- frontend (torch-free proof, BFISO Phase 8-21)",
                   theme=THEME, css=_load_style_css()) as demo:
        gr.Markdown(
            f"### Standalone frontend -- backend: `{backend_url}`\n"
            "BFISO Phase 8-21 proof: this process has no torch installed. txt2img + "
            "img2img, script control values are sent with the request, progress "
            "streams live, Interrupt/Skip are wired up, txt2img supports Hires. fix, "
            "and this uses the shipped UI's actual theme + style.css -- see PHASE9-21.md."
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
