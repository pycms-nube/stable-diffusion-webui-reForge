"""
BFISO Phase 8 -- standalone, torch-free frontend entry point.

Deliberately does NOT import webui.py (its module scope runs initialize_forge() /
initialize.imports(), both torch-heavy -- see PHASE3.md) or anything under
modules.scripts/modules.processing/modules_forge/ldm_patched. Only gradio, requests,
PIL, argparse, and modules_frontend/modules.ui_script_schema, all confirmed torch-free.

Usage:
    python webui_frontend.py --backend-url http://127.0.0.1:7860 --port 7870

See PHASE8.md for what this proves and its current scope limits.
"""
import argparse
import os

# modules.ui_components.InputAccordion (used by modules.ui_script_schema for real
# fidelity, see PHASE4.md) lazily imports modules.script_callbacks on first
# construction, which reaches modules.shared_cmd_options -- and THAT runs the full
# backend argparse parser at import time, which doesn't know this process's own
# flags (--backend-url etc.) and would otherwise crash. The codebase already has an
# escape hatch for exactly this (modules/shared_cmd_options.py checks this env var
# and uses parse_known_args instead of parse_args); use it rather than working
# around argv ourselves. Must be set before any import that could trigger the chain.
os.environ.setdefault("IGNORE_CMD_ARGS_ERRORS", "1")

from modules_frontend.txt2img_ui import DEFAULT_BACKEND_URL, create_ui  # noqa: E402


def _patch_gradio_template_response():
    """Gradio 3.41.2 calls Starlette's TemplateResponse with the old
    (name, context) signature; Starlette 0.36+ requires (request, name, context),
    and the mismatch surfaces as `TypeError: unhashable type: 'dict'` deep in
    Jinja2's template cache on the very first page load. The app's own fix for
    this is modules/ui_gradio_extensions.py's reload_javascript(), but that
    function imports modules.shared -> modules.shared_items -> modules.scripts,
    which loads every extension (most import torch, PHASE3.md) -- exactly what
    this process must not need. Reimplemented standalone, without the JS/CSS
    injection half (this frontend doesn't use progressbar.js), just the part
    that keeps Gradio's own UI from crashing on load."""
    import gradio as gr

    original = gr.routes.templates.TemplateResponse

    def patched(*args, **kwargs):
        if args and isinstance(args[0], str):
            name = args[0]
            context = args[1] if len(args) > 1 else {}
            request = context.get("request")
            return original(request, name, context, *args[2:], **kwargs)
        return original(*args, **kwargs)

    gr.routes.templates.TemplateResponse = patched


def main():
    _patch_gradio_template_response()
    parser = argparse.ArgumentParser(description="reForge standalone frontend (BFISO Phase 8)")
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL,
                         help="Base URL of a running backend, e.g. from webui-backend.sh")
    parser.add_argument("--port", type=int, default=7870)
    parser.add_argument("--listen", action="store_true", help="Bind 0.0.0.0 instead of 127.0.0.1")
    args = parser.parse_args()

    demo = create_ui(backend_url=args.backend_url)
    demo.launch(server_name="0.0.0.0" if args.listen else "127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
