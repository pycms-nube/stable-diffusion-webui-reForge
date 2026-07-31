"""
Builds live Gradio input components from a script's JSON UI schema -- the same shape
returned by GET /sdapi/v1/script-info -- without importing the script module itself.

This module only imports `gradio`, deliberately: it exists so a frontend process that
doesn't have torch/ldm_patched installed can still render script controls. See
PHASE3.md for why script modules can't be imported directly in a thin frontend
process, and PHASE4.md for how this module was verified against real script schemas.
"""
import gradio as gr

from modules.ui_components import InputAccordion


def _build_slider(arg):
    return gr.Slider(
        label=arg.get("label") or "",
        value=arg.get("value"),
        minimum=arg.get("minimum"),
        maximum=arg.get("maximum"),
        step=arg.get("step"),
    )


def _build_number(arg):
    return gr.Number(
        label=arg.get("label") or "",
        value=arg.get("value"),
        minimum=arg.get("minimum"),
        maximum=arg.get("maximum"),
        step=arg.get("step"),
    )


def _build_checkbox(arg):
    return gr.Checkbox(label=arg.get("label") or "", value=bool(arg.get("value")))


def _build_radio(arg):
    return gr.Radio(
        label=arg.get("label") or "",
        value=arg.get("value"),
        choices=arg.get("choices") or [],
    )


def _build_dropdown(arg):
    return gr.Dropdown(
        label=arg.get("label") or "",
        value=arg.get("value"),
        choices=arg.get("choices") or [],
        multiselect=bool(arg.get("multiselect")),
    )


def _build_checkbox_group(arg):
    return gr.CheckboxGroup(
        label=arg.get("label") or "",
        value=arg.get("value") or [],
        choices=arg.get("choices") or [],
    )


def _build_textbox(arg):
    return gr.Textbox(
        label=arg.get("label") or "",
        value=arg.get("value") or "",
        lines=arg.get("lines") or 1,
    )


def _build_input_accordion(arg):
    # InputAccordion (modules/ui_components.py) is a real gr.Checkbox subclass that
    # also drives an Accordion's open/closed state -- use the actual class so the
    # reconstructed control keeps that behavior instead of degrading to a plain
    # Checkbox.
    return InputAccordion(label=arg.get("label") or "", value=bool(arg.get("value")))


def _build_fallback(arg):
    # Unknown/unmapped component type: render read-only so the value stays visible
    # instead of silently dropping it, and the label flags what needs a builder.
    return gr.Textbox(
        label=f"{arg.get('label') or '(unnamed)'} [unsupported: {arg.get('component')}]",
        value=str(arg.get("value", "")),
        interactive=False,
    )


_BUILDERS = {
    "Slider": _build_slider,
    "Number": _build_number,
    "Checkbox": _build_checkbox,
    "Radio": _build_radio,
    "Dropdown": _build_dropdown,
    "CheckboxGroup": _build_checkbox_group,
    "Textbox": _build_textbox,
    "InputAccordion": _build_input_accordion,
}


def build_controls_from_schema(args):
    """
    args: list of dicts shaped like the ScriptArg JSON from /sdapi/v1/script-info
    (label, value, minimum, maximum, step, choices, component, multiselect, lines).

    Returns a list of live gr.Component instances, in the same order as `args`.
    """
    components = []
    for arg in args:
        builder = _BUILDERS.get(arg.get("component"), _build_fallback)
        components.append(builder(arg))
    return components
