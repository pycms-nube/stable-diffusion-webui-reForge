# PHASE16 — Two-Click Confirm for Skip/Interrupt

Closes the "no cancel confirmation UX" gap. Skip/Interrupt now require a second click
before actually firing, instead of acting immediately on the first click.

## What changed

- `modules_frontend/common.py`: new `build_confirm_action_button(backend_url, label,
  action_fn, output_box, variant="secondary")` — builds **and** wires a button in one
  call, implementing a two-click confirm pattern entirely with existing Gradio 3.x
  primitives (a `gr.State(False)` "armed" flag plus the button's own label as the
  confirmation surface), since Gradio 3.x has no built-in modal confirm dialog:
  - First click: button relabels to `"Confirm {label}?"`, arms, does not call the
    backend.
  - Second click (while armed): calls `action_fn(backend_url)`, writes its message to
    `output_box`, disarms, restores the original label.
- `modules_frontend/txt2img_ui.py` / `modules_frontend/img2img_ui.py`: Skip/Interrupt
  buttons now built via this helper instead of a plain `gr.Button` + manual `.click()`
  — removes the old direct wiring, no change to Generate's own button/click handler.

## Verification

**Build check**: `create_ui()` builds cleanly against the live backend with both
tabs' Skip/Interrupt now going through the shared helper.

**Live UI**: bundled into the Phase 16-19 consolidated review (see the end of that
phase's doc) rather than checked in isolation, per the user's explicit instruction to
implement this batch of non-functional/cosmetic gaps together and check once at the
end.

## What this phase did NOT do

- **No auto-reset timeout on the armed state** — once armed, a button stays armed
  (asking for confirmation) until either clicked again or the page is reloaded; it
  doesn't silently revert after N seconds of inactivity. Simple to add later
  (`gr.Timer` in newer Gradio, or a JS-level `setTimeout` here) but out of scope for
  this pass.
- **No "are you sure" styling/animation** — the confirmation surface is just the
  button's own text changing, not a distinct visual treatment (color pulse, icon,
  etc.) beyond the existing `variant="stop"` on Interrupt.
