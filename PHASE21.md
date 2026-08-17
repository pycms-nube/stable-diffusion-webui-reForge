# PHASE21 — Shipped-UI Theme + style.css

Applies the shipped UI's actual visual theme to the torch-free split frontend
(`modules_frontend/app.py`), which had been running on Gradio's raw, unstyled
default look since Phase 8. Purely cosmetic — no component moved/renamed/rewired,
no new HTTP calls to the backend.

## What changed

- `modules_frontend/app.py`: `create_ui()`'s `gr.Blocks(...)` now passes
  `theme=THEME` and `css=_load_style_css()`.
  - `THEME` is `gr.themes.Default(font=[...], font_mono=[...])`, constructed with
    the exact same `font`/`font_mono` arguments as
    `modules/shared_gradio_themes.py::reload_gradio_theme()`'s
    `default_theme_args` — confirmed by reading that function directly. This is a
    plain Gradio theme object construction (no torch/backend call), so it doesn't
    reintroduce a torch dependency.
  - `_load_style_css()` reads `style.css` from this repo checkout's own local disk
    (`Path(__file__).resolve().parent.parent / "style.css"`) — the same file the
    shipped UI serves. Reading a CSS file off disk isn't a torch import either.
    Returns `None` (Gradio's own "no extra CSS" default) on any `OSError`, so a
    checkout missing `style.css` still starts rather than crashing.

## Explicit scope decision

Only the `gradio_theme="Default"` case is reproduced — that setting's own default
value. Non-Default named themes (Gradio Hub themes) are **not** reproduced:
`/sdapi/v1/options` only exposes the theme's *name* as a JSON string, not its
resolved CSS variables, and fetching an arbitrary Hub theme's definition would
require the same kind of direct backend/filesystem/Hub access this frontend
deliberately avoids. If the backend's configured theme isn't "Default", the split
frontend keeps using this hardcoded `Default` theme rather than matching whatever
the backend is actually set to.

## Verification

- Confirmed `THEME`'s constructor arguments match
  `modules/shared_gradio_themes.py:48-50`'s `default_theme_args` exactly (both
  `font` and `font_mono` lists, in order).
- Confirmed `modules_frontend/` still imports no `modules.*`/`ldm_patched.*`
  backend modules and no `torch` — this phase only added `pathlib.Path` (stdlib)
  and a `gr.themes.Default(...)` call, neither of which pulls in torch.
- No functional code path changed: every existing `.click()`/`.change()` input and
  output list is untouched, so this phase carries none of the risk a real logic
  change would.

## What this phase did NOT do

- Did not reproduce non-Default Hub themes (see scope decision above).
- Did not add any new settings/toggle for the frontend user to pick a theme —
  matches the shipped UI's own behavior of theme being a launch-time/settings-page
  choice, not a per-session UI control.
- Did not touch `webui.sh`'s own theme handling or `modules/shared_gradio_themes.py`
  itself.
