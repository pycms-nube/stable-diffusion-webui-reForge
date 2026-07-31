# PHASE4 — Backend-Served Script UI Schema

Executes the recommendation from [PHASE3.md](PHASE3.md): since deferring imports across
577 files (including vendored third-party ML codebases) isn't tractable, extend the
existing `/sdapi/v1/script-info` schema so a torch-free frontend can build real script
controls from JSON instead of importing script modules.

## What changed

1. **[modules/api/models.py](modules/api/models.py)** — `ScriptArg` gained three fields:
   `component` (the Gradio widget class name backing the control, e.g. `"Slider"`),
   `multiselect` (for Dropdown/CheckboxGroup, whether the value is a list), `lines` (for
   Textbox). All optional, default `None` — existing consumers of `/sdapi/v1/script-info`
   are unaffected.
2. **[modules/scripts.py:678](modules/scripts.py#L678)** — `create_script_ui_inner`
   (already backend-side, already has the live `gr.Component` in hand) now populates
   `component=type(control).__name__` and reads `multiselect`/`lines` off the control the
   same way it already read `value`/`minimum`/`maximum`/`step`. Correct by construction —
   it's reading the real live object's own class name and attributes, not inferring or
   guessing.
3. **New: [modules/ui_script_schema.py](modules/ui_script_schema.py)** — the frontend-side
   half. `build_controls_from_schema(args)` takes a list of dicts shaped like the
   `ScriptArg` JSON and returns live `gr.Component` instances. Only imports `gradio` and
   `modules.ui_components` (confirmed torch-free — see verification below). Maps
   `Slider`/`Number`/`Checkbox`/`Radio`/`Dropdown`/`CheckboxGroup`/`Textbox`/
   `InputAccordion` to real Gradio (or Forge-custom, for `InputAccordion`) component
   constructors; anything else falls back to a read-only `Textbox` showing the raw value
   and flagging the unsupported type in its label, rather than silently dropping data or
   crashing.

## Verification (not a smoke test)

Fetched the **real** `/sdapi/v1/script-info` payload from a live backend
(`webui-backend.sh` + `waiIllustriousSDXL_v170.safetensors`) — 55 real scripts, 415
total args. For every arg: built the component via `build_controls_from_schema`, then
asserted the live component's actual attributes (`.value`, `.minimum`, `.maximum`,
`.step`, `.choices`, `.multiselect`, `.lines`) exactly match the schema.

First run found 66 apparent mismatches, all `choices`-related — traced to a genuine
Gradio 3.41 quirk, not a bug in the builder: `gr.Radio` normalizes plain-string choices
into `(label, value)` tuples internally while `gr.Dropdown` doesn't. Fixed the
*verification script's* comparison to apply the same `[c[0] if isinstance(c, tuple) else
c for c in choices]` normalization `modules/scripts.py` itself already uses — re-ran:

```
Total scripts: 55
Total args verified: 415
Component type distribution: {'Checkbox': 99, 'Radio': 62, 'Slider': 195, 'Textbox': 12,
  'Dropdown': 24, 'Number': 8, 'State': 6, 'CheckboxGroup': 4, 'InputAccordion': 2,
  'Markdown': 1, 'HTML': 2}
Fallback (unmapped component type) hits: 9  (State x6, Markdown x1, HTML x2)
Mismatches: 0

RESULT: PASS -- every reconstructed component's attributes exactly match the schema.
```

406/415 args (97.8%) reconstruct with byte-for-byte attribute fidelity via real component
classes. The 9 fallback hits are `State` (internal script bookkeeping, not user-facing —
e.g. ControlNet's tab-index tracking), `Markdown`/`HTML` (static display text, not an
input). All three are legitimately not "arguments a user configures," so the graceful
read-only fallback is the right behavior, not a gap.

**`InputAccordion` note**: it's a real `gr.Checkbox` subclass
([modules/ui_components.py:88](modules/ui_components.py#L88)) that also drives an
Accordion's open/closed state. Initially fell into the generic fallback (exact-class-name
lookup missed the subclass); added an explicit builder that constructs the real
`InputAccordion` class (confirmed torch-free — only imports `gradio`) instead of
substituting a plain `Checkbox`, so the reconstructed control keeps its actual behavior.
This required verifying inside an active `gr.Blocks()` context — `InputAccordion.__init__`
registers a `.change()` handler at construction time, which Gradio requires a live Blocks
context for. Same requirement applies in real usage (`modules/scripts.py` already builds
all script UI inside a `Blocks` context), so this isn't a new constraint.

## Regression check

Re-ran `pytest test/test_txt2img.py test/test_img2img.py --no-server` against a live
backend with all changes applied: identical result to the PHASE2.md baseline — 11/12
txt2img pass (1 expected skip), 13/15 combined pass with the same 2 pre-existing img2img
422 failures documented in PHASE2.md (unrelated schema drift, not touched here). No new
failures introduced.

## What this phase did NOT do

- **Did not wire this into an actual frontend process.** `build_controls_from_schema`
  is built and verified against real data, but nothing yet calls it from a Gradio UI
  fetching `/sdapi/v1/script-info` over HTTP instead of importing script modules. That's
  the actual "make the frontend process torch-free" wiring — this phase built and proved
  the missing piece, not the integration.
- **Did not address layout.** `script.ui()` methods arrange controls in `gr.Row`/
  `gr.Group`/`gr.Accordion`/`gr.Tab` containers; `build_controls_from_schema` returns a
  flat list. Functionally complete (every value-bearing control is present and correctly
  typed/bounded), but a real frontend integration would render them in one column rather
  than the original layout unless it also fetches and replays some layout hints — not
  attempted here.
- **Did not handle conditional visibility.** Some scripts (e.g. ControlNet) show/hide
  controls dynamically based on other controls' values (`gr.update(visible=...)`
  callbacks registered in the original `.ui()` method). Those callbacks are Python code
  living in the script module itself — reproducing them without importing the script is
  a separate, harder problem than static schema reconstruction, not solved here.
- **Did not touch [modules/safe.py](modules/safe.py)** or the ~64 other core-module
  blockers PHASE3.md catalogued — this phase deliberately took the "serve a schema"
  path specifically so those files *don't* need to change.

## Next steps

- Wire `build_controls_from_schema` into an actual frontend entry point: fetch
  `/sdapi/v1/script-info` over HTTP at UI-startup time, call it per script, and place the
  resulting components into the txt2img/img2img tabs in place of today's
  `script.ui()`-driven construction.
- Decide how much layout fidelity matters enough to invest in (flat list vs. replaying
  Row/Accordion structure) before doing the wiring above.
- Decide what to do about conditional-visibility scripts (ControlNet et al.) — likely
  needs its own small design, out of scope for a generic schema.
