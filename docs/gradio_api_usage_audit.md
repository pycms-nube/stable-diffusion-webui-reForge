# Gradio Blocks/Component API Usage Audit

Scope: every file matching `modules/ui*.py` and `extensions-builtin/*.py` (899 files scanned via
filename/import grep; 57 files actually import/use `gradio`). Baseline installed version:
**gradio==3.41.2** (per `requirements_versions*.txt`). Targets evaluated: 4.x, 5.x, 6.x per
gradio's official migration guides (`gradio.app/changelog`,
`github.com/gradio-app/gradio/issues/6339` for 4.0, `github.com/gradio-app/gradio/issues/9463`
for 5.0, `gradio.app/guides/gradio-6-migration-guide` for 6.0).

Legend for impact columns: "none identified" = call signature is stable across that jump as far
as the public migration guides document; otherwise the cell names the concrete breaking change.

Key recurring breaking-change patterns found in this codebase (see per-file detail below for
exact lines):

- **P1 — `gr.<Component>.update(...)` class-method pattern.** Removed in Gradio 4.0. Must change
  to module-level `gr.update(...)` (if values only) or return a fresh component instance.
  Found extensively: `modules/ui.py`, `modules/ui_checkpoint_merger.py`, `modules/ui_common.py`,
  `modules/ui_extensions.py`, `modules/ui_prompt_styles.py`, `modules/ui_settings.py`,
  `extensions-builtin/Lora/ui_edit_user_metadata.py`,
  `extensions-builtin/reForge-advanced_model_sampling_backported/...`,
  `extensions-builtin/reForge-advanced_model_sampling/...`,
  `extensions-builtin/sd_forge_dynamic_thresholding/...`,
  `extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/preset.py`,
  `extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/controlnet_ui_group.py`.
- **P2 — `gr.Box` layout class.** Removed in Gradio 4.0 (merged into `gr.Group`).
  Found: `modules/ui_prompt_styles.py`, `modules/ui_extra_networks_user_metadata.py`.
- **P3 — `gr.Image`/`gr.Audio`/`gr.Video` `source=` param.** Renamed `source` -> `sources` (and
  type changed from a single string to a list) starting Gradio 4.0. Found on many `gr.Image(...,
  source="upload", ...)` calls across `modules/ui.py`, `modules/ui_postprocessing.py`,
  `extensions-builtin/sd_forge_svd/scripts/forge_svd.py`,
  `extensions-builtin/sd_forge_z123/scripts/forge_z123.py`,
  `extensions-builtin/sd_forge_controlnet_example/...`.
- **P4 — `gr.Image(..., tool="editor"/"sketch"/"color-sketch", image_mode=..., brush_color=...)`.**
  The `tool` parameter and associated brush/editor kwargs were removed/restructured when
  `gr.Image`'s sketch/mask editing was replaced by `gr.ImageEditor` in Gradio 4.x/5.x. This is a
  **silent behavior change risk**, not just an import error: old kwargs may be silently ignored
  pre-4.0-strict-kwargs removal, then hard-error once `**kwargs` passthrough is removed in 4.0.
  Found: `modules/ui.py` (img2img sketch/inpaint tabs), `modules/ui_postprocessing.py`.
- **P5 — `**kwargs` passthrough to components (e.g. `**shared.hide_dirs`).** Gradio 4.0 removes
  silent acceptance of unknown kwargs; any stray/unsupported key now raises `TypeError` instead
  of a warning. Found: `modules/ui.py` (`**shared.hide_dirs` on several `gr.Textbox`/`gr.Checkbox`
  calls), `modules/ui_postprocessing.py`.
- **P6 — `gr.deprecation.GradioDeprecationWarning`, `gradio.utils.version_check`,
  `gradio.utils.get_local_ip_address` monkeypatches.** These are private/internal gradio
  attributes (`modules/ui.py` lines 40, 57-58). Internal module layout changed significantly
  across 4.x->5.x->6.x reworks (SSR rewrite in 5.0); these monkeypatches are very likely to break
  silently (AttributeError at best, silently-ignored patch at worst) and must be re-verified
  against each target version's internals.
- **P7 — Component subclassing (`class X(gr.Button)`, `class X(FormComponent, gr.Row)`, etc.)**
  found in `modules/ui_components.py` and
  `extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/tool_button.py`. Gradio's
  internal base-class hierarchy (`IOComponent`, `FormComponent`, `Component`) was restructured in
  4.0 (new `gradio_client` component protocol) and again with the Gradio 5 SSR rewrite — any
  subclass relying on internal method names (e.g. `get_expected_parent`, `postprocess`,
  `preprocess`) needs re-validation at each major version, flagged as **silent runtime risk**
  since broken subclassing often doesn't raise until the component is actually rendered/used.
- **P8 — `modules/ui_tempdir.py` directly touches `gradio.components.IOComponent.pil_to_temp_file`
  and `gradio.temp_file_sets`/`gradio.temp_dirs` private attributes.** These are undocumented
  internals that changed across 3.9/3.15/4.x/5.x; every target version needs this file
  re-validated against the installed gradio's actual temp-file handling internals. High silent-
  breakage risk (temp files silently not cleaned up, or image previews failing to render).
- **P9 — `gr.routes.templates.TemplateResponse` monkeypatch** in
  `modules/ui_gradio_extensions.py` (custom HTML template injection for extra JS/CSS). Gradio's
  FastAPI routing layer changed with the Gradio 5 SSR rewrite; this is one of the highest-risk
  silent-behavior-change items in the whole audit — if the patch silently no-ops, custom
  JS/CSS hooks for the whole UI stop working with no error.
- **P10 — `gr.update()` used as a *sentinel/no-op* return value from event handler functions**
  (very common across this codebase, 100+ call sites). This pattern itself is NOT removed in any
  target version (module-level `gr.update()` survives through 6.x) — flagged "none identified"
  unless combined with a removed parameter.
- **P11 — `evt: gr.SelectData` select-event payload** (`extensions-builtin/Lora/ui_edit_user_metadata.py`).
  Stable API, no breaking change identified through 6.x.
- **P12 — `_js=` kwarg on event listeners** (`.click(fn=..., _js="...")`), used pervasively
  (`modules/ui.py`, `modules/ui_extensions.py`, `modules/ui_toprow.py`, controlnet UI files,
  etc.). `_js` was renamed/aliased over gradio's history; by 4.0 the documented parameter is
  `js=`, with `_js` kept only as a deprecated back-compat alias for some releases. **Needs
  explicit verification against the exact target version** — if the alias is dropped, every one
  of these call sites breaks silently (JS callback simply never fires, no exception). High
  silent-breakage risk given how heavily this repo depends on custom JS hooks.
- **P13 — `gr.Dropdown(..., tooltip=...)`** (`modules/ui_prompt_styles.py` line 63). `tooltip` is
  not a documented pre-6.0 Dropdown param in the public changelog; must verify against target
  version's `gr.Dropdown` signature — possible kwarg-removal breakage under P1-class strictness.
- **P14 — ControlNet's `gr.State`, `gr.Gallery`, `gr.UploadButton`.upload()`** usage
  (`extensions-builtin/sd_forge_controlnet/...`) — stable APIs, no breaking change identified.
- **gr.Chatbot / gr.ChatInterface** — NOT used anywhere in the audited file set. The Gradio 6.0
  tuple-format removal (`gr.Chatbot(value=[[user,bot]])` -> messages format) does **not** apply to
  this codebase.
- **gr.Dataframe `row_count`/`col_count`** — NOT used anywhere in the audited file set. The
  Gradio 6.0 restructuring of these params does **not** apply to this codebase.
- **`concurrency_count`, `enable_queue`, `show_tips`, `cache_examples="lazy"`** — NOT found
  anywhere in the audited file set (no `launch()` call site uses these removed params; the single
  `demo.launch(...)` call is in `webui.py`, outside this audit's file scope, and should be
  re-checked separately as part of the migration plan).

---

## File-by-file inventory

### modules/ui.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 12-13 | `import gradio as gr`, `import gradio.utils` | none identified | none identified | none identified |
| 40 | `gr.deprecation.GradioDeprecationWarning` | Internal module; `deprecation.py` removed in 4.0 per official changelog ("Removes deprecation.py") — **breaks**, must find new mechanism or drop the warning filter | re-verify | re-verify |
| 57-58 | `gradio.utils.version_check = lambda: None`, `gradio.utils.get_local_ip_address = lambda: '127.0.0.1'` monkeypatch | Internal `gradio.utils` symbols; may be renamed/removed — high risk, silent no-op if attrs don't exist | SSR rewrite in 5.0 restructured internals further — re-verify | re-verify |
| 105,111,144,149,154,207,210,214,222,353,469,480,484,732,812,1048,1082,1175 | `gr.update(...)` as handler return value | none identified (P10) | none identified | none identified |
| 251 | `gr.Dropdown([], label=..., multiselect=True, ...)` | none identified | none identified | none identified |
| 253-254 | `.change(fn=lambda x: gr.Dropdown.update(visible=bool(x)))` | **P1: `.update()` classmethod removed in 4.0** — return `gr.Dropdown(visible=...)` or `gr.update(visible=...)` instead | none identified once fixed | none identified |
| 262,276,520,881,884,909,1144 | `gr.Blocks(analytics_enabled=False, ...)` | none identified | none identified | **P6-adjacent: app-level params (incl. things set on Blocks) moved to `launch()` in 6.0** — verify `analytics_enabled`/`title` still accepted on `Blocks()` constructor at 6.0 |
| 281,523 | `gr.Tabs(elem_id=..., elem_classes=...)` | none identified | none identified | none identified |
| 284,526,558,562,566,570,584,588,637,646,916,929,950 | `gr.Tab(...)`/`gr.TabItem(...)` context managers | none identified | none identified | none identified |
| 287,529,599 | `gr.Accordion(...)` | none identified | none identified | none identified |
| 288,298,302,306,346,349,360,529,633,639,642,669,716,719,886,889,923,926,941,944,1006 | `gr.Column(...)` | none identified | none identified | none identified |
| 299-348 (many) | `gr.Slider(minimum=,maximum=,step=,label=,value=,elem_id=)` | none identified | none identified | **check `gr.Slider` removed params list in 6.0 guide (none of these are in the removed-param list for Slider published) — none identified** |
| 311,319,347,350,536,677,686,893,897,922,940,997,1176 | `gr.Row(...)` | none identified | none identified | none identified |
| 325,339,342,343,365,629,696,933,934,953,956,964,975 | `gr.Dropdown(choices=, value=, label=, elem_id=, type=)` | none identified | none identified | none identified |
| 348,351,596-598,601,634,683-684,701,917-918,930,932,937,960-961,965,971-972,996 | `gr.Textbox(label=, elem_id=, lines=, placeholder=, elem_classes=, **shared.hide_dirs)` | **P5: `**shared.hide_dirs` kwargs passthrough — breaks under strict-kwargs if `shared.hide_dirs` carries any key not in `Textbox.__init__`'s signature** | none identified | none identified |
| 365,697 | `create_override_settings_dropdown(...) -> gr.Dropdown` type hint | none identified | none identified | none identified |
| 377 | `component.release if isinstance(component, gr.Slider) else component.change` — `.release` event | none identified | none identified | none identified |
| 429,440,744,811,832,837,1048,1082 | `wrap_gradio_gpu_call(...)` fed to `.click()`/`.submit()` `fn=` | none identified (internal wrapper, not gradio API itself) | none identified | none identified |
| 436-437,806-807 | `toprow.prompt.submit(**txt2img_args)`, `toprow.submit.click(**txt2img_args)` | none identified | none identified | none identified |
| 439,447,449,616,621,652,663,735,811,819,832,837 | `.click(fn=, inputs=, outputs=, show_progress=False, _js=...)` | **P12: `_js=` kwarg — verify alias survives to target version; may silently stop firing** | re-verify | re-verify |
| 505-510,844-849 | `.change(fn=, inputs=, outputs=, show_progress=False)` | none identified | none identified | none identified |
| 559,567,571,585-586,587,596,887,1167 | `gr.Image(..., source="upload", tool="editor"/"sketch"/"color-sketch", image_mode=..., brush_color=...)` | **P3+P4: `source` -> `sources` rename; `tool`/`brush_color` kwargs replaced by `gr.ImageEditor` usage pattern — breaks (silent: these kwargs may be silently dropped before hard TypeError under strict kwargs in 4.0)** | re-verify against `gr.ImageEditor` migration | re-verify |
| 556,634,968,969,981,984-985,192,197 | `gr.Number(value=, visible=, precision=, label=, elem_id=)` | none identified | none identified | none identified |
| 572,653 | `gr.State(None)` | none identified | none identified | none identified |
| 582 | `inpaint_color_sketch.change(update_orig, [...], ...)` positional-args form | none identified | none identified | none identified |
| 590,595,890,892,901,909,942,951,1009-1010,1171 | `gr.HTML(...)` | none identified | none identified | **5.0: padding default changes discussed for HTML/Markdown in 6.0 guide — verify visual regressions, not a hard break** |
| 602,931 | `gr.CheckboxGroup(choices=, label=, value=, info=)` | none identified | none identified | none identified |
| 600,920,935-938,980,987,989-990,992 | `gr.Checkbox(label=, value=, elem_id=, **shared.hide_dirs)` | **P5 on line 600's `**shared.hide_dirs`** | none identified | none identified |
| 607,665-666,735,760,769 | `.select(fn=, inputs=, outputs=, _js=...)` | **P12 risk on `_js=` usages** | re-verify | re-verify |
| 629,710,713,717,995 | `gr.Radio(label=, choices=, value=, type="index")` | none identified | none identified | none identified |
| 647,651,674,678-679,706-707,720,919,993 | `gr.Slider(...)` (various) | none identified | none identified | none identified |
| 710 | `gr.Radio(..., type="index")` | none identified | none identified | none identified |
| 541,545,576-577,581,616,621,652,663,685,702,710(btn),927,945,998-999,1000,1012,1027,1047,1081,1114 | `gr.Button(..., value=, variant=, interactive=)` | none identified | none identified | none identified |
| 1007 | `gr.Text(elem_id=, value=, show_label=False)` (alias of Textbox) | none identified | none identified | none identified |
| 1008 | `gr.Gallery(label=, show_label=False, elem_id=, columns=4)` | none identified | none identified | none identified |
| 1048,1082 | `extra_outputs=[gr.update()]` | none identified (P10) | none identified | none identified |
| 1120 | `ui_loadsave.UiLoadsave[gradio_extensons.original_IOComponent_init]` — references internal `IOComponent` | **High risk: `IOComponent` base class reorganized in 4.0 (new component protocol) and again in 5.0 SSR rewrite — `original_IOComponent_init` monkeypatch target likely moves/renames, silent breakage if patch target doesn't exist post-upgrade** | re-verify | re-verify |
| 1137 | `extensions_interface: gr.Blocks = ui_extensions.create_ui()` type hint | none identified | none identified | none identified |
| 1144 | `gr.Blocks(theme=shared.gradio_theme, analytics_enabled=False, title="Stable Diffusion")` | none identified (theme/title still Blocks-constructor params pre-6.0) | none identified | **6.0 guide: "App-level parameters have been moved from Blocks to launch()" — `title`/`theme` may need to move to `launch()` call: verify from migration guide before porting to 6.x** |
| 1149 | `gr.Tabs(elem_id="tabs") as tabs` | none identified | none identified | none identified |
| 1156 | `gr.TabItem(label, id=ifid, elem_id=...)` | none identified | none identified | none identified |
| 1167 | `gr.Audio(interactive=False, value=, elem_id=, visible=False)` | none identified (no `source=`/`sources=` used here) | none identified | none identified |
| 1211 | `gr.__version__` f-string | none identified | none identified | none identified |

### modules/ui_checkpoint_merger.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 2 | `import gradio as gr` | none identified | none identified | none identified |
| 25 | `gr.Dropdown.update(choices=...)` returned from error handler | **P1: breaks in 4.0, replace with `gr.update(choices=...)`** | n/a once fixed | n/a |
| 31 | `gr.Blocks(analytics_enabled=False)` | none identified | none identified | see 6.0 app-level-param note above |
| 32,56,79 | `gr.Row(equal_height=False)`, `gr.Column(variant='compact')` | none identified | none identified | none identified |
| 34,81 | `gr.HTML(value=...)` | none identified | none identified | none identified |
| 37,40,43,61 | `gr.Dropdown(sd_models.checkpoint_tiles(), elem_id=, label=)` | none identified | none identified | none identified |
| 46,65 | `gr.Textbox(label=, elem_id=, value=)` | none identified | none identified | none identified |
| 47 | `gr.Slider(minimum=,maximum=,step=,label=,value=,elem_id=)` | none identified | none identified | none identified |
| 48,52,57 | `gr.Radio(choices=, value=, label=, elem_id=, type="index")` | none identified | none identified | none identified |
| 49 | `.change(fn=, inputs=, outputs=)` | none identified | none identified | none identified |
| 53,69-71 | `gr.Checkbox(value=, label=, elem_id=)` | none identified | none identified | none identified |
| 67 | `gr.Accordion("Metadata", open=False) as metadata_editor` | none identified | none identified | none identified |
| 73 | `gr.TextArea('{}', label=...)` | none identified | none identified | none identified |
| 74,77 | `gr.Button(...)` | none identified | none identified | none identified |
| 80 | `gr.Group(elem_id=...)` | none identified | none identified | none identified |
| 87 | `.change(lambda fmt: gr.update(visible=...), ..., show_progress=False)` | none identified (P10) | none identified | none identified |
| 89 | `.click(extras.read_metadata, inputs=, outputs=)` | none identified | none identified | none identified |
| 91-93 | `.click(fn=lambda: '', ...)`, `.click(fn=call_queue.wrap_gradio_gpu_call(...), extra_outputs=lambda: [gr.update() for _ in range(4)])` | none identified (P10) | none identified | none identified |

### modules/ui_common.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 8 | `import gradio as gr` | none identified | none identified | none identified |
| 24-29 | `gr.update()` return values | none identified (P10) | none identified | none identified |
| 152 | `gr.File.update(value=fullfns, visible=True)` | **P1: breaks in 4.0 — `gr.File.update` classmethod removed; use `gr.update(...)` or return `gr.File(...)`** | n/a once fixed | n/a |
| 181,185 | `gr.Column(elem_id=, variant='panel')` | none identified | none identified | none identified |
| 186 | `gr.Group(elem_id=...)` | none identified | none identified | none identified |
| 187 | `gr.Gallery(label=, show_label=False, elem_id=, columns=4, preview=True, height=...)` | none identified | check `preview=True` still valid param at target version | re-verify Gallery param list in 6.0 guide |
| 189 | `gr.Row(elem_id=, elem_classes=...)` | none identified | none identified | none identified |
| 206,226,234,250,305,318 | `.click(fn=, ...)` | none identified | none identified | none identified |
| 217 | `gr.File(None, file_count="multiple", interactive=False, show_label=False, visible=False, elem_id=...)` | none identified (no `source=` used) | none identified | none identified |
| 219 | `gr.Group()` | none identified | none identified | none identified |
| 220-221,266-268 | `gr.HTML(elem_id=, elem_classes=...)` | none identified | none identified | none identified |
| 223 | `gr.Textbox(visible=False, elem_id=...)` | none identified | none identified | none identified |
| 225 | `gr.Button(visible=False, elem_id=...)` | none identified | none identified | none identified |
| 302 | `gr.update(**(args or {}))` for a variable number of outputs | none identified (P10) | none identified | none identified |
| 314 | docstring reference to "`gr.Box`" (not actual code use — informational only) | **P2: `gr.Box` removed in 4.0 — if this docstring reflects actual intended behavior elsewhere, verify no live `gr.Box` use remains; here it's comment-only** | n/a | n/a |
| 319 | `.click(fn=lambda: gr.update(visible=True), ...)` | none identified (P10) | none identified | none identified |
| 322 | `.then(fn=None, _js="...")` | **P12: `_js=` risk** | re-verify | re-verify |
| 325 | `.click(fn=None, _js="closePopup")` | **P12: `_js=` risk** | re-verify | re-verify |

### modules/ui_components.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 1 | `import gradio as gr` | none identified | none identified | none identified |
| 6 | `gr.components.Form` reference (internal) | **High risk: `gr.components.Form` is a private internal symbol; component module layout reshuffled in 4.0 and again in 5.0's SSR rewrite — likely moves/renames silently** | re-verify | re-verify |
| 9 | `gr.Dropdown.get_expected_parent = FormComponent.get_expected_parent` — monkeypatching a classmethod onto `gr.Dropdown` | **P7: internal method `get_expected_parent` may not exist/behave the same after 4.0 component protocol rework — silent breakage (layout quirks) likely, not a hard crash** | re-verify | re-verify |
| 12-13 | `class ToolButton(FormComponent, gr.Button)` | **P7: component subclassing — base class MRO/behavior changes across major versions, re-validate constructor signature and render behavior each hop** | re-verify | re-verify |
| 23-24 | `class ResizeHandleRow(gr.Row)` | **P7, same concern, lower risk (gr.Row is a simpler layout component)** | re-verify | re-verify |
| 35-36,42-43,49-50 | `class FormRow(FormComponent, gr.Row)`, `FormColumn(FormComponent, gr.Column)`, `FormGroup(FormComponent, gr.Group)` | **P7** | re-verify | re-verify |
| 56-57 | `class FormHTML(FormComponent, gr.HTML)` | **P7** | re-verify | re-verify |
| 63-64 | `class FormColorPicker(FormComponent, gr.ColorPicker)` | **P7** | re-verify | re-verify |
| 70-71,79-80 | `class DropdownMulti(FormComponent, gr.Dropdown)`, `class DropdownEditable(FormComponent, gr.Dropdown)` | **P7** | re-verify | re-verify |
| 88-89 | `class InputAccordion(gr.Checkbox)` | **P7: InputAccordion wraps gr.Checkbox to fake an Accordion-like input; Checkbox internals/postprocess signature changes are the main risk** | re-verify | re-verify |
| 121 | `self.change(fn=None, _js='...')` | **P12** | re-verify | re-verify |
| 130 | `self.accordion = gr.Accordion(**kwargs_accordion)` | none identified unless `kwargs_accordion` carries a removed param | none identified | none identified |
| 146 | `gr.Column(elem_id=, elem_classes=, min_width=0)` | none identified | none identified | none identified |

### modules/ui_extensions.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 12 | `import gradio as gr` | none identified | none identified | none identified |
| 86 | `gr.Dropdown.update(value=, choices=)` | **P1: breaks in 4.0** | n/a once fixed | n/a |
| 437 | `gr.CheckboxGroup.update(choices=tags)` | **P1: breaks in 4.0** | n/a once fixed | n/a |
| 570 | `gr.Blocks(analytics_enabled=False) as ui` | none identified | none identified | see 6.0 app-level-param note |
| 571,617,681,694 | `gr.Tabs(elem_id=...)`, `gr.TabItem(...)` | none identified | none identified | none identified |
| 574,594,597,601... (many) | `gr.Row(...)` | none identified | none identified | none identified |
| 576-577,581,610,619,623,625,639,645,652,685,688,699,702,708,711 | `gr.Button(value=, variant=)` and `.click(fn=, inputs=, outputs=, show_progress=False)` | none identified | none identified | none identified |
| 578,598,602,605,611,627,630-631,698 | `gr.Radio(...)`, `gr.CheckboxGroup(value=, label=, choices=, elem_classes=...)` | none identified | none identified | none identified |
| 579-580,621-622,634,682-684,701 | `gr.Text(elem_id=, visible=, container=False, label=, value=, placeholder=...)` | none identified (uses `container=` which is stable) | none identified | none identified |
| 595,598,636-637,704-705 | `gr.HTML(...)` | none identified | none identified | none identified |
| 603,610-611,639-640,645-646,651-652,657-658,663-664,669-670,675-676,688-689 | `wrap_gradio_call(..., extra_outputs=[gr.update(), ...])` | none identified (P10) | none identified | none identified |
| 696 | `gr.Dropdown(label=, elem_id=, value=, choices=...)` | none identified | none identified | none identified |
| 710 | `gr.Label(visible=False)` | none identified | none identified | none identified |
| 711 | `.click(fn=, _js="config_state_confirm_restore", inputs=, outputs=)` | **P12** | re-verify | re-verify |
| 713 | `config_states_list.change(...)` | none identified (truncated context; pattern consistent with rest of file) | none identified | none identified |

### modules/ui_extra_networks.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 12 | `import gradio as gr` | none identified | none identified | none identified |
| 733 | `def create_ui(interface: gr.Blocks, ...)` type hint | none identified | none identified | none identified |
| 744 | `gr.Tab(page.title, elem_id=, elem_classes=...) as tab` | none identified | none identified | none identified |
| 745 | `gr.Column(elem_id=, elem_classes=...)` | none identified | none identified | none identified |
| 749 | `gr.HTML(page.create_html(...), elem_id=...)` | none identified | none identified | none identified |
| 756 | `gr.Button('Save preview', elem_id=, visible=False)` | none identified | none identified | none identified |
| 757 | `gr.Textbox('Preview save filename', elem_id=, visible=False)` | none identified | none identified | none identified |
| 760,769 | `tab.select(fn=None, _js=..., inputs=[], outputs=[], show_progress=False)` | **P12** | re-verify | re-verify |
| 777 | `gr.Button("Refresh", elem_id=, visible=False)` | none identified | none identified | none identified |
| 778 | `.click(fn=refresh, inputs=[], outputs=ui.pages).then(fn=lambda: None, _js="...").then(fn=lambda: None, _js='setupAllResizeHandles')` — chained `.then()` with `_js=` | **P12: two `_js=` sites; `.then()` chaining itself stable** | re-verify | re-verify |
| 788 | `interface.load(fn=pages_html, inputs=[], outputs=ui.pages).then(fn=lambda: None, _js='setupAllResizeHandles')` | **`Blocks.load()` used as an instance method (page-load event) — still valid in 4.0+ per official migration notes ("Blocks.load() can only be used as an instance method... this is unchanged"), so none identified for the `.load()` call itself; the chained `_js=` still carries P12 risk** | re-verify | re-verify |
| 828 | `ui.button_save_preview.click(...)` | none identified | none identified | none identified |

### modules/ui_extra_networks_checkpoints.py
No gradio usage found (file does not import gradio).

### modules/ui_extra_networks_checkpoints_user_metadata.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 1 | `import gradio as gr` | none identified | none identified | none identified |
| 37 | `gr.Row()` | none identified | none identified | none identified |
| 38 | `gr.Dropdown(choices=, value=, label=, elem_id=...)` | none identified | none identified | none identified |
| 41 | `gr.TextArea(label='Notes', lines=4)` | none identified | none identified | none identified |
| 55 | `.click(fn=self.put_values_into_components, inputs=, outputs=viewed_components)` | none identified | none identified | none identified |
| 56 | `.then(fn=lambda: gr.update(visible=True), inputs=[], outputs=[self.box])` | none identified (P10) | none identified | none identified |
| 65 | `.click(fn=self.update_vae, inputs=...)` | none identified | none identified | none identified |

### modules/ui_extra_networks_hypernets.py
No gradio usage found (file does not import gradio).

### modules/ui_extra_networks_textual_inversion.py
No gradio usage found (file does not import gradio).

### modules/ui_extra_networks_user_metadata.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 6 | `import gradio as gr` | none identified | none identified | none identified |
| 49-50,57 | `gr.Row()`, `gr.Column(scale=, min_width=0)` | none identified | none identified | none identified |
| 51,53,58,67 | `gr.HTML(elem_classes=...)` | none identified | none identified | none identified |
| 52,156 | `gr.Textbox(label=, lines=4)`, `gr.TextArea(label='Notes', lines=4)` | none identified | none identified | none identified |
| 62 | `gr.Row(elem_classes=...)` | none identified | none identified | none identified |
| 63-65 | `gr.Button(...)` | none identified | none identified | none identified |
| 69 | `.click(fn=None, _js="closePopup")` | **P12** | re-verify | re-verify |
| 150-151 | `.click(fn=func, inputs=, outputs=[]).then(fn=None, _js=..., inputs=, outputs=[])` | **P12** | re-verify | re-verify |
| 161-162 | `.click(fn=self.put_values_into_components, ...).then(fn=lambda: gr.update(visible=True), ...)` | none identified (P10) | none identified | none identified |
| 167 | `gr.Box(visible=False, elem_id=, elem_classes=...) as box` | **P2: `gr.Box` removed in Gradio 4.0 — must replace with `gr.Group`** | n/a once fixed | n/a |
| 170 | `gr.Textbox("Edit user metadata card id", visible=False, elem_id=...)` | none identified | none identified | none identified |
| 171 | `gr.Button("Edit user metadata", visible=False, elem_id=...)` | none identified | none identified | none identified |
| 197,202 | `.click(...)`, `.then(...)` | none identified unless `_js` used (not shown in excerpt) | none identified | none identified |

### modules/ui_gradio_extensions.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 2 | `import gradio as gr` | none identified | none identified | none identified |
| 44 | `from modules.shared_gradio_themes import resolve_var` (not a gradio API call itself — internal repo module) | n/a | n/a | n/a |
| 71 | `gr.routes.templates.TemplateResponse = template_response` — monkeypatch of internal FastAPI/Starlette template routing | **P9: `gr.routes` internals changed across 4.0 (new routing for file serving/`allowed_paths`) — moderate risk of silent breakage (custom template injection may stop working without error)** | **High risk: Gradio 5.0's SSR rewrite fundamentally changes how `gr.routes` renders templates server-side — this monkeypatch is very likely to break, possibly silently (templates render but without the custom injection)** | re-verify against 6.0 routing (`show_api`/`footer_links` changes touch this area too) |
| 75 | `shared.GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse` — saves original for chaining | same as above | same as above | same as above |

### modules/ui_loadsave.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 4 | `import gradio as gr` | none identified | none identified | none identified |
| 10 | comment: "gradio 3.41 changes choices from list of values to list of pairs" — documents a prior internal format change for `Dropdown.choices`; relevant context for verifying Dropdown behavior at each new target | n/a (historical note) | n/a | n/a |
| 52 | `isinstance(obj, gr.Accordion)` | none identified | none identified | none identified |
| 60,62 | `isinstance(obj, gr.Textbox)`, `isinstance(obj, gr.Number)` | none identified | none identified | none identified |
| 75,78,84,87,90,93,96,124,131,138 | `type(x) in [gr.Slider, gr.Radio, gr.Checkbox, gr.Textbox, gr.Number, gr.Dropdown, ToolButton, gr.Button]`, `type(x) == gr.Slider/.Radio/.Checkbox/.Textbox/.Number/.Dropdown/.Tabs`, `isinstance(x, gr.TabItem)`, `isinstance(x, gr.Tabs)`, `isinstance(x, gr.Button)` | **Moderate risk: exact `type(x) == gr.X` checks (not `isinstance`) are brittle against internal class hierarchy changes — if gradio wraps components differently (e.g. proxy/wrapper objects introduced in a major version), these exact-type checks can silently stop matching, causing saved UI defaults to silently not apply.** This is a **silent runtime behavior change risk**, not an import error. | re-verify same risk | re-verify same risk |
| 218 | `gr.HTML(...)` | none identified | none identified | none identified |
| 225-227,237-238 | `gr.Row()`, `gr.Button(value=, elem_id=, variant=)`, `.click(fn=, inputs=, outputs=)` | none identified | none identified | none identified |
| 229 | `gr.HTML("")` | none identified | none identified | none identified |

### modules/ui_postprocessing.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 1 | `import gradio as gr` | none identified | none identified | none identified |
| 8 | `gr.Label(visible=False)` | none identified | none identified | none identified |
| 9 | `gr.Number(value=0, visible=False)` | none identified | none identified | none identified |
| 12 | `gr.Column(variant='compact')` | none identified | none identified | none identified |
| 13 | `gr.Tabs(elem_id="mode_extras")` | none identified | none identified | none identified |
| 14,17,20 | `gr.TabItem(...)` | none identified | none identified | none identified |
| 15 | `gr.Image(label="Source", source="upload", interactive=True, type="pil", elem_id=, image_mode="RGBA")` | **P3: `source` -> `sources` rename — breaks in 4.0** | re-verify | re-verify |
| 18 | `gr.Files(label=, interactive=True, elem_id=...)` | none identified | none identified | none identified |
| 21-22 | `gr.Textbox(label=, **shared.hide_dirs, placeholder=, elem_id=...)` | **P5: `**shared.hide_dirs` kwargs passthrough risk** | none identified | none identified |
| 23 | `gr.Checkbox(label=, value=True, elem_id=...)` | none identified | none identified | none identified |
| 27 | `gr.Column()` | none identified | none identified | none identified |
| 34-36 | `.select(fn=lambda: N, inputs=[], outputs=[tab_index])` ×3 | none identified | none identified | none identified |
| 38-39 | `.click(fn=call_queue.wrap_gradio_gpu_call(...), ...)` | none identified | none identified | none identified |
| 61 | `extras_image.change(...)` | none identified (signature not shown, consistent with rest of file) | none identified | none identified |

### modules/ui_prompt_styles.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 1 | `import gradio as gr` | none identified | none identified | none identified |
| 15-18,23,32,53 | `gr.update(...)` returns | none identified (P10) | none identified | none identified |
| 49 | `[gr.Textbox.update(value=prompt), gr.Textbox.update(value=negative_prompt), gr.Dropdown.update(value=[])]` | **P1: three `.update()` classmethod calls — all break in Gradio 4.0; replace with `gr.update(value=...)` for each** | n/a once fixed | n/a |
| 62 | `gr.Row(elem_id=...)` | none identified | none identified | none identified |
| 63 | `gr.Dropdown(label="Styles", show_label=False, elem_id=, choices=, value=[], multiselect=True, tooltip="Styles")` | **P13: `tooltip=` kwarg on `gr.Dropdown` is non-standard pre-6.0 per public docs — verify it's accepted at target version or will raise under strict-kwargs (4.0) / be silently dropped** | re-verify | re-verify |
| 66 | `gr.Box(elem_id=, elem_classes="popup-dialog") as styles_dialog` | **P2: `gr.Box` removed in 4.0 — replace with `gr.Group`** | n/a once fixed | n/a |
| 67-68,73-74,76-77,79-82 | `gr.Row()`, `gr.Dropdown(..., allow_custom_value=True, info=...)`, `gr.Textbox(label=, show_label=True, elem_id=, lines=3, elem_classes=...)`, `gr.Button(..., variant=, visible=False)` | none identified | none identified | none identified |
| 84-96 | `.change(fn=, inputs=, outputs=, show_progress=False)`, `.click(fn=, inputs=, outputs=).then(refresh_styles, outputs=, show_progress=False)` | none identified | none identified | none identified |
| 98-104,108-118 | `.click(...)`, `.then(refresh_styles, ...)` | none identified | none identified | none identified |
| 123 | `.then(fn=None, _js="function(){update_"+self.tabname+"_tokens(); closePopup();}", show_progress=False)` | **P12** | re-verify | re-verify |

### modules/ui_settings.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 1 | `import gradio as gr` | none identified | none identified | none identified |
| 4,7 | internal repo imports referencing `wrap_gradio_call_no_job`, `reload_javascript` (not gradio API) | n/a | n/a | n/a |
| 18,92,95 | `gr.update(value=, **args)`, `gr.update(visible=True)`, `gr.update(value=getattr(...))` | none identified (P10) | none identified | none identified |
| 33,35,37 | `comp = gr.Textbox` / `gr.Number` / `gr.Checkbox` — stores the *class* (not instance) for later dynamic instantiation | none identified for the class reference itself; downstream instantiation subject to same per-component risk rows above | none identified | none identified |
| 118 | `gr.Blocks(analytics_enabled=False) as settings_interface` | none identified | none identified | see 6.0 app-level-param note |
| 119-120,122,135,146-147,149,174-175,178-179,182,186,189,193-194,196,318 | `gr.Row()`, `gr.Column(scale=...)`, `gr.Group()`, `gr.TabItem(elem_id=, label=)`, `gr.Tabs(elem_id=...)` | none identified | none identified | none identified |
| 121,123,183-185,187-188,190-191,194,196,202-203,349 | `gr.Button(value=, variant=, elem_id=, visible=False)` | none identified | none identified | none identified |
| 125,172,178,200 | `gr.HTML(elem_id=, value=...)` | none identified | none identified | none identified |
| 176 | `gr.File(label=, type='binary')` | **Verify: `type='binary'` on `gr.File` — stable param through 4.x+ per docs, none identified, but flag for explicit re-check since File's `type` enum values have shifted historically** | re-verify | re-verify |
| 192,197 | `gr.Number(value=, label=, elem_id=, precision=0, minimum=0)` | none identified | none identified | none identified |
| 204 | `self.show_one_page.click(lambda: None)` | none identified | none identified | none identified |
| 206,208 | `gr.Textbox(value=, elem_id=, max_lines=1, placeholder=, show_label=False, visible=False)` | none identified | none identified | none identified |
| 224,230,236,242,248,254,261,272,278,294,310,324,350 | `.click(fn=, inputs=, outputs=...)` | none identified | none identified | none identified |
| 237 | `(sd_models.list_loaded_models(), gr.Number.update(maximum=get_max_model_index()))` | **P1: `.update()` classmethod — breaks in 4.0; replace with `gr.update(maximum=...)`** | n/a once fixed | n/a |
| 325 | `wrap_gradio_call_no_job(..., extra_outputs=[gr.update()])` | none identified (P10) | none identified | none identified |
| 334 | `isinstance(component, gr.Textbox)` | none identified | none identified | none identified |
| 372 | `[gr.update(visible=...) for comp in self.components]` | none identified (P10) | none identified | none identified |

### modules/ui_tempdir.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 6 | `import gradio.components` | none identified at import level; see below for the attribute accesses | | |
| 16-21 | `register_tmp_file(gradio, filename)`: `hasattr(gradio, 'temp_file_sets')` (gradio 3.15), `hasattr(gradio, 'temp_dirs')` (gradio 3.9) — directly probing **undocumented internal module-level state** on the `gradio` module object | **P8: High risk. These internal attributes are almost certainly reorganized by 4.0's new temp-file/`allowed_paths` handling (4.0 changelog explicitly reworks file serving: "working directory is now not served by default... use allowed_paths"). The code's own comments show it already had to adapt per-minor-version (3.9 vs 3.15); expect this to require a rewrite, not just a verification, for 4.0+.** | **Further risk: 5.0 SSR rewrite likely changes file-serving internals again** | re-verify against 6.0 |
| 24-28,37 | `check_tmp_file(gradio, filename)` — same `temp_file_sets`/`temp_dirs` attribute probing | **P8, same as above** | re-verify | re-verify |
| 70 | `gradio.components.IOComponent.pil_to_temp_file = save_pil_to_file` — monkeypatches `IOComponent`, a core private base class, to override image-to-tempfile serialization | **P8/P7: `IOComponent` base class significantly reorganized in Gradio 4.0's new component protocol — this monkeypatch target may not exist post-4.0, causing previews/temp images to silently fail to clean up or render incorrectly rather than crash** | **5.0 SSR rewrite: further restructuring expected, re-verify monkeypatch target exists** | re-verify against 6.0 |
| 97,99,104-107 | `is_gradio_temp_path(path)`: checks `os.environ.get("GRADIO_TEMP_DIR")` and `Path(tempfile.gettempdir()) / "gradio"` — environment-variable / filesystem-path based, not a direct gradio API call | none identified (relies on documented `GRADIO_TEMP_DIR` env var, stable across versions) | none identified | none identified |

### modules/ui_toprow.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 1 | `import gradio as gr` | none identified | none identified | none identified |
| 47,68,81-82,86,97,117,135 | `gr.Row(elem_id=, variant="compact", elem_classes=...)` | none identified | none identified | none identified |
| 55,69,71 | `gr.Column(scale=, elem_id=, elem_classes=...)` | none identified | none identified | none identified |
| 83,87 | `gr.Textbox(label=, elem_id=, show_label=False, lines=3, placeholder=, elem_classes=...)` | none identified | none identified | none identified |
| 84 | `gr.File(label="", elem_id=, file_count="single", type="binary", visible=False)` | none identified (no `source=`) | none identified | none identified |
| 89 | `self.prompt_img.change(...)` | none identified | none identified | none identified |
| 100-103 | `gr.Button('Interrupt'/'Skip'/'Interrupting...'/'Generate', elem_id=, elem_classes=, variant=, tooltip=...)` | **P13-adjacent: `tooltip=` kwarg on `gr.Button` — verify accepted at target version; non-standard pre-6.0 per public docs** | re-verify | re-verify |
| 108 | `gr.Info("Generation will stop after finishing this image, click again to stop immediately.")` | none identified (stable toast API through 6.x) | none identified | none identified |
| 112-114 | `.click(fn=shared.state.skip)`, `.click(fn=interrupt_function, _js='...')`, `.click(fn=interrupt_function)` | **P12 on line 113's `_js=`** | re-verify | re-verify |
| 130,132 | `gr.HTML(value=, elem_id=, elem_classes=, visible=False)` | none identified | none identified | none identified |
| 131,133 | `gr.Button(visible=False, elem_id=...)` | none identified | none identified | none identified |
| 135 | `self.clear_prompt_button.click(...)` | none identified (signature not fully shown, consistent pattern) | none identified | none identified |

---

### extensions-builtin/mahiro_reforge/scripts/mahiro_cfg_script.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 4 | `import gradio as gr` | none identified | none identified | none identified |
| 23 | `gr.Accordion(open=False, label=self.title())` | none identified | none identified | none identified |
| 24 | `gr.HTML(...)` | none identified | none identified | none identified |
| 25 | `gr.Checkbox(label=, value=self.enabled)` | none identified | none identified | none identified |
| 27 | `enabled.change(...)` | none identified | none identified | none identified |

### extensions-builtin/reForge-RescaleCFG/scripts/advanced_model_sampling_script.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 4 | `import gradio as gr` | none identified | none identified | none identified |
| 26 | `gr.Accordion(open=False, label=...)` | none identified | none identified | none identified |
| 27 | `gr.HTML(...)` | none identified | none identified | none identified |
| 28 | `gr.Checkbox(label=, value=...)` | none identified | none identified | none identified |
| 29 | `gr.Dropdown(...)` | none identified | none identified | none identified |
| 34 | `gr.Slider(label=, minimum=, maximum=, step=, value=...)` | none identified | none identified | none identified |
| 36 | `enabled.change(...)` | none identified | none identified | none identified |

### extensions-builtin/reForge-advanced_model_sampling_backported/scripts/advanced_model_sampling_script_backported.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 2 | `import gradio as gr` | none identified | none identified | none identified |
| 24 | `gr.Accordion(open=False, label=...)` | none identified | none identified | none identified |
| 25 | `gr.HTML(...)` | none identified | none identified | none identified |
| 27 | `gr.Checkbox(label=, value=...)` | none identified | none identified | none identified |
| 29 | `gr.Radio(...)` | none identified | none identified | none identified |
| 35 | `gr.Group(visible=True) as discrete_group` | none identified | none identified | none identified |
| 36 | `gr.Radio(...)` | none identified | none identified | none identified |
| 41 | `gr.Checkbox(label='Zero SNR', value=...)` | none identified | none identified | none identified |
| 43 | `gr.Group(visible=False) as continuous_edm_group` | none identified | none identified | none identified |
| 44,49,56 | `gr.Radio(...)`, `gr.Slider(...)` ×2 | none identified | none identified | none identified |
| 66-67 | `gr.Group.update(visible=(mode == "Discrete"))`, `gr.Group.update(visible=(mode == "Continuous EDM"))` | **P1: `.update()` classmethod — breaks in 4.0; replace with `gr.update(visible=...)` or return fresh `gr.Group(...)` instances** | n/a once fixed | n/a |
| 70 | `sampling_mode.change(...)` | none identified | none identified | none identified |

### extensions-builtin/sd_forge_svd/scripts/forge_svd.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 2 | `import gradio as gr` | none identified | none identified | none identified |
| 62 | `gr.Blocks() as svd_block` | none identified | none identified | see 6.0 app-level-param note |
| 64,105 | `gr.Column()` | none identified | none identified | none identified |
| 65 | `gr.Image(label='Input Image', source='upload', type='numpy', height=400)` | **P3: `source` -> `sources` rename — breaks in 4.0** | re-verify | re-verify |
| 67 | `gr.Row()` | none identified | none identified | none identified |
| 68 | `gr.Dropdown(label="SVD Checkpoint Filename", ...)` | none identified | none identified | none identified |
| 72-73 | `refresh_button.click(fn=lambda: gr.update(choices=update_svd_filenames()))` | none identified (P10) | none identified | none identified |
| 76-86 | `gr.Slider(label=, minimum=, maximum=, step=, value=...)` ×8 | none identified | none identified | none identified |
| 87,94 | `gr.Radio(label=, choices=...)` | none identified | none identified | none identified |
| 97 | `gr.Number(label='Seed', value=, precision=0)` | none identified | none identified | none identified |
| 99 | `gr.Button(value="Generate")` | none identified | none identified | none identified |
| 106 | `gr.Video(autoplay=True)` | none identified (no `source`/subtitle tuple use — not affected by P3 or the 6.0 Video-tuple removal) | none identified | none identified |
| 107 | `gr.Gallery(label=, show_label=False, object_fit='contain', ...)` | none identified | none identified | re-verify Gallery param list at 6.0 |
| 110 | `generate_button.click(predict, inputs=ctrls, outputs=[...])` | none identified | none identified | none identified |

### extensions-builtin/sd_forge_hypertile/scripts/forge_hypertile.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 1 | `import gradio as gr` | none identified | none identified | none identified |
| 20 | `gr.Accordion(open=False, label=...)` | none identified | none identified | none identified |
| 21,25 | `gr.Checkbox(label=, value=False)` | none identified | none identified | none identified |
| 22-24 | `gr.Slider(label=, minimum=, maximum=, step=, value=...)` ×3 | none identified | none identified | none identified |

### extensions-builtin/ScuNET/scripts/scunet_model.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 68 | `import gradio as gr` | none identified | none identified | none identified |
| 70-71 | `shared.OptionInfo(..., gr.Slider, {"minimum":..., "maximum":..., "step":...}, section=...)` — passes the `gr.Slider` **class** (not instance) to the repo's own settings-registration framework, which instantiates it dynamically later | none identified for the class reference itself; actual instantiation happens in `modules/ui_settings.py` (see its row for `comp = gr.Slider`-style dynamic use) — same component-kwarg risk applies downstream | none identified | none identified |

### extensions-builtin/reForge-APGIsYourCFG/scripts/APG_CFGGuidance_script.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 6 | `import gradio as gr` | none identified | none identified | none identified |
| 32 | `gr.Accordion(open=False, label=...)` | none identified | none identified | none identified |
| 34,67 | `gr.HTML(...)` | none identified | none identified | none identified |
| 35,68 | `gr.Checkbox(label=, value=...)` | none identified | none identified | none identified |
| 36,72 | `gr.Group(visible=True)` | none identified | none identified | none identified |
| 37,44,51,58,73,80 | `gr.Slider(label=, minimum=, maximum=, step=, value=...)` ×6 | none identified | none identified | none identified |

### extensions-builtin/extra-options-section/scripts/extra_options_section.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 3 | `import gradio as gr` | none identified | none identified | none identified |
| 30 | `gr.Blocks() as interface` | none identified | none identified | see 6.0 app-level-param note |
| 31 | `gr.Accordion("Options", open=False, elem_id=...) if ... else gr.Group(elem_id=...)` | none identified | none identified | none identified |
| 36 | `gr.Row()` | none identified | none identified | none identified |
| 78 | `shared.OptionInfo(1, ..., gr.Slider, {"step":1,"minimum":1,"maximum":20})` — passes class, dynamically instantiated elsewhere | none identified directly (downstream risk only, see ui_settings.py row) | none identified | none identified |

### extensions-builtin/sd_forge_sure_ag/scripts/forge_sure_ag.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 1 | `import gradio as gr` | none identified | none identified | none identified |
| 20 | `gr.Accordion(open=False, label=...)` | none identified | none identified | none identified |
| 21 | `gr.Checkbox(label="Enabled", value=False)` | none identified | none identified | none identified |
| 22,33 | `gr.Row()` | none identified | none identified | none identified |
| 23,28 | `gr.Slider(...)` | none identified | none identified | none identified |
| 34 | `gr.Radio(...)` | none identified | none identified | none identified |
| 40 | `gr.Slider(...)` | none identified | none identified | none identified |

### extensions-builtin/sd_forge_kohya_hrfix/scripts/kohya_hrfix.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 1 | `import gradio as gr` | none identified | none identified | none identified |
| 21 | `gr.Accordion(open=False, label=...)` | none identified | none identified | none identified |
| 22,27 | `gr.Checkbox(label=, value=...)` | none identified | none identified | none identified |
| 23-26 | `gr.Slider(label=, value=, minimum=, maximum=, step=...)` ×4 | none identified | none identified | none identified |
| 28-29 | `gr.Radio(label=, choices=, value=...)` ×2 | none identified | none identified | none identified |

### extensions-builtin/LDSR/scripts/ldsr_model.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 64 | `import gradio as gr` | none identified | none identified | none identified |
| 66-67 | `shared.OptionInfo(..., gr.Slider, {...})`, `shared.OptionInfo(..., gr.Checkbox, {"interactive": True})` — class references, dynamically instantiated | none identified directly (downstream risk only) | none identified | none identified |

### extensions-builtin/sd_forge_neveroom/scripts/forge_never_oom.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 1 | `import gradio as gr` | none identified | none identified | none identified |
| 59 | `gr.Accordion(open=False, label=...)` | none identified | none identified | none identified |
| 60-61 | `gr.Checkbox(label=, value=False)` ×2 | none identified | none identified | none identified |
| 62-63 | `gr.Slider(label=, minimum=, maximum=, step=, value=...)` ×2 | none identified | none identified | none identified |

### extensions-builtin/sd_forge_multidiffusion/scripts/forge_multidiffusion.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 1 | `import gradio as gr` | none identified | none identified | none identified |
| 21 | `gr.Accordion(open=False, label=...)` | none identified | none identified | none identified |
| 22 | `gr.Checkbox(label=, value=False)` | none identified | none identified | none identified |
| 23 | `gr.Radio(label='Method', ...)` | none identified | none identified | none identified |
| 26,29,32 | `gr.Row()` ×3 | none identified | none identified | none identified |
| 27-28,30-31 | `gr.Slider(...)` ×4 | none identified | none identified | none identified |
| 33 | `gr.Radio(label='Shift Method (SpotDiffusion)', ...)` | none identified | none identified | none identified |

### extensions-builtin/sd_forge_clpc/scripts/forge_clpc.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 8 | `import gradio as gr` | none identified | none identified | none identified |
| 24 | `gr.Accordion(open=False, label=...)` | none identified | none identified | none identified |
| 25 | `gr.Markdown(...)` | none identified | none identified | **5->6: HTML/Markdown `padding` default change discussed in 6.0 guide — visual-only, not a hard break** |
| 31 | `gr.Radio(...)` | none identified | none identified | none identified |
| 43,64,83,115,124,133 | `gr.Row()` ×6 | none identified | none identified | none identified |
| 44,53 | `gr.Checkbox(...)` ×2 | none identified | none identified | none identified |
| 65,84,100,116,118,120,127(no,Number),133... | `gr.Slider(...)` (many) | none identified | none identified | none identified |
| 77,139 | `gr.Checkbox(...)` ×2 | none identified | none identified | none identified |
| 125-126 | `gr.Number(label="atol"/"rtol", value=, precision=6)` | none identified | none identified | none identified |
| 144 | `gr.Slider(label="Max steps (hard limit)", ...)` | none identified | none identified | none identified |

### extensions-builtin/sd_webui_random_resolutions/scripts/random_res_script.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 2 | `import gradio as gr` | none identified | none identified | none identified |
| 71 | `gr.Accordion("Random Resolution", open=False)` | none identified | none identified | none identified |
| 72,95,100 | `gr.Column(...)` | none identified | none identified | none identified |
| 73 | `gr.Checkbox(False, label=...)` | none identified | none identified | none identified |
| 75,81,87,94,109,114 | `gr.Row()` ×6 | none identified | none identified | none identified |
| 76-77 | `gr.Radio(choices=, value=, label=...)` ×2 | none identified | none identified | none identified |
| 82,84,105 | `gr.Slider(minimum=, maximum=, step=, value=, label=...)` ×3 | none identified | none identified | none identified |
| 88 | `gr.Textbox(...)` | none identified | none identified | none identified |
| 96-97 | `gr.Number(label=, precision=0)` ×2 | none identified | none identified | none identified |
| 98,107,110-112,115-116 | `gr.Button(...)` (many) | none identified | none identified | none identified |
| 101 | `gr.Dropdown(...)` | none identified | none identified | none identified |
| 125 | `return gr.update()` | none identified (P10) | none identified | none identified |
| 219,224,228,232,236,240,244,248,252 | `.change(fn=, ...)`, `.click(fn=, ...)` | none identified | none identified | none identified |

### extensions-builtin/SwinIR/scripts/swinir_model.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 91 | `import gradio as gr` | none identified | none identified | none identified |
| 93-95 | `shared.OptionInfo(..., gr.Slider, {...})` ×2, `shared.OptionInfo(..., gr.Checkbox, {"interactive": True})` — class references, dynamically instantiated | none identified directly (downstream risk only) | none identified | none identified |

### extensions-builtin/sd_forge_sure_wav_ag/scripts/forge_sure_wav_ag.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 1 | `import gradio as gr` | none identified | none identified | none identified |
| 19 | `gr.Accordion(open=False, label=...)` | none identified | none identified | none identified |
| 20 | `gr.Checkbox(label="Enabled", value=False)` | none identified | none identified | none identified |
| 21,38,50,66 | `gr.Row()` ×4 | none identified | none identified | none identified |
| 22,39,51,61,67,72,77 | `gr.Slider(...)` (many) | none identified | none identified | none identified |
| 27,44 | `gr.Radio(...)` ×2 | none identified | none identified | none identified |
| 56 | `gr.Dropdown(...)` | none identified | none identified | none identified |

### extensions-builtin/sd_forge_latent_modifier/scripts/forge_latent_modifier.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 1 | `import gradio as gr` | none identified | none identified | none identified |
| 25 | `gr.Accordion(open=False, label=...)` | none identified | none identified | none identified |
| 27,43,58,79,96 | `gr.Tab("...")` ×5 | none identified | none identified | none identified |
| 28,44,61,72,86 | `gr.Markdown("""...""")` (many) | none identified | none identified | **5->6: HTML/Markdown padding default change — visual-only** |
| 32,48,59,80,97 | `gr.Checkbox(label=, value=False)` ×5 | none identified | none identified | none identified |
| 33,49,60,81,98 | `gr.Group(visible=True)` ×5 | none identified | none identified | none identified |
| 34,50,55-56,62,65,69-70,73,76-77,83-84,90,93-94,100 | `gr.Slider(label=, minimum=, maximum=, step=, value=...)` (very many) | none identified | none identified | none identified |
| 35,38,51,65,73,90,100 | `gr.Radio(label=, choices=...)` (many) | none identified | none identified | none identified |

### extensions-builtin/Lora/scripts/lora_script.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 3 | `import gradio as gr` | none identified | none identified | none identified |
| 36-37,41-42 | `shared.OptionInfo(..., gr.Dropdown, lambda: {...}, ...)`, `shared.OptionInfo(..., gr.Radio, {...})`, `shared.OptionInfo(..., gr.CheckboxGroup, {...})`, `shared.OptionInfo(..., gr.Number, {"precision": 0})` — class references, dynamically instantiated elsewhere | none identified directly (downstream risk only) | none identified | none identified |
| 62 | `def api_networks(_: gr.Blocks, app: FastAPI)` type hint | none identified | none identified | none identified |

### extensions-builtin/Lora/ui_edit_user_metadata.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 5 | `import gradio as gr` | none identified | none identified | none identified |
| 125 | `gradio_tags = [(tag, str(count)) ...]` (local var name only, not an API call) | n/a | n/a | n/a |
| 130 | `gr.HighlightedText.update(value=gradio_tags, visible=...)` | **P1: `.update()` classmethod — breaks in 4.0; replace with `gr.update(value=..., visible=...)`** | n/a once fixed | n/a |
| 134-135 | `gr.update(visible=...)`, `gr.update(value=..., visible=...)` | none identified (P10) | none identified | none identified |
| 163 | `gr.Dropdown([...], value=, label=, interactive=True)` | none identified | none identified | none identified |
| 168 | `gr.HighlightedText(label="Training dataset tags")` | none identified | none identified | none identified |
| 169,171 | `gr.Text(label=, info=...)` ×2 | none identified | none identified | none identified |
| 170 | `gr.Slider(label=, info=, minimum=, maximum=, step=...)` | none identified | none identified | none identified |
| 172,176 | `gr.Row()`, `gr.Column(scale=, min_width=...)` | none identified | none identified | none identified |
| 174 | `gr.Textbox(label=, lines=4, max_lines=4, interactive=False)` | none identified | none identified | none identified |
| 177 | `gr.Button('Generate', size="lg", scale=1)` | none identified (`size=` is a valid Button param through current versions) | none identified | none identified |
| 179 | `gr.TextArea(label='Notes', lines=4)` | none identified | none identified | none identified |
| 181 | `.click(fn=, inputs=, outputs=, show_progress=False)` | none identified | none identified | none identified |
| 183,193 | `evt: gr.SelectData` parameter type, `self.taginfo.select(fn=, inputs=, outputs=, show_progress=False)` | **P11: stable, none identified** | none identified | none identified |
| 213-214 | `.click(fn=, inputs=, outputs=viewed_components).then(fn=lambda: gr.update(visible=True), inputs=[], outputs=[self.box])` | none identified (P10) | none identified | none identified |

### extensions-builtin/sd_forge_z123/scripts/forge_z123.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 2 | `import gradio as gr` | none identified | none identified | none identified |
| 55 | `gr.Blocks() as model_block` | none identified | none identified | see 6.0 app-level-param note |
| 57,92 | `gr.Column()` | none identified | none identified | none identified |
| 58 | `gr.Image(label='Input Image', source='upload', type='numpy', height=400)` | **P3: `source` -> `sources` rename — breaks in 4.0** | re-verify | re-verify |
| 60 | `gr.Row()` | none identified | none identified | none identified |
| 61 | `gr.Dropdown(label="Zero123 Checkpoint Filename", ...)` | none identified | none identified | none identified |
| 65-66 | `refresh_button.click(fn=lambda: gr.update(choices=update_model_filenames))` | none identified (P10) | none identified | none identified |
| 69-76 | `gr.Slider(label=, minimum=, maximum=, step=, value=...)` ×7 | none identified | none identified | none identified |
| 77,84 | `gr.Radio(label=, choices=...)` ×2 | none identified | none identified | none identified |
| 87 | `gr.Number(label='Seed', value=, precision=0)` | none identified | none identified | none identified |
| 88 | `gr.Button(value="Generate")` | none identified | none identified | none identified |
| 93 | `gr.Gallery(label=, show_label=False, object_fit='contain', ...)` | none identified | none identified | re-verify Gallery param list at 6.0 |
| 96 | `generate_button.click(predict, inputs=ctrls, outputs=[output_gallery])` | none identified | none identified | none identified |

### extensions-builtin/reForge-advanced_model_sampling/scripts/advanced_model_sampling_script.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 4 | `import gradio as gr` | none identified | none identified | none identified |
| 279 | `gr.Accordion(open=False, label=...)` | none identified | none identified | none identified |
| 280 | `gr.HTML(...)` | none identified | none identified | none identified |
| 282,296 | `gr.Checkbox(label=, value=...)` ×2 | none identified | none identified | none identified |
| 284,291,299 | `gr.Radio(...)` ×3 | none identified | none identified | none identified |
| 290,298,307,311,314,317,320 | `gr.Group(visible=(initial_mode == "..."))` ×7 | none identified | none identified | none identified |
| 304-305,308-309,312,315,318,321-324 | `gr.Slider(label=, minimum=, maximum=, step=, value=...)` (many) | none identified | none identified | none identified |
| 328-334 | `gr.Group.update(visible=...)` ×7 (one per mode) | **P1: all 7 break in 4.0; replace with `gr.update(visible=...)` or component-return pattern** | n/a once fixed | n/a |
| 337 | `sampling_mode.change(...)` | none identified | none identified | none identified |

### extensions-builtin/soft-inpainting/scripts/soft_inpainting.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 2 | `import gradio as gr` | none identified | none identified | none identified |
| 534 | `gr.Group()` | none identified | none identified | none identified |
| 535,566,599,611,621,633,642,652,662 | `gr.Markdown(...)` (many, mostly help text) | none identified | none identified | **5->6: HTML/Markdown padding default change — visual-only** |
| 542,550,558,572,581,590 | `gr.Slider(label=...)` ×6 | none identified | none identified | none identified |
| 598 | `gr.Accordion("Help", open=False)` | none identified | none identified | none identified |

### extensions-builtin/sd_forge_controlnet_example/scripts/sd_forge_controlnet_example.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 4 | `import gradio as gr` | none identified | none identified | none identified |
| 28 | `gr.Accordion(open=False, label=...)` | none identified | none identified | none identified |
| 29-30 | `gr.HTML(...)` ×2 | none identified | none identified | none identified |
| 31 | `gr.Image(source='upload', type='numpy')` | **P3: `source` -> `sources` rename — breaks in 4.0** | re-verify | re-verify |
| 32 | `gr.Slider(label=...)` | none identified | none identified | none identified |

### extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/openpose_editor.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 2 | `import gradio as gr` | none identified | none identified | none identified |
| 45 | `gr.Button(visible=False, elem_classes=...)` | none identified | none identified | none identified |
| 48 | `gr.Textbox(visible=False, elem_classes=...)` | none identified | none identified | none identified |
| 60,68 | `gr.HTML(...)` ×2 | none identified | none identified | none identified |
| 79-81 | type hints: `generated_image: gr.Image`, `use_preview_as_input: gr.Checkbox`, `model: gr.Dropdown` | none identified | none identified | none identified |
| 91,107,109,119,138,148,150 | `gr.update(...)` returns (several), docstring reference "An gr.update event" | none identified (P10) | none identified | none identified |
| 112 | `self.render_button.click(...)` | none identified | none identified | none identified |
| 121 | `model.change(fn=update_upload_link, inputs=[model], outputs=[self.upload_link])` | none identified | none identified | none identified |

### extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/modal.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 1 | `import gradio as gr` | none identified | none identified | none identified |
| 5 | `class ModalInterface(gr.Interface)` | **P7: subclassing `gr.Interface` directly — `Interface` internals were touched across 4.0 (Interface.load() removed, default api_name/caching changes) and 6.0 (default API name now the function name, not "predict") — a subclass relying on specific internal method overrides is at risk of silent behavior change (e.g. different auto-generated API names, changed caching defaults) rather than a hard crash. Needs explicit re-validation.** | re-verify | re-verify (6.0: default `api_name` for Interface now uses function name, not `predict` — could silently change any API consumer expecting the old name) |
| 38 | `return gr.HTML(value=html_code, visible=visible)` | none identified | none identified | none identified |

### extensions-builtin/sd_forge_sure_token_ag/scripts/forge_sure_token_ag.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 3 | `import gradio as gr` | none identified | none identified | none identified |
| 30 | `gr.Accordion(open=False, label=...)` | none identified | none identified | none identified |
| 31 | `gr.Markdown(...)` | none identified | none identified | **5->6: padding default change — visual-only** |
| 39,94 | `gr.Checkbox(label=, value=...)` ×2 | none identified | none identified | none identified |
| 40,52,62,71,87 | `gr.Row()` ×5 | none identified | none identified | none identified |
| 41 | `gr.Checkbox(...)` | none identified | none identified | none identified |
| 47 | `gr.CheckboxGroup(...)` | none identified | none identified | none identified |
| 53,59,63,67,72,79 | `gr.Slider(...)` (many) | none identified | none identified | none identified |
| 88 | `gr.Radio(...)` | none identified | none identified | none identified |

### extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/photopea.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 1 | `import gradio as gr` | none identified | none identified | none identified |
| 129 | `gr.Group(elem_classes=...)` | none identified | none identified | none identified |
| 148 | `gr.HTML(...)` | none identified | none identified | none identified |
| 155 | type hint `generated_image: gr.Image` | none identified | none identified | none identified |
| 171 | `gr.Image(...)` (invisible mirror target, per docstring) | none identified from shown kwargs (no `source=` in the excerpt — re-check full constructor call for `source=`) | re-verify | re-verify |
| 178 | `output.upload(...)` | none identified | none identified | none identified |

### extensions-builtin/sd_forge_controlnet/lib_controlnet/api.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 7 | `import gradio as gr` | none identified | none identified | none identified |
| 40 | `def controlnet_api(_: gr.Blocks, app: FastAPI)` type hint only | none identified | none identified | none identified |

### extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/preset.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 2 | `import gradio as gr` | none identified | none identified | none identified |
| 49,83 | `gr.Row()` ×2 | none identified | none identified | none identified |
| 50 | `gr.Dropdown(...)` | none identified | none identified | none identified |
| 79 | `gr.Box(...)` | **P2: `gr.Box` removed in 4.0 — replace with `gr.Group`** | n/a once fixed | n/a |
| 84 | `gr.Textbox(...)` | none identified | none identified | none identified |
| 99 | type hint `control_type: gr.Radio` | none identified | none identified | none identified |
| 108,110,131,133,156-157,159,175,182,188-190,213,227,232,243,252,265,310 | `gr.update(...)`, `gr.skip()` | `gr.skip()` is a stable sentinel (added pre-4.0, kept through 6.x) — none identified; `gr.update(...)` none identified (P10) | none identified | none identified |
| 174 | `.then(...)` | none identified | none identified | none identified |
| 193,215,236,243 | `.click(...)` (several) | none identified | none identified | none identified |
| 210 | `return gr.Dropdown.update(...)` | **P1: `.update()` classmethod — breaks in 4.0; replace with `gr.update(...)`** | n/a once fixed | n/a |
| 268,272 | `isinstance(ui_state, gr.Image)`, `isinstance(ui_state, gr.Slider)` | none identified | none identified | none identified |

### extensions-builtin/sd_forge_controlnet/lib_controlnet/infotext.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 3 | `import gradio as gr` | none identified | none identified | none identified |
| 58 | `List[Tuple[gr.components.IOComponent, str]]` type hint — references internal `IOComponent` | **P7: `gr.components.IOComponent` is a private internal type whose hierarchy was reorganized in 4.0's new component protocol — type-hint usage itself won't crash, but anything downstream relying on `IOComponent`-specific attributes needs re-validation** | re-verify | re-verify |

### extensions-builtin/sd_forge_freeu/scripts/forge_freeu.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 1 | `import gradio as gr` | none identified | none identified | none identified |
| 62 | `gr.Accordion(open=False, label=...)` | none identified | none identified | none identified |
| 63 | `gr.Checkbox(label='Enabled', value=False)` | none identified | none identified | none identified |
| 64-67 | `gr.Slider(label=, minimum=, maximum=, step=, value=...)` ×4 | none identified | none identified | none identified |

### extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/controlnet_ui_group.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 3 | `import gradio as gr` | none identified | none identified | none identified |
| 34-53 | `Optional[gr.components.IOComponent]` type hints (many) | **P7: `IOComponent` internal type — see infotext.py note above** | re-verify | re-verify |
| 56,64 | `Tuple[gr.components.IOComponent]` return type hints | **P7, same as above** | re-verify | re-verify |
| 88 | `def set_component(self, component: gr.components.IOComponent)` | **P7, same as above** | re-verify | re-verify |
| 136 | `gr.Textbox(...)` (global dummy component) | none identified | none identified | none identified |
| 229,236,652-653,682 | `gr.State(InputMode.SIMPLE)`, `gr.State(None)`, `gr.State("")`, `gr.State(self.default_unit)` | none identified | none identified | none identified |
| 256 | `gr.Number(value=0, visible=False)` (dummy update trigger) | none identified | none identified | none identified |
| 259,264,284,296,310 | `gr.Group(visible=..., elem_classes=...)` | none identified | none identified | none identified |
| 260 | `gr.Checkbox(value=True, visible=False)` | none identified | none identified | none identified |
| 261,344,351 | `gr.Tabs()` | none identified | none identified | none identified |
| 262,329,343 | `gr.Tab(label=...) as self.X_tab` | none identified | none identified | none identified |
| 263,284(sub),296(sub),310(sub),330,336,344,352,358,377,387,416,422,427,434,441,448,459,466,475,501,511,520,531,560,622,631 | `gr.Row(...)`, `gr.Accordion(...)`, `gr.Column(visible=False)` | none identified | none identified | none identified |
| 265,287,313 | `gr.Image(...)` — need to verify exact kwargs for `source=`/`sources=` in full file (not captured in this excerpt — flag for explicit re-check since sibling ControlNet image components elsewhere in the repo do use `source=`) | **P3 risk: re-check full constructor kwargs; if `source=` is present, breaks in 4.0** | re-verify | re-verify |
| 304 | `gr.HTML(...)` | none identified | none identified | none identified |
| 331,336,460,476,489,568,576 | `gr.Textbox(...)`, `gr.Dropdown(...)` | none identified | none identified | none identified |
| 361,369,502,511,520,532,541,550 | `gr.Slider(...)` (many) | none identified | none identified | none identified |
| 378,382 | `gr.Button(...)` ×2 | none identified | none identified | none identified |
| 416,422,427,434,441,450 | `gr.Checkbox(...)` (many) | none identified | none identified | none identified |
| 467,561,622,631 | `gr.Radio(...)` (many) | none identified | none identified | none identified |
| 592,594,796-802,869,882-884,943,945,947,974,976,978,980,982,1027-1030,1051,1064-1070,1077-1083,1103,1105,1148,1181-1183 | `gr.update(...)` (very many sites) | none identified (P10) | none identified | none identified |
| 609,770,845-846,850,853 | `gr.Dropdown.update(...)` | **P1: multiple `.update()` classmethod sites — all break in 4.0; replace with `gr.update(...)`** | n/a once fixed | n/a |
| 611-615 | `.change(fn=, inputs=, outputs=)` (several) | none identified | none identified | none identified |
| 688,694,1164 | `isinstance(comp, gr.Slider) and hasattr(comp, "release")` | none identified | none identified | none identified |
| 711,738,754,763,774,820,823,858,871,950,999,1005,1017,1032,1050,1086,1100 | `.click(...)`, `.change(...)` event wiring (very many) | none identified | none identified | none identified |
| 736 | `return gr.Slider.update(), gr.Slider.update()` | **P1: breaks in 4.0** | n/a once fixed | n/a |
| 1000,1006 | `lambda: gr.Accordion.update(visible=True)`, `lambda: gr.Accordion.update(visible=False)` | **P1: breaks in 4.0** | n/a once fixed | n/a |
| 1013 | `gr.Accordion.update(...)` (multi-line call) | **P1: breaks in 4.0** | n/a once fixed | n/a |
| 1260 | `input_tab.select(...)` | none identified | none identified | none identified |

### extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/multi_inputs_gallery.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 1 | `import gradio as gr` | none identified | none identified | none identified |
| 20 | `gr.Group(**self.group_kwargs) as self.group` | none identified unless `group_kwargs` carries a removed param | none identified | none identified |
| 21 | `gr.Column()` | none identified | none identified | none identified |
| 22 | `gr.Gallery(...)` | none identified | none identified | re-verify Gallery param list at 6.0 |
| 29 | `gr.Row()` | none identified | none identified | none identified |
| 30 | `gr.UploadButton(...)` | none identified | none identified | none identified |
| 35 | `gr.Button("Clear Images")` | none identified | none identified | none identified |
| 45 | `self.clear_button.click(...)` | none identified | none identified | none identified |
| 56 | `self.upload_button.upload(...)` | none identified | none identified | none identified |
| 65 | `handle.then(**change_trigger)` | none identified | none identified | none identified |

### extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/tool_button.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 1 | `import gradio as gr` | none identified | none identified | none identified |
| 3 | `class ToolButton(gr.Button, gr.components.FormComponent)` | **P7: multiple inheritance from `gr.Button` and private `gr.components.FormComponent` — same base-class-reorganization risk as `modules/ui_components.py`'s `ToolButton`. Note: this is a SEPARATE duplicate ToolButton implementation from `modules/ui_components.py`'s — both need independent re-validation, and the duplication itself is worth flagging to the migration plan as a place two components could diverge after upgrade.** | re-verify | re-verify |

### extensions-builtin/sd_forge_controlnet/scripts/controlnet.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 10 | `import gradio as gr` | none identified | none identified | none identified |
| 43-44 | `gradio_tempfile_path = os.path.join(tempfile.gettempdir(), 'gradio')` — hardcodes gradio's temp dir naming convention | **Moderate risk: relies on gradio continuing to use a subdirectory literally named `gradio` under the system tempdir — if `GRADIO_TEMP_DIR` env var or internal naming changes (see `modules/ui_tempdir.py`'s `is_gradio_temp_path` which reads `GRADIO_TEMP_DIR` env var instead), this hardcoded path could silently point to a stale/wrong directory** | re-verify | re-verify |
| 82 | `gr.Group(elem_id=elem_id_tabname)` | none identified | none identified | none identified |
| 83-84 | `gr.Accordion("ControlNet Integrated", open=False, elem_id=...)` | none identified | none identified | none identified |
| 90 | `gr.Row(elem_id=, elem_classes=...)` | none identified | none identified | none identified |
| 620,623,625,627,629,631,633,635,637,640,642,644,646 | `shared.OptionInfo(..., gr.Slider, {...})`, `shared.OptionInfo(..., gr.Checkbox, {"interactive": True})` (many) — class references, dynamically instantiated | none identified directly (downstream risk only) | none identified | none identified |

### extensions-builtin/sd_forge_sag/scripts/forge_sag.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 1 | `import gradio as gr` | none identified | none identified | none identified |
| 20 | `gr.Accordion(open=False, label=...)` | none identified | none identified | none identified |
| 21 | `gr.Checkbox(label='Enabled', value=False)` | none identified | none identified | none identified |
| 22-23 | `gr.Slider(label=, minimum=, maximum=, step=, value=...)` ×2 | none identified | none identified | none identified |

### extensions-builtin/sd_forge_dynamic_thresholding/scripts/forge_dynamic_thresholding.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 1 | `import gradio as gr` | none identified | none identified | none identified |
| 35 | `gr.Accordion(open=False, label=...)` | none identified | none identified | none identified |
| 36,68 | `gr.Row()`, `gr.Group(visible=True) as advanced_controls` | none identified | none identified | none identified |
| 37,42 | `gr.Checkbox(...)` ×2 | none identified | none identified | none identified |
| 49 | `gr.Group(visible=True) as simple_controls` | none identified | none identified | none identified |
| 50,58,75,83(no,Radio),89,97,123 | `gr.Slider(...)` (many) | none identified | none identified | none identified |
| 69,89(Radio),105,111,117 | `gr.Radio(...)` (many) | none identified | none identified | none identified |
| 133 | `gr.Markdown(value="", visible=True)` | none identified | none identified | **5->6: padding default — visual-only** |
| 140,145,154 | `mimic_mode.change(...)`, `cfg_mode.change(...)`, `simple_mode.change(...)` | none identified | none identified | none identified |
| 152 | `return gr.Group.update(visible=True), gr.Group.update(visible=not simple)` | **P1: both break in 4.0; replace with `gr.update(visible=...)`** | n/a once fixed | n/a |

### extensions-builtin/canvas-zoom-and-pan/scripts/hotkey_config.py
| Line | API Used | 3->4 | 4->5 | 5->6 |
|---|---|---|---|---|
| 1 | `import gradio as gr` | none identified | none identified | none identified |
| 5-6 | `shared.OptionInfo(..., gr.Radio, {"choices": [...]})` ×2 — class reference, dynamically instantiated | none identified directly (downstream risk only) | none identified | none identified |
| 16 | `shared.OptionInfo(..., gr.CheckboxGroup, {"choices": [...]})` — class reference | none identified directly (downstream risk only) | none identified | none identified |

---

## Summary counts

- Files scanned for `import gradio`/`from gradio`: 57 (18 under `modules/ui*.py`, 39 under `extensions-builtin/*`).
- Files with zero gradio usage despite matching the glob: `modules/ui_extra_networks_checkpoints.py`,
  `modules/ui_extra_networks_hypernets.py`, `modules/ui_extra_networks_textual_inversion.py`.
- Distinct `.update()` classmethod call sites requiring mechanical fix for Gradio 4.0 (P1): **~25**
  across `modules/ui.py`, `modules/ui_checkpoint_merger.py`, `modules/ui_common.py`,
  `modules/ui_extensions.py`, `modules/ui_prompt_styles.py`, `modules/ui_settings.py`,
  `extensions-builtin/Lora/ui_edit_user_metadata.py`,
  `extensions-builtin/reForge-advanced_model_sampling_backported/...`,
  `extensions-builtin/reForge-advanced_model_sampling/...`,
  `extensions-builtin/sd_forge_dynamic_thresholding/...`,
  `extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/preset.py`,
  `extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/controlnet_ui_group.py`.
- `gr.Box` usages requiring replacement with `gr.Group` (P2): 3 sites (`modules/ui_prompt_styles.py`,
  `modules/ui_extra_networks_user_metadata.py`, `extensions-builtin/sd_forge_controlnet/.../preset.py`).
- `gr.Image(..., source=...)` sites requiring `source`->`sources` migration (P3): at least 6 confirmed
  (`modules/ui.py` ×5 in img2img/pnginfo/txt2img, `modules/ui_postprocessing.py`,
  `extensions-builtin/sd_forge_svd/scripts/forge_svd.py`,
  `extensions-builtin/sd_forge_z123/scripts/forge_z123.py`,
  `extensions-builtin/sd_forge_controlnet_example/scripts/sd_forge_controlnet_example.py`), plus
  `extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/controlnet_ui_group.py`'s
  `gr.Image` calls flagged for explicit re-check (kwargs not fully captured in this pass).
- Component-subclassing / private-internal-type sites needing explicit re-validation each major
  version (P7/P8/P9): `modules/ui_components.py` (8 classes), `modules/ui_tempdir.py` (IOComponent
  monkeypatch), `modules/ui_gradio_extensions.py` (routes.templates monkeypatch — **highest risk
  item in the whole audit**), `modules/ui.py` (`gradio.utils` monkeypatch, `IOComponent` reference
  via `gradio_extensons.original_IOComponent_init`), `extensions-builtin/sd_forge_controlnet/lib_controlnet/infotext.py`
  and `.../controlnet_ui_group.py` (`IOComponent` type hints),
  `extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/tool_button.py` and
  `extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/modal.py` (`gr.Interface` subclass).
- `_js=` kwarg usage (P12, needs explicit version-by-version verification that the alias to `js=`
  still works): pervasive — found in `modules/ui.py`, `modules/ui_common.py`, `modules/ui_components.py`,
  `modules/ui_extensions.py`, `modules/ui_extra_networks.py`, `modules/ui_extra_networks_user_metadata.py`,
  `modules/ui_prompt_styles.py`, `modules/ui_toprow.py`. This is the single highest-count silent-breakage
  risk category by file count.
- No `gr.Chatbot`/`gr.ChatInterface` usage anywhere — the Gradio 6.0 tuple-format removal does not apply.
- No `gr.Dataframe` `row_count`/`col_count` usage anywhere — the Gradio 6.0 restructuring does not apply.
- No `launch()` call sites within this audit's scope (`webui.py`'s `demo.launch(...)` is outside
  `modules/ui*.py` and `extensions-builtin/*` and should be audited separately before finalizing
  the migration plan, since `concurrency_count`, `enable_queue`, `show_tips`, and the 6.0
  app-level-param-to-`launch()` moves all center on that call site).
