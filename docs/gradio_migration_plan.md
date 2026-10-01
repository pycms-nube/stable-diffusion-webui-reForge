# Gradio Migration Plan: 3.41.2 -> 6.15.1

Input: `docs/gradio_api_usage_audit.md` (per-file, per-line inventory of all gradio Blocks/Component
API usage across `modules/ui*.py` (18 files) and `extensions-builtin/*` (39 files), 57 files total).

## 1. Decision

**Target version: gradio==6.15.1** (full remediation target), not the 4.14.0 minimum-viable floor.

### Options weighed

| | 4.14.0 (minimum viable) | 6.15.1 (full remediation) |
|---|---|---|
| CVE coverage | Clears the 12 critical CVEs | Clears all 12 critical + remaining high/medium gradio CVEs |
| Pillow unblock | No — still caps pillow at <11.0-ish ceiling from gradio 4.x's pin | Yes — unblocks `pillow>=12.1.1`, which the separate pillow remediation task needs (`pillow>=12.3.0`) |
| Migration surface touched | Phase 1 only (mechanical `.update()`/`gr.Box`/`source=` renames, ~35 sites) | Phases 1-3: mechanical renames + Gradio 5 SSR-rewrite re-validation + Gradio 6 app-level-param moves |
| Risk profile | Low — audit found no SSR/routing-internals usage affected at the 4.x boundary | Higher — this repo's highest-risk items (P6/P7/P8/P9 in the audit: `gr.routes.templates` monkeypatch, `IOComponent` monkeypatches, 8+ component subclasses, `_js=` alias) all hinge on Gradio 5's SSR rewrite, which a 4.14.0-only migration would defer, not avoid |
| Rework cost | A second migration project (4.x -> 6.x) would still be required later to unblock pillow, re-touching the same ~35 mechanical sites plus doing the SSR-era re-validation anyway | One migration pass; all 857 audit rows get addressed once |

### Rationale

1. The pillow CVE remediation task is an explicit sibling/dependency of this migration (`docs/dependabot_triage_2026-09.md`, 13 high-severity pillow CVEs, fix requires `pillow>=12.1.1`). Stopping at 4.14.0 would require a *second* gradio migration later to unblock it — re-touching the same files twice instead of once.
2. The audit found **no blockers** that make a direct 3.41.2 -> 6.15.1 jump infeasible: no `gr.Chatbot`/`gr.ChatInterface` usage (so the 6.0 tuple-format removal doesn't apply), no `gr.Dataframe` `row_count`/`col_count` usage, no removed `launch()` kwargs (`concurrency_count`, `enable_queue`, `show_tips`) in `webui.py`'s single `demo.launch()` call site.
3. The genuinely risky items in this codebase (SSR-era monkeypatches, component subclassing, `_js=` alias survival) are tied to the **Gradio 5.0 SSR rewrite**, which sits *between* 4.14.0 and 6.15.1 either way. Picking 4.14.0 as the final target does not avoid this risk, it only postpones paying for it. Since the risk must be paid regardless, pay it once, in one coordinated PR, with the full audit as a checklist, rather than twice across two separate migration efforts.
4. Effort is still bounded and sequenceable (see §3): the mechanical fixes (Phase 1) are independent of the SSR-risk fixes (Phase 2), which are independent of the Gradio-6-specific app-level-param moves (Phase 3). Each phase can be committed, tested, and reverted independently if an unexpected regression surfaces — this is not an all-or-nothing jump.

### Caveat / escape hatch

If Phase 2 (the SSR-rewrite re-validation, see §3) turns up a monkeypatch that cannot be reproduced against Gradio 5's internals in reasonable time, the fallback is to ship Phase 1 alone (pin `gradio==4.14.0` or the latest 4.x patch), take the 12-critical-CVE win immediately, and re-open a follow-up task for the 5.x/6.x jump once a replacement for the broken monkeypatch is designed. `docs/gradio_migration_plan.md` (this document) should be updated with that decision if it is taken.

## 2. Requirements files affected

The task body mentions "all 4 requirements files"; the audit/repo scan found **5** files pinning gradio and must all be updated together:

- `requirements.txt` (line 12: `gradio==3.41.2`)
- `requirements_versions.txt` (line 18: `gradio==3.41.2`)
- `requirements_versions_legacy.txt` (line 11: `gradio==3.41.2`)
- `requirements_versions_py314.txt` (line 24: `gradio==3.41.2`)
- `requirements_new.txt` (line 92: `gradio==3.41.2`, line 93: `gradio_client==0.5.0` — this explicit `gradio_client` pin must also be bumped or removed to let gradio 6.15.1's own `gradio_client` dependency resolve; 0.5.0 predates gradio 4.x entirely and will conflict)

All five get `gradio==6.15.1` (and a compatible `gradio_client` version, or the explicit pin dropped) in the same change that implements Phase 3 below. Do not bump the pin until Phases 1-3 are implemented and verified (`t_6958cdf8` handles the actual code migration; this plan is its required input).

## 3. Migration order, mapped to audit items

Work proceeds in three phases corresponding to the three major-version boundaries crossed. Each
row names the audit pattern (Pxx), the concrete code change required, and whether it risks **silent
runtime behavior change** (wrong/degraded behavior with no exception) vs. a **hard break** (import
error / exception, fails loudly and is caught by smoke test or CI immediately).

### Phase 1 — 3.41.2 -> 4.14.0 (mechanical, low risk, clears the 12 critical CVEs)

| # | Audit item | Required code change | Files (see audit for full line lists) | Risk |
|---|---|---|---|---|
| 1 | P1 — `gr.<Component>.update(...)` classmethod | Replace every `gr.X.update(...)` call with module-level `gr.update(...)`, or return a fresh `gr.X(...)` instance where the call sets non-value fields | `modules/ui.py`, `modules/ui_checkpoint_merger.py`, `modules/ui_common.py`, `modules/ui_extensions.py`, `modules/ui_prompt_styles.py`, `modules/ui_settings.py`, `extensions-builtin/Lora/ui_edit_user_metadata.py`, `extensions-builtin/reForge-advanced_model_sampling_backported/...`, `extensions-builtin/reForge-advanced_model_sampling/...`, `extensions-builtin/sd_forge_dynamic_thresholding/...`, `extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/preset.py`, `extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/controlnet_ui_group.py` (~25 sites) | Hard break — `AttributeError` at call time, caught immediately by import/smoke test |
| 2 | P2 — `gr.Box` | Replace `gr.Box(...)` with `gr.Group(...)` | `modules/ui_prompt_styles.py`, `modules/ui_extra_networks_user_metadata.py`, `extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/preset.py` (3 sites) | Hard break |
| 3 | P3 — `gr.Image/Audio/Video(source=...)` | Rename `source="upload"` (str) to `sources=["upload"]` (list) | `modules/ui.py` (~5 sites), `modules/ui_postprocessing.py`, `extensions-builtin/sd_forge_svd/scripts/forge_svd.py`, `extensions-builtin/sd_forge_z123/scripts/forge_z123.py`, `extensions-builtin/sd_forge_controlnet_example/scripts/sd_forge_controlnet_example.py`; also re-check `extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/controlnet_ui_group.py` lines 265/287/313 (kwargs not fully captured in audit excerpt — confirm no `source=` before closing this item) | Hard break |
| 4 | P4 — `gr.Image(tool=..., brush_color=..., image_mode=...)` sketch/editor kwargs | Port img2img sketch/inpaint/color-sketch tabs to `gr.ImageEditor` per Gradio 4.0's migration guide; the old kwargs are not 1:1 — behavior (brush defaults, layering) must be manually re-verified against the pre-migration UI, not just "does it import" | `modules/ui.py` (img2img sketch/inpaint tabs), `modules/ui_postprocessing.py` | **Silent runtime risk** — old kwargs may be silently dropped before `TypeError` lands; sketch/mask editing could subtly misbehave (wrong brush size/color, missing layers) without raising |
| 5 | P5 — `**shared.hide_dirs` kwargs passthrough | Audit `shared.hide_dirs`'s actual keys against `gr.Textbox.__init__`/`gr.Checkbox.__init__` signatures at 4.14.0; drop/rename any key not accepted under strict-kwargs enforcement | `modules/ui.py`, `modules/ui_postprocessing.py` | Hard break if an unsupported key is present (`TypeError` at component construction) |
| 6 | P6 — `gr.deprecation.GradioDeprecationWarning`, `gradio.utils.version_check`/`get_local_ip_address` monkeypatches | `deprecation.py` is removed in 4.0 per gradio's own changelog — delete or replace the warning-filter mechanism; re-point the `gradio.utils` monkeypatches at their new location (or drop them if no longer needed, e.g. if newer gradio no longer phones home for version checks) | `modules/ui.py` lines 40, 57-58 | Hard break on the `deprecation` import; **silent risk** on the `gradio.utils` patch if the attribute still exists post-4.0 but at a different semantic (patch "succeeds" but doesn't suppress what it used to) |
| 7 | `webui.py`'s `shared.demo.launch(...)` and `shared.demo.queue(64)` calls (outside this audit's file scope, flagged for separate check) | Confirmed clean: no removed launch kwargs (`concurrency_count`, `enable_queue`, `show_tips`) are used. However, `queue(64)` passes a bare positional int — verify this still maps to `default_concurrency_limit` (renamed from `concurrency_count` in 4.0's queue signature) and not a different positional parameter at 4.14.0; if the signature shifted, pass `queue(default_concurrency_limit=64)` explicitly | `webui.py` lines 136-137, 149-165 | Hard break if positional arg no longer matches; low-but-nonzero silent risk if it silently binds to the wrong now-repositioned parameter |

**Phase 1 exit criteria:** `pip install gradio==4.14.0`, `python -c "import modules.ui"` succeeds, webui launches and all tabs render with no exceptions in the log.

### Phase 2 — 4.14.0 -> 5.x (Gradio 5 SSR rewrite — highest risk cluster in this repo)

| # | Audit item | Required code change | Files | Risk |
|---|---|---|---|---|
| 8 | P9 — `gr.routes.templates.TemplateResponse` monkeypatch | Re-implement the custom-JS/CSS template injection against Gradio 5's SSR-rewritten routing layer. This is the single highest-risk item in the whole audit: if the patch target moved/renamed, the patch assignment may silently no-op (templates render, but without the custom injection) rather than raising. Must be verified by **manually loading the UI and confirming custom JS/CSS actually executes** (e.g. check a JS-driven UI behavior works), not just by absence of exceptions | `modules/ui_gradio_extensions.py` lines 71, 75 | **Silent runtime risk — highest in audit.** No exception; the whole custom JS/CSS layer for the UI could silently stop working |
| 9 | P8 — `gradio.temp_file_sets`/`gradio.temp_dirs` probing, `IOComponent.pil_to_temp_file` monkeypatch | Rewrite against Gradio 5's actual temp-file/file-serving internals (this file's own comments show it already needed per-minor-version adaptation at 3.9 vs 3.15 — expect a rewrite, not a one-line fix, at the 4.0+ `allowed_paths` rework and again at 5.0's SSR changes). Verify temp image previews render and temp files get cleaned up, by manual UI test (generate an image, confirm preview shows, confirm temp dir doesn't grow unbounded) | `modules/ui_tempdir.py` lines 16-21, 24-28, 37, 70 | **Silent runtime risk** — temp files silently not cleaned up, or image previews silently failing to render, with no exception |
| 10 | P7 — Component subclassing re-validation | Re-validate each subclass's constructor/render/`postprocess`/`preprocess` behavior against Gradio 5's internals: `modules/ui_components.py`'s 8 classes (`ToolButton`, `ResizeHandleRow`, `FormRow`/`FormColumn`/`FormGroup`, `FormHTML`, `FormColorPicker`, `DropdownMulti`/`DropdownEditable`, `InputAccordion`), the **duplicate** `ToolButton` in `extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/tool_button.py` (flag to keep in sync or de-duplicate as a follow-up), and `gr.Dropdown.get_expected_parent` monkeypatch (`modules/ui_components.py` line 9) | `modules/ui_components.py`, `extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/tool_button.py` | **Silent runtime risk** — broken subclassing often doesn't raise until the component is actually rendered/used; layout quirks (buttons mis-sized, forms mis-grouped) are the likely symptom, not an exception |
| 11 | P12 — `_js=` kwarg on event listeners | Verify the `_js=` -> `js=` alias still works at the chosen 5.x release; if dropped, mechanically rename every `_js=` to `js=` | `modules/ui.py`, `modules/ui_common.py`, `modules/ui_components.py`, `modules/ui_extensions.py`, `modules/ui_extra_networks.py`, `modules/ui_extra_networks_user_metadata.py`, `modules/ui_prompt_styles.py`, `modules/ui_toprow.py` (pervasive, 8+ files) | **Silent runtime risk — highest count by file.** If the alias is dropped, the JS callback simply never fires; no exception, no log line, just dead UI behavior (buttons that don't trigger their JS side effects) |
| 12 | P13 — `gr.Dropdown(tooltip=...)`, `gr.Button(tooltip=...)` | Verify `tooltip=` is accepted at the target 5.x release (non-standard pre-6.0 per public docs); drop or replace if it now raises under strict kwargs | `modules/ui_prompt_styles.py` line 63, `modules/ui_toprow.py` lines 100-103 | Hard break if rejected; cosmetic-only loss (no tooltip shown) if silently accepted-but-ignored |
| 13 | `gr.Gallery(preview=...)`/`object_fit=...` param list drift | Re-verify `gr.Gallery`'s accepted kwargs at 5.x (flagged "re-verify" in audit, not a confirmed break) | `modules/ui_common.py` line 187, `extensions-builtin/sd_forge_svd/scripts/forge_svd.py` line 107, `extensions-builtin/sd_forge_z123/scripts/forge_z123.py` line 93, `extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/multi_inputs_gallery.py` line 22 | Hard break if a removed kwarg is used; low risk per audit (no confirmed removal found) |

**Phase 2 exit criteria:** `pip install gradio==5.x` (pick the latest 5.x patch as a stepping stone before 6.15.1), full manual UI smoke test covering: custom JS/CSS hooks fire (item 8), image generation + preview + temp cleanup (item 9), every `ToolButton`/form-component renders correctly (item 10), every button/dropdown wired with `_js=`/`js=` actually triggers its JS side effect (item 11) — this phase cannot be verified by import-success alone, since every flagged risk here is silent.

### Phase 3 — 5.x -> 6.15.1 (Gradio 6 app-level-param moves)

| # | Audit item | Required code change | Files | Risk |
|---|---|---|---|---|
| 14 | App-level params moved from `Blocks()` constructor to `launch()` | Gradio 6's migration guide moves app-level parameters (e.g. `title`) off the `Blocks()` constructor and onto `launch()`. Verify `gr.Blocks(theme=..., analytics_enabled=False, title="Stable Diffusion")` (`modules/ui.py` line 1144) still accepts these at 6.15.1; if not, move `title=`/`theme=` to the `shared.demo.launch(...)` call in `webui.py` (lines 149-165) | `modules/ui.py` line 1144, `webui.py` launch() call | Hard break if removed from `Blocks()` without being added to `launch()`; silent risk (title/theme silently ignored) if the constructor accepts-but-drops the kwarg |
| 15 | `gr.Interface` default `api_name` change | Gradio 6.0 changes the default auto-generated API name for `gr.Interface` subclasses from `"predict"` to the function name. `ModalInterface(gr.Interface)` in `extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/modal.py` line 5 needs its effective API name re-checked; if any API consumer (internal or external, e.g. the `/api` docs this repo exposes via `app_kwargs={"docs_url": "/docs"}` in `webui.py`) hardcodes `"predict"`, it will silently call the wrong/non-existent endpoint | `extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/modal.py` | **Silent runtime risk** — no exception on the gradio side; an external API caller using the old endpoint name gets a 404, which is its own failure mode outside this repo's logs |
| 16 | HTML/Markdown padding default change | Cosmetic-only, not a functional break. Visual QA pass across pages using `gr.Markdown`/`gr.HTML` (many files: `modules/ui.py`, `modules/ui_common.py`, `extensions-builtin/sd_forge_clpc/...`, `extensions-builtin/sd_forge_latent_modifier/...`, `extensions-builtin/soft-inpainting/...`, `extensions-builtin/sd_forge_dynamic_thresholding/...`, `extensions-builtin/sd_forge_sure_token_ag/...`) | Visual-only, no behavior change; low priority, fix via CSS touch-up if layout looks broken |
| 17 | `IOComponent` type hints (no functional code, informational) | `gr.components.IOComponent` type hints in `extensions-builtin/sd_forge_controlnet/lib_controlnet/infotext.py` line 58 and `.../controlnet_ui_group.py` lines 34-53/56/64/88 don't execute at runtime, but confirm the referenced type still exists under that import path (or update the hint) to avoid breaking static analysis / IDE tooling | `extensions-builtin/sd_forge_controlnet/lib_controlnet/infotext.py`, `.../controlnet_ui_group.py` | None at runtime; type-checker/IDE-only breakage |
| 18 | `modules/ui_loadsave.py`'s `type(x) == gr.X` exact-type checks | Already flagged in the audit as brittle against internal class-hierarchy changes independent of version — confirm at 6.15.1 that `type(x) == gr.Slider` etc. still matches plain component instances (no proxy/wrapper introduced). If gradio 5/6 wraps components differently, these checks silently stop matching and saved UI defaults silently fail to restore on reload | `modules/ui_loadsave.py` lines 75-96 (various) | **Silent runtime risk** — saved per-user UI defaults (slider positions, checkbox states) could silently stop being restored, with no error, only a UX regression users would report as "my settings don't save" |
| 19 | Finalize: bump gradio pin | Set `gradio==6.15.1` (and resolve/bump the explicit `gradio_client==0.5.0` pin in `requirements_new.txt`) in all 5 requirements files listed in §2 | `requirements.txt`, `requirements_versions.txt`, `requirements_versions_legacy.txt`, `requirements_versions_py314.txt`, `requirements_new.txt` | N/A — mechanical pin bump, last step |

**Phase 3 exit criteria:** `pip install gradio==6.15.1`, `python -c "import modules.ui"` succeeds, full manual UI smoke test repeated (title/theme render, ControlNet modal API still reachable under its new name if anything calls it, saved UI defaults still restore after reload), then hand off to the pillow remediation task (now unblocked for `pillow>=12.3.0`).

## 4. Summary of silent-runtime-risk items (cannot be verified by import/smoke-test alone)

These require **manual UI interaction**, not just "the app starts", to confirm they still work:

1. `modules/ui_gradio_extensions.py` — custom JS/CSS template injection (P9, item 8) — **highest risk in the audit**
2. Pervasive `_js=`/`js=` event wiring across 8+ files (P12, item 11) — highest risk by file count
3. `modules/ui_tempdir.py` — temp file cleanup / image preview rendering (P8, item 9)
4. `modules/ui_components.py` + duplicate `tool_button.py` — component subclass rendering (P7, item 10)
5. `modules/ui.py` img2img sketch/inpaint `gr.ImageEditor` port (P4, item 4)
6. `modules/ui_loadsave.py` — exact-type checks gating saved-UI-default restoration (item 18)
7. `extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/modal.py` — `gr.Interface` default API name for any external caller (item 15)

Each of these must be in the verification checklist for `t_6958cdf8` (the follow-up implementation
task), in addition to `python -c "import modules.ui"` succeeding.
