# BFISO — Backend/Frontend Process Isolation Plan

## 1. Problem

Gradio 3.41.2 hard-pins `numpy<2` and `pillow<11` (see comment block at
[requirements_versions.txt:2-4](requirements_versions.txt#L2)). Because the UI and the
inference backend (`ldm_patched/`, `modules_forge/`) share one virtualenv, that pin
propagates to the backend too — 34 files in `ldm_patched/`+`modules_forge/` and 23 in
`modules/` import numpy directly. As long as Gradio stays pinned, the backend can't move
to a newer numpy (or anything that transitively wants one) without a fork of Gradio or a
venv split.

Goal: split the WebUI into two OS processes with independent virtualenvs — a Gradio
frontend pinned exactly as today, and an inference backend free to run whatever
numpy/torch stack it needs — communicating over a wire protocol instead of in-process
Python calls.

## 2. Baseline: what already crosses the boundary today

The codebase currently runs as a single in-process call chain: Gradio UI →
`modules/txt2img.py`/`img2img.py` → `StableDiffusionProcessing*`
([modules/processing.py:138-232](modules/processing.py#L138)) → `process_images_inner` →
`Processed`. `p` is a mutable object passed by reference through the whole pipeline and
into every extension — that's the crux of what makes this hard.

Useful fact: `modules/api/api.py` already reaches the *identical* pipeline through a JSON
request instead of Gradio component values ([modules/api/api.py:467,537](modules/api/api.py#L467)),
which proves most of the payload is already serializable.

| Data | Current shape | Crosses cleanly as data? |
|---|---|---|
| Request scalars (prompt, steps, cfg, seed, sampler, ...) | plain Python types | Yes |
| `init_images` / `mask` | `PIL.Image` | Yes — already base64 PNG/JPEG/WEBP via `encode_pil_to_base64`/`decode_base64_to_image` ([modules/api/api.py:76-131](modules/api/api.py#L76)) |
| Response (seeds, infotexts, extra params, images) | `Processed`, `.js()` already emits pure JSON + `list[PIL.Image]` ([modules/processing.py:685](modules/processing.py#L685)) | Yes |
| Progress / live preview | HTTP long-poll: `progress`, `eta`, `textinfo`, base64 preview ([modules/progress.py:85-139](modules/progress.py#L85)) | Yes — already a wire protocol |
| Interrupt / skip | booleans on a shared global ([modules/shared_state.py:76-82](modules/shared_state.py#L76)) | Trivial as data, but currently relies on being in-process |
| `p.sd_model`, `p.scripts`, `p.script_args`, `gr.Request` | live object refs baked into `p` at construction ([modules/processing.py:277-283](modules/processing.py#L277), [modules/txt2img.py:48-51](modules/txt2img.py#L48)) | **No** |
| Per-step callback payloads: `CFGDenoiserParams`, `CFGDenoisedParams`, `MaskBlendArgs` ([modules/script_callbacks.py:45-84](modules/script_callbacks.py#L45), [modules/scripts.py:16-26](modules/scripts.py#L16)) | raw `torch.Tensor` fired every denoising step | **No** |
| ControlNet's `params.control_cond` / `params.model` ([extensions-builtin/sd_forge_controlnet/scripts/controlnet.py:389-436](extensions-builtin/sd_forge_controlnet/scripts/controlnet.py#L389)) | `torch.Tensor` + live `ControlModelPatcher` merged into `p.sd_model.forge_objects` | **No** |
| `CheckpointInfo` (for model dropdowns) ([modules/sd_models.py:172-231](modules/sd_models.py#L172)) | filename/hash/title strings | Yes |
| Loaded model graph (`ForgeSD`, `load_checkpoint_guess_config`) ([modules_forge/forge_loader.py:183-322](modules_forge/forge_loader.py#L183)) | `nn.Module` graph | Stays backend-internal already, except two direct UI reads noted below |

Two UI call sites read live model attributes directly and would need to become backend
queries instead: `modules/ui.py:1175` (`cond_stage_key`) and
`modules/ui_extra_networks_checkpoints_user_metadata.py:22` (`sd_checkpoint_info.name_for_extra`).

## 3. Where to cut the boundary

The naive cut — "Gradio UI vs. compute" — doesn't work, because scripts and extensions
read/mutate live tensors and model-patcher objects on *every sampling step*
(`CFGDenoiserParams`, ControlNet's `control_cond`). No wire format changes that; it has
to be solved by not putting scripts on the wire at all.

**Decision:** cut above `process_images_inner`, not above the script system.

- **Frontend process** (numpy/pillow pinned to whatever Gradio needs): `modules/ui.py`,
  Gradio component definitions and event wiring, a thin HTTP client that mirrors what
  `modules/api/api.py` already accepts/returns.
- **Backend process** (free to pick any numpy/torch): `modules/processing.py`,
  `modules/scripts.py`, `modules/script_callbacks.py`, everything in
  `extensions-builtin/` and `extensions/`, `ldm_patched/`, `modules_forge/`,
  `modules/sd_models.py` (loading), served behind a formalized version of
  `modules/api/api.py`.

This keeps `p`, its tensors, and every script callback entirely inside the backend
process — nothing script-related needs to serialize. The wire boundary only ever carries
what row 1-5 of the table above already carries today.

Open wrinkle (see §5): scripts currently define *both* their Gradio UI (`Script.ui()`
returning `gr.Component`s) and their processing logic (`process()`, `postprocess()`, ...)
in one class. If script logic moves fully backend-side, the frontend still needs each
script's UI schema to render controls. `modules/api/api.py`'s `init_script_args`
(`api.py:334`) already reflects script UI into a JSON-describable arg list for the REST
API — Phase 0 needs to confirm that reflection path doesn't itself require importing
torch/`ldm_patched`, or the frontend process can't stay thin.

## 4. Wire protocol: HTTP + JSON

Options considered: raw IPC (Unix domain socket / shared memory), HTTP+JSON, HTTP+Protobuf
(or gRPC).

**Recommendation: HTTP + JSON**, formalizing `modules/api/api.py`'s existing contract as
the internal frontend↔backend boundary, not just the external REST surface.

Reasoning against the alternatives, given the actual payload profile (small JSON scalars,
occasional moderate-size PNGs, low-frequency polling, **no** per-step tensor streaming
across the boundary — see §3):

- **Raw IPC / shared memory** only pays off if we were streaming tensors across the
  boundary. We're deliberately not — scripts and callbacks stay backend-side — so there's
  nothing large or latency-sensitive enough to justify it, and it reinvents routing,
  versioning, and debuggability that HTTP gives for free.
- **HTTP + Protobuf / gRPC** gives smaller frames and real server-streaming (push instead
  of poll for progress), but adds schema/codegen maintenance to a boundary that's
  Python-to-Python, isn't payload-size-constrained, and already works over JSON today.
  Worth revisiting only if we later want true push-based per-step progress instead of
  polling — see the open item below.
- **HTTP + JSON** is already built, proven in this exact codebase
  (`modules/api/api.py`), keeps the boundary curl/devtools-debuggable, and the base64
  image overhead (~33%) is negligible at these sizes and frequencies.

Concretely:
- Images: base64, format per `shared.opts.samples_format`, same encode/decode helpers
  already in `modules/api/api.py`.
- Progress: reuse the `/internal/progress` long-poll shape from `modules/progress.py`.
- Interrupt/skip: real `POST` endpoints (already close to this — `api.py` has
  `/sdapi/v1/interrupt` etc.) instead of shared-global mutation.
- Response: `Processed.js()` is already the JSON shape; images travel alongside as
  base64.

## 5. Open risks — resolved in Phase 0

All three were investigated; see [PHASE0.md](PHASE0.md) for full evidence and citations.

1. **Stale-memory discrepancy on progress streaming — resolved.** No SSE implementation
   exists on this branch; progress is confirmed HTTP long-poll only
   (`modules/progress.py`). A real SSE/push implementation does exist, but unmerged, on
   branch `feat/user-session-jobqueue` (`modules/job_queue/api.py`), using thread-safe
   `queue.Queue`, not `asyncio.Queue`/`call_soon_threadsafe` as the stale memory claimed.
   Phase 4 must decide whether to port that branch's work or build push-based progress
   fresh.
2. **Script UI/logic coupling — resolved, and it's a real blocker.** `load_scripts()`
   (`modules/scripts.py:499`) unconditionally imports every script module, and 30 of 44
   `extensions-builtin/*/scripts/*.py` files import `torch`/`ldm_patched` at module
   scope. `init_default_script_args` (`modules/api/api.py:314`) reflects UI by
   constructing real Gradio components, not reading metadata. A thin frontend process
   cannot render script controls today without torch installed — this requires an actual
   code change (deferred imports or a UI-only metadata layer) before Phase 3, not just a
   process split.
3. **Extension inventory beyond ControlNet — resolved.** ~20 extensions clone/patch
   `p.sd_model.forge_objects.unet` (CFG/guidance rewrites, tiled diffusion, SAG, latent
   modifiers, hypertile, freeu, kohya-hrfix, dynamic thresholding, the SURE-AG family,
   Lora); 5 more (`sd_forge_controlnet`, `sd_forge_controlllite`, `sd_forge_ipadapter`,
   `sd_forge_photomaker`, `sd_forge_fooocus_inpaint`) carry `ControlModelPatcher`-style
   tensor conditioning; `sd_forge_svd`/`sd_forge_z123` run standalone pipelines
   independent of `p` but still need backend placement. A dozen extensions (upscalers,
   JS-only UI, `sd_forge_clpc`, `sd_webui_random_resolutions`, third-party `extensions/`
   add-ons) are confirmed low-risk — scalar/image-only, no reach into `p`/tensors. This
   confirms §3's "all of extensions-builtin/, extensions/ run backend-side" boundary is
   necessary, though the low-risk set only reduces testing risk, not the import-coupling
   problem from #2.

## 6. Phased migration plan

- **Phase 0 — Investigate & decide.** Resolve the three open risks in §5. Confirm the
  exact set of modules that must move backend-side (start from §3's list, extend from
  the extension inventory). No code changes.
- **Phase 1 — Prove the venv split. Done, see [PHASE1.md](PHASE1.md).** Two
  virtualenvs, still one process for now: backend code imports from its own venv via a
  subprocess-launched worker, just to prove numpy versions can diverge without touching
  the wire protocol yet. Confirmed: `venv-gr3` stays on numpy 1.26.4 while a
  subprocess-spawned worker in a separate venv independently runs numpy 2.5.1 and a
  newer Python (3.12 vs 3.13), round-tripping a computation via JSON over stdio. The
  subprocess+stdio mechanism itself is a Phase-1-only stand-in — Phase 2 uses HTTP+JSON
  per §4, not this pattern.
- **Phase 2 — Extract the backend service. Partially done, see [PHASE2.md](PHASE2.md).**
  Move `processing.py`, `scripts.py`, `script_callbacks.py`, `extensions-builtin/`,
  `extensions/`, `ldm_patched/`, `modules_forge/`, and model loading behind a standalone
  FastAPI process built by generalizing `modules/api/api.py`. The Gradio frontend becomes
  a client of this service for txt2img/img2img/interrupt/progress, using the exact JSON
  contract from §4. **Scoped down on execution**: the existing `--nowebui --api` mode
  already provides this as a standalone, debugger-attachable process (new launcher:
  [webui-backend.sh](webui-backend.sh)) — proven with a real SDXL generation and a
  passing `pytest test/test_txt2img.py` run against it. The Gradio frontend itself was
  **not** rewired to call it; that remains future work, still gated on Phase 3 below.
- **Phase 3 — Script UI schema. Investigated, see [PHASE3.md](PHASE3.md).** Resolve
  risk #2: either confirm cheap script-module import for UI reflection stays
  frontend-safe, or add a backend "describe controls" endpoint the frontend calls at
  startup to build Gradio components without importing script processing code.
  **Resolved: the first branch is decisively no**, at a scale far beyond PHASE0's
  estimate — a full libcst scan found 577 files / 1306 unguarded backend-only imports,
  including ~69 in core `modules/`+`modules_forge/` itself (not just extensions) and
  hundreds more in vendored third-party ML codebases (`detectron2`, `mmcv`, `oneformer`)
  bundled inside `forge_legacy_preprocessors`, many with torch baked into class
  definitions rather than movable import statements. One real, zero-risk fix was found
  and applied ([modules/sd_models_types.py](modules/sd_models_types.py) — a
  never-instantiated typing stub that didn't need its base class import at runtime,
  verified with no regression via `pytest test/test_txt2img.py`), but the next blocker
  (`modules/safe.py`'s genuine, load-bearing `torch.load` unpickler, reached via a bare
  side-effect import in `modules/paths.py`) is a real design decision, not a mechanical
  fix — and that pattern doesn't scale to hundreds of files. **Recommendation: pursue
  the second branch** (backend-served schema) — `/sdapi/v1/script-info` and
  `ScriptInfo`/`ScriptArg` already exist and already capture most of what's needed
  (label/value/min/max/step/choices); what's missing is an explicit widget-type field
  and a frontend-side JSON-to-`gr.Component` constructor, which is bounded, buildable
  work, unlike auditing hundreds of vendored files.
- **Phase 4 — Script UI schema builder. Done, see [PHASE4.md](PHASE4.md).** Extended
  `ScriptArg` ([modules/api/models.py](modules/api/models.py)) with `component`/
  `multiselect`/`lines`, populated backend-side from the live control's own class/attrs
  ([modules/scripts.py:678](modules/scripts.py#L678)), and built
  [modules/ui_script_schema.py](modules/ui_script_schema.py) — a torch-free
  JSON-schema-to-`gr.Component` constructor. Verified against the real
  `/sdapi/v1/script-info` payload from a live backend: 415 real args across 55 scripts,
  0 mismatches on every mapped component type (406/415, 97.8%), the remaining 9
  gracefully fall back (State/Markdown/HTML — not real user-facing arguments). No
  regression in `pytest test/test_txt2img.py test/test_img2img.py`. Not yet wired into
  an actual frontend entry point, and layout/conditional-visibility reconstruction is
  explicitly out of scope — see PHASE4.md's "Next steps."
- **Phase 5 — Live preview & progress polish. Done, see [PHASE5.md](PHASE5.md).**
  Decide whether to keep long-polling or invest in real server push (SSE/WebSocket) for
  progress and live preview, now that it's crossing an actual process boundary instead
  of being an in-process nicety. Resolves risk #1 either way — by building the SSE path
  for real, or by explicitly confirming long-polling is the intended design.
  **Resolved: built a minimal purpose-built `GET /internal/progress-stream` SSE
  endpoint** ([modules/progress.py](modules/progress.py)), deliberately not porting the
  `feat/user-session-jobqueue` job-queue subsystem (DB/sessions/task-runner — SSE was
  just its delivery mechanism, not separable). Server-side polls the existing progress
  state at the same interval the current long-poll JS client already uses, so it needed
  no sampler/`shared_state.py` changes. Along the way found and fixed a real pre-existing
  bug: `webui.py`'s `api_only_worker()` (the `--nowebui --api` backend-only path since
  Phase 2) never called `progress.setup_progress_api(app)` at all, so `/internal/progress`
  was silently missing from backend-only mode even before this phase. Verified against a
  real generation: 10 SSE events, progress monotonically 0.0→0.967, clean `completed=true`
  close; existing long-poll endpoint and `pytest test/test_txt2img.py test/test_img2img.py`
  unaffected. Not wired into the shipped Gradio UI's JS — proven server-side only.
- **Phase 6 — Wire SSE into the frontend JS. Done, see [PHASE6.md](PHASE6.md).**
  Replaced `javascript/progressbar.js`'s two independent XHR-polling loops
  (`funProgress` + `funLivePreview`) with one native `EventSource` connection to
  Phase 5's `/internal/progress-stream`, preserving `requestProgress()`'s exact public
  contract so `ui.js`/`extensions.js`/`textualInversion.js` needed no changes. Checked
  the WASM angle explicitly (per prior claude-mem finding: Gradio-Lite/Pyodide is
  infeasible for this backend — needs GPU/torch) and concluded it doesn't apply to this
  file either — the live-preview path is browser-native image decode, not CPU-bound JS;
  the real "written before newer APIs existed" gap was the missing `EventSource`, not
  WASM. Verified in a real browser against a real generation: network tab showed exactly
  one `/internal/progress-stream` request for the whole run (zero `POST
  /internal/progress` polls), tab title updated live from the stream, no new console
  errors, image rendered normally on completion.
- **Phase 7 — Launch & docs. Done, see [PHASE7.md](PHASE7.md).** Update
  `webui.sh`/`launch.py` to spawn both processes (or provide a single-process fallback
  for simple installs), update the "Running the Application" section of `CLAUDE.md`, and
  add whatever minimal integration test proves the two-process path produces the same
  output as the current in-process path for a representative txt2img request.
  **Reconciled scope first**: no process spawns both together, because the Gradio
  frontend doesn't call the backend over HTTP yet (that's still open — see PHASE7.md's
  "genuinely remaining work"). Added a "Backend-only mode" section to CLAUDE.md instead.
  Reinterpreted the equivalence requirement as "does the backend-only extraction change
  output" and verified it directly: identical deterministic request against
  `webui.sh`/`--api` and `webui-backend.sh` produced byte-identical images (same SHA-256)
  and identical `info` dicts (seed, subseed, sampler, model hash, version — everything)
  on a clean re-run. Also closes out the phase list with an honest accounting of what
  Phases 0–7 actually proved versus what real cutover work remains.
- **Phase 8 — The actual cutover. Done, see [PHASE8.md](PHASE8.md).** A genuinely
  torch-free process (new `modules_frontend/txt2img_ui.py` + `webui_frontend.py` +
  `webui-frontend.sh` — not `modules/ui.py`/`webui.py`, both confirmed torch-coupled)
  drove a real txt2img generation against a separate `webui-backend.sh` process purely
  over HTTP. Verified in a from-scratch venv with `gradio==3.41.2 numpy==1.26.4
  pillow==10.4.0 requests` and **no torch package at all** (confirmed via `pip list`).
  Found and fixed 4 real bugs surfaced only by actually running this: an argparse
  collision (`IGNORE_CMD_ARGS_ERRORS`, an existing escape hatch), a Gradio/Starlette
  `TemplateResponse` incompatibility (standalone shim, avoiding
  `modules.ui_gradio_extensions`'s torch-heavy import chain), a pre-existing
  `SamplerItem.options` type bug (`dict[str,str]` → `dict[str,Any]`, real booleans in
  real sampler configs), and — the important one — PHASE3.md's `modules/safe.py`
  blocker manifesting for real via `InputAccordion`'s lazy import, fixed by reverting
  that one component to a plain Checkbox. Script args aren't wired into the generation
  request yet, and there's no live progress — both named explicitly as follow-up, not
  hidden. Committed as its own isolated commit, separate from the Phase 0–7 baseline, so
  it can be reverted on its own if something's wrong.
- **Phase 9 — Script args + live progress. Done, see [PHASE9.md](PHASE9.md).** Closed
  Phase 8's two named gaps in the same `modules_frontend/txt2img_ui.py`. Script controls'
  values are now sliced back into `alwayson_scripts` and sent with the request —
  verified genuinely reaching the backend via a real script's own `info` output
  (`mahiro_cfg_enabled: True`), not just a payload-shape assumption. Live progress uses
  the request's existing `force_task_id` field plus Phase 5's SSE stream, via a
  generator-based click handler backed by a background thread for the blocking POST.
  Found and fixed two real Gradio 3.x quirks along the way: a `lambda` wrapping a
  generator function isn't itself detected as one (`functools.partial` fixes it), and
  generator-yielding event handlers need `demo.queue()` explicitly enabled. Progress and
  script-arg wiring verified live in a real browser; live-preview verification switched
  mid-phase from automated browser testing to the user manually confirming it in their
  own browser, after a host OS crash during heavy automated generation testing (cause
  undiagnosed) — the user confirmed both progress ticking through real values and the
  preview image updating live. No regression in
  `pytest test/test_txt2img.py test/test_img2img.py` (identical baseline to every prior
  phase). Torch-freedom verified by import inspection rather than a fresh from-scratch
  venv re-run this time (only stdlib additions: `functools`/`threading`/`time`/`uuid`).
- **Phase 10 — Interrupt/Skip + missing core params. Done, see [PHASE10.md](PHASE10.md).**
  Added Skip/Interrupt buttons (thin wrappers over the existing `/sdapi/v1/interrupt` /
  `/sdapi/v1/skip` endpoints) and batch count/size + restore faces/tiling, all previously
  missing from the request payload. Found and fixed a real bug the user caught by
  clicking the buttons mid-generation: Gradio 3.41.2's `demo.queue()` defaults to
  `concurrency_count=1`, which serializes all queued events through one slot — the
  long-running Generate generator held that slot for the whole run, so Skip/Interrupt
  clicks silently queued up behind it and only fired once generation had already
  finished on its own. Fixed with `concurrency_count=3`. Verification methodology
  changed deliberately after the Phase 9 OS crash: single one-shot `curl` checks for
  anything that's a plain request/response (confirmed `restore_faces`/`tiling` reach the
  pipeline via the real returned infotext), and live/dynamic behavior (does
  Skip/Interrupt actually interrupt) handed to the user for manual browser confirmation
  instead of automated rapid-polling loops, per the user's explicit request to avoid the
  testing pattern that preceded the crash.
- **Phase 11 — Hires. fix. Done, see [PHASE11.md](PHASE11.md).** Added `enable_hr` plus
  its five core sub-params (`hr_scale`, `hr_upscaler`, `hr_second_pass_steps`,
  `denoising_strength`, `hr_resize_x`/`hr_resize_y`) in their own Accordion. The
  upscaler dropdown is built from the union of `/sdapi/v1/latent-upscale-modes` and
  `/sdapi/v1/upscalers`, matching exactly what backend-side validation checks against.
  Verified reaching the real pipeline via a single `curl` request's returned infotext.
  Investigated a user-reported discrepancy (Hires steps = 0 appearing to skip the second
  pass) with a controlled pixel-level comparison — two identical requests differing only
  in that field decoded to byte-for-byte identical images, proving the `0 → use base
  steps` fallback works correctly; the apparent skip was `modules/processing.py` only
  writing `"Hires steps"` into infotext when the value is truthy, which the shipped
  UI's own identically-defaulted-to-0 slider shares, so no fix was applied. No
  regression risk since no backend code changed.
- **Phase 12 — Basic img2img tab. Done, see [PHASE12.md](PHASE12.md).** Added a second
  pipeline alongside txt2img, structured as a `gr.Tabs()` layout. Extracted shared
  helpers (sampler/upscaler/script-info fetching, progress SSE streaming, interrupt/
  skip, image codec) into new `modules_frontend/common.py` so img2img didn't duplicate
  txt2img's machinery; `txt2img_ui.py`'s `create_ui()` became `create_txt2img_tab()`
  (no longer owns its own `gr.Blocks()`/`.queue()`), and new `modules_frontend/app.py`
  assembles both tabs and owns the single queue. `build_alwayson_script_controls()`
  generalized with an `is_img2img` flag instead of a hardcoded txt2img-only filter.
  Verified the exact payload shape `run_img2img()` builds against the real
  `/sdapi/v1/img2img` endpoint via a single one-shot `curl` check (tiny synthetic init
  image, real infotext returned), confirmed `create_ui()` builds cleanly against the
  live backend, re-confirmed torch-freedom by import inspection across all
  `modules_frontend/*.py` files, and had the user manually confirm both tabs work live
  (txt2img unaffected by the refactor, img2img's upload → Generate flow works
  end-to-end). Scoped down deliberately: basic img2img only, no inpainting (mask/mask
  blur/inpaint_full_res), no Sketch/Inpaint/Batch sub-tabs.
- **Phase 13 — Mask-upload inpainting. Done, see [PHASE13.md](PHASE13.md).** Closed
  Phase 12's named biggest remaining gap: a separate mask-image upload plus
  `mask_blur`/`inpainting_mask_invert`/`inpainting_fill`/`inpaint_full_res`/
  `inpaint_full_res_padding`, matching the shipped UI's own "Inpaint upload" sub-tab
  semantics exactly (same Radio choice/index conventions as `modules/ui.py`). The
  `mask` payload key is only added when a mask was actually uploaded, mirroring
  `img2imgapi`'s own `if mask:` check, so plain (non-masked) img2img requests are
  unaffected. Verified with a single one-shot `curl` request (synthetic init image +
  mask) whose returned infotext showed `Mask blur`/`Inpaint area`/`Masked area
  padding`, confirming inpainting genuinely activated rather than the mask being
  silently dropped. User manually confirmed live: masked region changed, rest of the
  image stayed close to the original. **Deliberately still no sketch-on-canvas mask
  drawing** — this is upload-a-separate-mask-image only, not the shipped UI's
  paint-directly-on-the-image workflow.
- **Phase 14 — Sketch-on-canvas mask drawing. Done, see [PHASE14.md](PHASE14.md).**
  Closed Phase 13's remaining gap: a "Mask input" toggle switches between the
  separate-upload mode and a single `gr.Image(tool="sketch")` where the mask is drawn
  directly over the image, matching the shipped UI's "Inpaint" sub-tab. Confirmed by
  reading Gradio's own `Image.preprocess()` that `tool="sketch"` + `type="pil"` already
  splits the canvas into `{"image": ..., "mask": ...}` PIL images before reaching this
  code — no manual canvas decoding needed. Since drawing on a canvas isn't automatable
  through this session's tools, verification split cleanly: the dict-unpacking and
  payload-building logic was tested by calling `run_img2img()` directly with a
  synthetic dict matching Gradio's documented format (real backend call, real infotext
  confirming inpainting activated), while the actual "does drawing work" question was
  handed to the user, who confirmed it live.
- **Phase 15 — Advanced Hires. fix overrides. Done, see [PHASE15.md](PHASE15.md).**
  Added `hr_checkpoint_name`/`hr_sampler_name`/`hr_scheduler`/`hr_prompt`/
  `hr_negative_prompt`/`hr_cfg`, tucked into a nested sub-accordion. Found (by reading
  the actual REST API code path this frontend uses, before writing the payload logic,
  not by hitting the bug live) that the shipped Gradio UI's "Use same X" sentinel
  strings only get translated to `None` inside the Gradio-only
  `modules/txt2img.py::txt2img_create_processing()` — `modules/api/api.py`'s
  `text2imgapi` REST path this frontend calls does no such translation, so sending the
  literal sentinel text would have looked up a sampler/scheduler literally named "Use
  same sampler"/"Use same scheduler" and failed. `run_txt2img()` replicates that
  translation client-side. Verified both the sentinel-default path (completes
  successfully, infotext shows the base pass's own CFG as the effective hires CFG) and
  an explicit-override path (`hr_sampler_name`/`hr_scheduler`/`hr_prompt`/`hr_cfg` all
  showed up correctly in the returned infotext) by calling `run_txt2img()` directly
  against the real backend. User manually confirmed the dropdowns populate with real
  data and a full sentinel-default generation completes successfully.
- **Phase 16 — Two-click confirm for Skip/Interrupt. Done, see [PHASE16.md](PHASE16.md).**
  Closed the "no cancel confirmation UX" gap. New
  `modules_frontend/common.py::build_confirm_action_button()` builds and wires a
  button implementing a two-click confirm pattern (arm on first click, relabel to
  "Confirm X?", act on second) using only `gr.State` + the button's own label, since
  Gradio 3.x has no built-in modal dialog. Both tabs' Skip/Interrupt now go through
  this helper. Part of a batch of non-functional/cosmetic gaps (Phases 16-19) the user
  explicitly asked to implement together and review once at the end, rather than
  stopping for live confirmation after each one.
- **Phase 17 — Paint (color-sketch) image source. Done, see [PHASE17.md](PHASE17.md).**
  Closed the "no Sketch sub-tab" gap: an "Image source" toggle (Upload/Paint) lets
  img2img's init image come from a `gr.Image(tool="color-sketch")` canvas instead of
  an upload. Confirmed by reading `gradio/components/image.py` that `color-sketch`
  (unlike `sketch`) is NOT split into an `{"image", "mask"}` dict, so it behaves as a
  plain painted image independent of Phase 14's mask-drawing toggle. Both toggles now
  share one visibility-computation function since they're orthogonal but both affect
  `init_image`'s visibility. Verified by calling `run_img2img()` directly with
  `init_image=None` and a synthetic painted image — succeeded only because the
  override genuinely supplied the image (the function's own guard would otherwise
  have rejected the missing init image).
- **Phase 18 — Batch (multiple images). Done, see [PHASE18.md](PHASE18.md).** Closed
  the "no Batch sub-tab" gap: upload several images and run the tab's current
  prompt/steps/sampler/etc. settings against each via real `/sdapi/v1/img2img` calls,
  looped client-side. Deliberately not the shipped UI's local `input_dir`/`output_dir`
  approach — that only works when frontend and backend share a filesystem, which
  BFISO's whole premise is to not require. Verified with synthetic file objects
  shaped like Gradio's own `gr.File` output (`.name` attribute) against the real
  backend: real progress ticks `0/2 → 1/2 → 2/2 → done`, 2 real result images
  returned.
- **Phase 19 — Layout polish. Done, see [PHASE19.md](PHASE19.md).** Closed the "flat
  layout" gap named since Phase 8 with lightweight `gr.Markdown()` section headers
  (Sampling/Size/Batch/Options/Sampler & Seed) on both tabs — purely cosmetic, no
  component moved, renamed, or rewired; every `.click()` input list is byte-for-byte
  unchanged. Deliberately not a full rebuild into the shipped UI's nested-accordion/
  conditional-visibility structure, given this was explicitly the lowest-priority item
  on the list. No regression in `pytest test/test_txt2img.py test/test_img2img.py`
  (unsurprising, since no backend/request-payload logic changed). Closes out the
  Phase 16-19 batch of non-functional/cosmetic gaps the user asked to implement
  together and review once at the end, rather than after each individual phase.
- **Phase 20 — Combined launch script. Done, see [PHASE20.md](PHASE20.md).** First
  concrete step toward the user's longer-term goal of eventually replacing the
  current single-process implementation (so future Gradio/backend upgrades are
  easier) — specifically the piece they chose when asked to prioritize: launch
  integration, not new feature coverage. New `webui-split.sh` backgrounds
  `webui-backend.sh`, polls `/sdapi/v1/samplers` until healthy, then backgrounds
  `webui-frontend.sh` against it, with a trap tearing both down on exit.
  `FRONTEND_BACKEND_URL` tracks a custom `BACKEND_PORT` automatically. Verified with
  a real cold start (real SDXL checkpoint load, health-check loop succeeding, both
  processes reachable). Found and honestly reported a real signal-handling nuance
  while testing shutdown: `SIGTERM` cleanup is directly proven (both children die,
  "Shutting down..." logged); `SIGINT` did NOT fire cleanup in this specific test,
  traced to bash's rule that asynchronous (`&`-launched) commands from a
  non-interactive shell force SIGINT to be ignored — an artifact of how this
  sandboxed environment (no TTY) had to launch the script for testing, not of a real
  user running it in the foreground of their own terminal, where that rule doesn't
  apply. Documented as an open item to manually confirm rather than silently assumed
  or overclaimed. Explicitly NOT a step toward feature parity — Settings, Extras,
  PNG Info, Train, and extension UI remain exclusively on the original `webui.sh`
  path; `webui.sh` itself untouched per its own no-modify convention.

## 7. Non-goals

- Not attempting to make the backend importable from a non-Python frontend — both ends
  stay Python, so there's no cross-language requirement pushing toward Protobuf/gRPC.
- Not attempting to eliminate numpy from either side — the goal is independent version
  freedom, not removal.
