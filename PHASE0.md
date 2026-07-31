# PHASE0 — Investigation Findings

Resolves the three open risks listed in [BFISO.md](BFISO.md) §5, ahead of committing to
the phased migration plan in BFISO.md §6. Pure investigation, no code changes made.

## Risk 1 — SSE vs. long-poll progress: RESOLVED

**Current branch (`backend-frontend-isolation`) and the whole working tree: no SSE
implementation exists.** Whole-repo grep for `text/event-stream`, `EventSourceResponse`,
`StreamingResponse`, `EventSource(`, `asyncio.Queue`, `call_soon_threadsafe` returned zero
hits anywhere in modules/, modules_forge/, ldm_patched/, extensions-builtin/, extensions/.

`modules/progress.py` is confirmed pure HTTP long-poll:
- [modules/progress.py:74-76](modules/progress.py#L74) registers `POST /internal/progress`
  → `progressapi` and `GET /internal/pending-tasks`.
- [modules/progress.py:85-139](modules/progress.py#L85) `progressapi()` is one-shot: given
  `id_task`/`id_live_preview`, computes `active/queued/completed`, `progress`, `eta`, and
  an optional base64 `live_preview`, then returns immediately.
- `javascript/progressbar.js:75` `requestProgress()` calls `request("./internal/progress")`
  and re-schedules itself via `setTimeout(...)` (lines 171, 202) — recursive polling, not a
  persistent connection.

**But SSE work does exist — unmerged, on branch `feat/user-session-jobqueue`** (not an
ancestor of `main` or `backend-frontend-isolation`; last commit 2026-06-07, matching the
date of the claude-mem session memory that prompted this check):
- `modules/job_queue/api.py:154` — `StreamingResponse(..., media_type="text/event-stream")`
- `modules/job_queue/api.py:106` — `GET /forge/v1/queue/events`
- `modules/job_queue/api.py:44` — per-user subscriber registry `_subscribers: dict[str, list]`
- `modules/job_queue/api.py:65-73` — `push_event(user_id, event)` fans out via `q.put_nowait`
- `extensions-builtin/forge_job_queue/javascript/job_queue.js:253` — client `new
  EventSource("/forge/v1/queue/events")`

**Correction to the memory claim:** that branch uses thread-safe `queue.Queue`
(`stdlib_queue.Queue`), not `asyncio.Queue`, and `call_soon_threadsafe` doesn't appear on it
either. The SSE generator does a non-blocking `q.get_nowait()` poll with a 20s heartbeat
sleep — push-notified via a thread-safe queue, but read via polling inside the async
generator, not via event-loop injection.

**Verdict:** the memory wasn't fabricated, just stale/misattributed — real SSE work exists
but on a different, unmerged branch, with a different implementation than described. Phase 4
of BFISO.md should treat the current branch's progress delivery as long-poll-only, and
separately decide whether to port/rebase `feat/user-session-jobqueue`'s SSE work rather than
building push-based progress from scratch.

## Risk 2 — Script UI/logic import coupling: RESOLVED — real blocker, not just a wrinkle

**Verdict: no.** A thin frontend process cannot import `modules/scripts.py` + script modules
today to render script UI without torch/`ldm_patched` installed.

- [modules/scripts.py:499](modules/scripts.py#L499) `load_scripts()` unconditionally imports
  **every** discovered script file via `script_loading.load_module(scriptfile.path)`
  ([modules/scripts.py:533](modules/scripts.py#L533)) — a real Python import executing all
  top-level code, not metadata parsing. `Script.ui()` is invoked later, per-tab, in
  `ScriptRunner.create_script_ui_inner` ([modules/scripts.py:665](modules/scripts.py#L665)),
  but by then the module is already fully imported.
- [modules/api/api.py:314-332](modules/api/api.py#L314) `init_default_script_args` builds a
  real `gr.Blocks()` and calls `script.ui(script.is_img2img)` to read `.value` off live
  Gradio components — there is no lightweight schema file being read instead.
- A repo-wide scan of `extensions-builtin/*/scripts/*.py` (44 files) found **30 of 44 import
  `torch`, `ldm_patched`, or `modules_forge` at module scope**, not deferred into `process()`.
  Examples: `extensions-builtin/sd_forge_freeu/scripts/forge_freeu.py:10,13` imports
  `ldm_patched.contrib.nodes_freelunch.FreeU_V2` and instantiates it (`opFreeU_V2 =
  FreeU_V2()`) at import time; `extensions-builtin/sd_forge_controlnet/scripts/controlnet.py:5`
  imports `torch` directly. Counter-examples that stay light:
  `extensions-builtin/Lora/scripts/lora_script.py`,
  `modules/processing_scripts/seed.py`/`sampler.py`.

**Implication:** this cannot be solved by process separation alone — it requires
restructuring script loading itself (lazy/deferred heavy imports inside `process()` rather
than at module scope, or introducing a UI-only metadata layer independent of the script
module). This becomes a required sub-task, not an assumption, before Phase 3 of BFISO.md.

## Risk 3 — Extension live-tensor/model-ref inventory: RESOLVED

`extensions-builtin/` has 43 dirs, `extensions/` has 3 real third-party dirs (plus a
placeholder file). Legacy webui hooks (`on_cfg_denoiser`, `on_cfg_denoised`,
`on_cfg_after_cfg`, `CFGDenoiserParams`, `CFGDenoisedParams`) have **zero hits** anywhere in
extensions — this fork doesn't use them; the equivalent hook point is
`process_before_every_sampling` mutating `p.sd_model.forge_objects`. `on_mask_blend`/
`MaskBlendArgs` has exactly one consumer: `extensions-builtin/soft-inpainting/scripts/
soft_inpainting.py:707`.

**Dominant pattern (~20 extensions):** clone/patch `p.sd_model.forge_objects.unet` (and
occasionally `.vae`/`.clip`) inside `process_before_every_sampling`-style hooks — CFG/guidance
rewrites (`mahiro_reforge`, `reForge-APGIsYourCFG`, `reForge-RescaleCFG`,
`reForge-advanced_model_sampling(_backported)`), tiled diffusion (`sd_forge_multidiffusion`),
attention/latent guidance (`sd_forge_sag`, `sd_forge_latent_modifier`,
`sd_forge_dynamic_thresholding`, `sd_forge_kohya_hrfix`, `sd_forge_freeu`,
`sd_forge_hypertile`), the SURE-AG family, and `Lora`. All confirmed backend-only.

**ControlModelPatcher family (tensor conditioning + live model):** `sd_forge_controlnet`
(confirmed previously), plus `sd_forge_controlllite`, `sd_forge_ipadapter`,
`sd_forge_photomaker`, `sd_forge_fooocus_inpaint`, and the `forge_preprocessor_*` family
(`inpaint`, `tile`, `reference`, `revision`) — all backend-only, reaching `.forge_objects.unet`/
`.vae`/`.clip` directly.

**Standalone pipelines that never touch `p` but still must be backend-side:** `sd_forge_svd`,
`sd_forge_z123` — build their own live UNet/VAE/latents via `load_checkpoint_guess_config`
from a dedicated UI tab, independent of the main txt2img/img2img `p` object.

**Confirmed low-risk (scalar/JSON-safe or image-buffer-only, no reach into `p`/tensors):**
`sd_forge_clpc` (forwards choices via `p.extra_generation_params`, actual math lives in core
`ldm_patched/k_diffusion/clpc_sampler.py`), `sd_webui_random_resolutions` (reads
`p.sd_model.is_sdxl` bool only), `sd_forge_neveroom` (queries device memory only),
`LDSR`/`ScuNET`/`SwinIR` (operate on PIL/numpy images passed in, no `p`/`forge_objects` ref —
though they load their own torch models internally, so "runs a model" ≠ "reaches into `p`"),
`forge_preprocessor_recolor`, `forge_legacy_preprocessors`, all JS-only UI extensions
(`canvas-zoom-and-pan`, `extra-options-section`, `mobile`, `prompt-bracket-checker`), and all
three third-party `extensions/` add-ons. `forge_job_queue` has no `.py` source in the tree
(only a stale `.pyc`) and isn't assessable from source.

No extension anywhere accesses `p.script_args` directly.

**Implication:** the §3 boundary in BFISO.md ("all of extensions-builtin/, extensions/ run
backend-side") is confirmed correct and necessary — roughly two dozen extensions would break
if they couldn't reach `p.sd_model.forge_objects`. The dozen low-risk extensions identified
above don't need special handling, but also don't provide a shortcut: they still need their
`Script` modules backend-loadable per Risk 2, so their "low risk" status only reduces
*testing* risk, not the import-coupling problem.

## Net effect on the Phase plan (BFISO.md §6)

- **Phase 0** (this document): done.
- **Phase 2** ("Extract the backend service") should explicitly include: decide whether to
  port `feat/user-session-jobqueue`'s SSE/job-queue work into the extraction, or keep it
  parked and revisit in Phase 4.
- **New required work before Phase 3** ("Script UI schema"): Risk 2 means Phase 3 isn't just
  "confirm cheap import" — it's "restructure `load_scripts()`/script modules to separate
  UI-schema loading from processing-logic loading," since today they're inseparable. This is
  a real code change to `modules/scripts.py` and to ~30 script files' import structure, not
  just an architectural decision.
- **Phase 4** ("Live preview & progress polish") now has a concrete decision to make instead
  of an open question: adopt/rebase the `feat/user-session-jobqueue` SSE mechanism (with the
  `queue.Queue`-based design, not the `asyncio.Queue` design the stale memory implied) versus
  building a new push mechanism from scratch.
