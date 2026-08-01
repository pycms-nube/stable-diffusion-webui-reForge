# PHASE15 — Advanced Hires. fix Overrides

Closes the "no advanced hires overrides" gap named since PHASE11.md: `hr_checkpoint_name`,
`hr_sampler_name`, `hr_scheduler`, `hr_prompt`, `hr_negative_prompt`, `hr_cfg`, letting
the second pass use a different checkpoint/sampler/scheduler/prompt/CFG than the base
pass, tucked into a nested "Advanced Hires. fix overrides" sub-accordion.

## The real gotcha: sentinel translation the REST API doesn't do for you

The shipped Gradio UI's hires dropdowns default to sentinel strings — "Use same
checkpoint" / "Use same sampler" / "Use same scheduler" — and `modules/txt2img.py`'s
`txt2img_create_processing()` (the function the Gradio *click handler* calls) converts
those sentinels to `None` before constructing the `Processing` object:
```python
hr_sampler_name=None if hr_sampler_name == 'Use same sampler' else hr_sampler_name,
hr_scheduler=None if hr_scheduler == 'Use same scheduler' else hr_scheduler,
```
This conversion lives in the Gradio-only wrapper, **not** in `modules/api/api.py`'s
`text2imgapi` (the REST endpoint this frontend actually calls). `hr_checkpoint_name`
has an internal fallback for its own sentinel (`processing.py`'s `init()` explicitly
checks `!= 'Use same checkpoint'`), but `hr_sampler_name`/`hr_scheduler` have no such
check baked into the model itself — `img2img_sampler_name = self.hr_sampler_name or
self.sampler_name` only handles `None`/empty, not the literal sentinel string. Sending
the sentinel text as-is over the REST API would have looked for a sampler literally
named "Use same sampler" and failed. Found this by reading the actual code path this
frontend uses (not the Gradio one) before writing `run_txt2img()`'s payload-building
code, not by hitting the bug live — `run_txt2img()` replicates the same
sentinel-to-`None` translation client-side.

## What changed

- `modules_frontend/common.py`: `fetch_sd_models()` (returns checkpoint `title`s,
  matching what `sd_models.get_closet_checkpoint_match()` fuzzy-matches against) and
  `fetch_schedulers()` (returns scheduler `label`s, matching the shipped UI's own
  `hr_scheduler` dropdown convention — `processing.py` stores whatever string it's
  given verbatim for the *hires* scheduler, unlike the base `scheduler` field which
  gets normalized through `get_sampler_and_scheduler()`).
- `modules_frontend/txt2img_ui.py`: new sub-accordion with Hires checkpoint/sampler/
  schedule-type dropdowns (each prefixed with its own "Use same X" sentinel, default
  selected), Hires CFG Scale slider (0-30, default 0.0 — matches the shipped UI's own
  default, which is *not* the backend dataclass's own default of 1.0), and Hires
  prompt/negative prompt textboxes.
- `run_txt2img()`: the three sentinel translations plus `hr_prompt`/
  `hr_negative_prompt`/`hr_cfg` added to the payload unconditionally.

## Verification

**Sentinel-default path, called directly**: ran `run_txt2img()` as a plain generator
with all six overrides left at their sentinel defaults — completed successfully
against the real backend, `Hires CFG Scale: 7.0` in the returned infotext matching the
base CFG (7.0), confirming the `hr_cfg=0.0` sentinel value correctly fell back rather
than being sent as a literal (and nonsensical) zero CFG scale.

**Explicit-override path, called directly**: same call with real values —
`hr_sampler_name="DPM++ 2M"`, `hr_scheduler="Karras"`, `hr_prompt="a hires-only
prompt"`, `hr_cfg=3.5` — returned infotext included `Hires sampler: DPM++ 2M`,
`Hires prompt: a hires-only prompt`, `Hires CFG Scale: 3.5`, `Hires schedule type:
Karras`, confirming all four genuinely reached and affected the pipeline (not just
accepted without effect).

**Live UI**: user manually confirmed the dropdowns populate with real checkpoint/
sampler/scheduler data, and a full generation with all overrides left at "Use same X"
defaults completed successfully.

## What this phase did NOT do

- **`hr_checkpoint_name`'s actual checkpoint-switching behavior wasn't independently
  verified** (only its sentinel-default no-op path was) — picking a *different* real
  checkpoint for the hires pass would trigger a model swap mid-generation, which is
  slow and wasn't exercised this phase; the payload wiring is verified, the runtime
  behavior of an actual mid-generation checkpoint switch is not.
- Still **no Sketch/Batch sub-tabs, no cancel confirmation UX, flat layout** — same
  named gaps as every prior phase.
