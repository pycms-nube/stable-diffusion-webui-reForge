# PHASE11 — Hires. fix

Adds Hires. fix to `modules_frontend/txt2img_ui.py`: `enable_hr` plus its five core
sub-params (`hr_scale`, `hr_upscaler`, `hr_second_pass_steps`, `denoising_strength`,
`hr_resize_x`/`hr_resize_y`), in their own Accordion next to Scripts.

## What changed

- New "Hires. fix" `gr.Accordion` with: Enable checkbox, Upscaler dropdown, Hires
  steps slider, Upscale-by slider, Denoising strength slider, and two resize-to number
  boxes (0 = use Upscale-by, matching `hr_resize_x`/`hr_resize_y`'s own `0` sentinel
  semantics in `modules/processing.py`).
- `fetch_hr_upscalers()` — new helper fetching `/sdapi/v1/latent-upscale-modes` +
  `/sdapi/v1/upscalers` and concatenating them, because `hr_upscaler` is validated
  backend-side against the union of `shared.latent_upscale_modes` and
  `shared.sd_upscalers` (`modules/processing.py`), not either list alone. A dropdown
  built from just one list would offer illegal choices that 422 on submit.
- `run_txt2img()` and the click handler's `inputs` extended with the 7 new values,
  slotted in payload order right after `restore_faces`/`tiling`.

## Verification

**Backend actually applies hires-fix, not just accepts the fields**: single `curl`
request with `enable_hr=true, hr_scale=1.5, hr_upscaler=Latent, hr_second_pass_steps=1,
denoising_strength=0.4` — returned infotext included `Hires upscale: 1.5`,
`Hires steps: 1`, `Hires upscaler: Latent`, `Denoising strength: 0.4`, confirming the
values reached the actual processing pipeline.

**Live UI**: user manually confirmed (per the established Phase 10 methodology — light
automated one-shot checks, dynamic/live UI behavior handed to manual review) the
Accordion renders, Enable + generate produces a genuinely upscaled 1024x1024 result
from a 512x512 base.

## Investigated: apparent "hires pass skipped" when Hires steps = 0

The user's manual test set Hires steps to `0` (the control's own default, matching the
shipped UI's own `hr_second_pass_steps` slider default in `modules/ui.py:326`) and
observed what looked like the hires pass being skipped — output resized to 1024x1024
correctly, but seemingly without the expected second denoising pass, contradicting the
documented "0 means use the same step count as the base sampling steps" semantics
(`modules/processing.py`: `steps=self.hr_second_pass_steps or self.steps`).

**Investigated with a controlled pixel-level comparison**, not just infotext trust:
two identical `curl` requests (same seed, same prompt, same everything else) differing
only in `hr_second_pass_steps` — one `0`, one explicit `20` (matching the base `steps`
value) — decoded both returned PNGs to raw pixel arrays and diffed them.
**Result: max pixel diff 0, byte-for-byte identical images.** The hires pass genuinely
runs at the full base step count in both cases; the `0` fallback works exactly as
documented.

**What's actually going on**: `modules/processing.py` only writes `"Hires steps": N`
into the infotext when the value is truthy (`if self.hr_second_pass_steps:`) — so
passing literal `0` (even though it correctly resolves to the base step count
internally) silently omits that line from the returned infotext, reading like the pass
never happened. Confirmed this is shared, pre-existing behavior and not something this
frontend introduced or diverges on: the shipped UI's own `hr_second_pass_steps` slider
also defaults to `0` (`modules/ui.py:326`), so it has the exact same infotext gap.
**No fix applied** — resolving the value client-side before sending it (so the infotext
would always show the actual number used) would make this frontend's request payload
diverge from what the reference shipped UI actually sends, trading one kind of fidelity
for another without a clear win. Documented here instead so the behavior is understood,
not silently worked around.

## What this phase did NOT do

- **No hr_checkpoint_name / hr_sampler_name / hr_scheduler / hr_prompt /
  hr_negative_prompt / hr_cfg** — the "advanced" hires overrides that let the second
  pass use a different checkpoint/sampler/scheduler/prompt/CFG than the base pass. Left
  at their backend defaults (same as base). Real gap if someone needs per-pass control.
- **No conditional visibility** — the Hires. fix controls always render inside their
  Accordion regardless of whether Enable is checked, same flat-layout tradeoff as every
  prior phase.
- Still **txt2img only, no img2img/other tabs**.
