# PHASE3 — How Deep the Torch-Import Coupling Goes

Executes [BFISO.md](BFISO.md) §6 Phase 3: resolve PHASE0.md Risk 2 by either confirming
script-module UI imports can be made frontend-safe, or concluding they can't and a
backend-served UI-description schema is needed instead. PHASE0 only checked
`extensions-builtin/*/scripts/*.py` (30/44 import torch at module scope). This phase
asks the bigger question: is the "just defer the imports" approach tractable at all,
once you follow the full transitive closure — including core `modules/` itself and
every extension's vendored dependency tree?

## Method

Two tools, used together:

1. **A libcst-based static scanner** (temporary dev tool — `pip install libcst` into
   `venv-gr3`, not added to `requirements.txt`; harmless to leave installed, or `pip
   uninstall libcst` if you want the venv clean again) that walks every `.py` file under
   `modules/`, `modules_forge/`, `extensions-builtin/`, `extensions/` and reports
   module-top-level imports of a "backend-only" package set (torch, ldm, ldm_patched
   minus a few confirmed-lightweight submodules, pytorch_lightning, transformers,
   diffusers, torchvision, safetensors, kornia, einops, accelerate, spandrel,
   open_clip, facexlib, basicsr, realesrgan, gfpgan, tomesd, clean_fid, resize_right,
   blendmodes), distinguishing true module-scope imports from ones already guarded by
   `TYPE_CHECKING` or nested in a function/class. This gives full-repo coverage in one
   pass, but over-approximates: it flags every file that *could* be a problem if
   reached, not just the files actually reached when building the script UI.
2. **An import-blocking probe**: hooks `builtins.__import__` to raise on any backend-only
   package, then does `import modules.scripts` for real and reports the exact first
   failure — the true, minimal, actually-executed import chain. Fix, re-run, repeat.
   This is slower (one failure at a time) but precise — it doesn't count code that's
   never actually reached.

## Full-repo static scan result

**577 files, 1306 unguarded module-top-level backend-only imports, 19 already
try/except-guarded.** Breakdown:

- **Core `modules/` (~55 files) and `modules_forge/` (~14 files).** Not a periphery
  problem — the backbone itself (`sd_models.py`, `processing.py`, `sd_hijack*.py`,
  `sd_samplers*.py`, `devices.py`, `interrogate.py`, `deepbooru.py`,
  `hypernetworks/hypernetwork.py`, `textual_inversion/*`, `models/sd3/*`, all of
  `modules_forge/`) imports torch/ldm/ldm_patched at module scope. Most of this is
  legitimate — it's the actual inference backend — but it means "frontend imports
  `modules.scripts`" was never just an extensions problem.
- **`extensions-builtin/` script entry points**: consistent with PHASE0.md's 30/44
  finding, now confirmed by AST instead of grep.
- **Vendored third-party ML codebases inside extensions — the real scale surprise.**
  `extensions-builtin/forge_legacy_preprocessors/annotator/` alone bundles substantial
  chunks of `mmcv`, `mmseg`, `detectron2`, and `oneformer` — complete research
  codebases, hundreds of files, many defining classes like `class X(torch.nn.Module)`
  where the torch dependency is baked into the class hierarchy at class-definition
  time, not something you can move into a function body. `forge_preprocessor_inpaint`
  vendors `lama`; `forge_preprocessor_normalbae` vendors an `efficientnet_repo`;
  `forge_preprocessor_marigold` depends on `diffusers` directly for its pipeline.

## Import-blocking probe: two real fixes attempted, two very different outcomes

**Fix 1 — applied, real, zero-risk.** [modules/sd_models_types.py](modules/sd_models_types.py)
had `from ldm.models.diffusion.ddpm import LatentDiffusion` at module scope, used only
so `class WebuiSdModel(LatentDiffusion)` — a pure typing stub the docstring itself says
"is not actually instantiated, but its fields are created and filled by webui" — could
inherit its field types. Confirmed via repo-wide grep that `WebuiSdModel` is never
instantiated or `isinstance`-checked anywhere (only used as a type annotation in
`modules/processing.py:481` and `modules/shared.py:47`). Changed to:
```python
if TYPE_CHECKING:
    from ldm.models.diffusion.ddpm import LatentDiffusion
else:
    LatentDiffusion = object
```
This unblocks `modules/shared.py` — imported by nearly everything, including
`modules/scripts.py` — with **no runtime behavior change**. Verified: re-ran
`pytest test/test_txt2img.py --no-server` against a live SDXL-loaded backend
(`webui-backend.sh` + `waiIllustriousSDXL_v170.safetensors`) after the change — still
11 passed, 1 skipped, identical to the PHASE2.md baseline. No regression.

**Next blocker — real, and qualitatively different.** With Fix 1 applied, the probe gets
past `cmd_args.py` → `shared.py` → `shared_items.py` → `script_callbacks.py` →
`extensions.py` → `cache.py` → `modules/paths.py:5` (`import modules.safe  # noqa: F401`)
→ [modules/safe.py:6](modules/safe.py#L6) (`import torch`). Unlike `sd_models_types.py`,
this is **not** a type-only usage — `modules/safe.py` implements a security-hardened
`torch.load` unpickler (`TypedStorage = torch.storage.TypedStorage`,
`unsafe_torch_load = torch.load` as live module-level bindings, used at runtime by
`RestrictedUnpickler`/`load_with_extra`). It's genuinely load-bearing backend
functionality. It's reached from `modules/paths.py` — a foundational module imported by
nearly everything — via a bare side-effect import (the `# noqa: F401` confirms nothing
in `paths.py` actually references `modules.safe` directly). Deferring this is *possible*
(move the import out of `paths.py` to wherever the pickle-safety wiring is actually
needed, and turn the module-level `TypedStorage`/`unsafe_torch_load` bindings into
lazy accessors) but it's a real design decision about where checkpoint-loading security
should be wired up, not a mechanical one-line change — stopped here rather than making
that call unilaterally.

## Conclusion

BFISO.md Phase 3's first option — "confirm cheap script-module import for UI reflection
stays frontend-safe" — is **decisively no**, at a scale well beyond what PHASE0.md
estimated. It's not "fix 30 extension scripts." It's core `modules/` (dozens of files,
mostly legitimate backend code that shouldn't move) plus, transitively through several
extensions, hundreds of files from vendored third-party ML research codebases where
torch is often baked into class definitions rather than movable import statements. A
mechanical "defer every import" sweep is not a tractable engineering task here — most of
what's flagged is either code that genuinely needs to stay backend-side (correctly) or
vendored code not worth hand-refactoring.

**Recommendation: pursue BFISO.md's second option — a backend-served UI-description
schema — rather than continuing the defer-imports approach.** This is not starting from
zero: `/sdapi/v1/script-info` ([modules/api/api.py:244](modules/api/api.py#L244)) and
`ScriptInfo`/`ScriptArg` ([modules/scripts.py:673-696](modules/scripts.py#L673)) already
exist and already capture `label`/`value`/`minimum`/`maximum`/`step`/`choices` for every
script argument — generated backend-side today, where torch is already available. What's
missing for a true frontend rewrite: the schema doesn't currently capture *widget type*
(Slider vs. Checkbox vs. Dropdown vs. Textbox — today it's inferred implicitly from
which Gradio component happened to produce those fields) or layout (Accordions, Rows,
conditional visibility toggles used by e.g. ControlNet's dynamic dropdowns). Extending
`ScriptArg` with an explicit component-type field and building a small frontend-side
"JSON schema → gr.Component" constructor is bounded, real work — but it's boundable and
low-risk in a way that hand-auditing hundreds of vendored files is not. That's the
concrete next phase, not attempted here.

## What this phase did NOT do

- Did not defer `modules/safe.py`'s import or move it out of `modules/paths.py` — a real
  design decision (where should pickle-safety wiring live?) rather than a mechanical fix,
  left for whoever picks up the backend-schema work to decide alongside it.
- Did not attempt to fix any of the ~64 remaining core `modules/`/`modules_forge/` files
  or any extension's vendored dependency tree — per the conclusion above, most of that
  work isn't the right thing to do at all, not just deferred.
- Did not extend `ScriptInfo`/`ScriptArg` or build the frontend-side schema-to-UI
  constructor described in the recommendation — that's the actual next phase.
