# PHASE7 — Launch & Docs, and Closing Out the BFISO Phase List

Executes [BFISO.md](BFISO.md) §6 Phase 7, the last phase in the original plan.

## Scope reconciliation

Phase 7 was originally worded assuming a fully-wired two-process architecture: "update
`webui.sh`/`launch.py` to spawn both processes... prove the two-process path produces
the same output as the current in-process path." That presupposes a Gradio frontend that
*calls* a separate backend process over HTTP. That's not what exists.

Back at the Phase 2 scope decision, the choice was explicitly "backend-only extraction"
over "full split: rewire Gradio UI to call the backend over HTTP" — the larger option was
correctly deferred because it collides with PHASE3.md's finding (script UI construction
can't be made torch-free without either the ScriptArg-schema work from PHASE4.md fully
wired in, or a much bigger core-module refactor). So what actually exists today is **two
independent, alternative launch modes**:

- `webui.sh` / `python launch.py` — the original, unchanged, single-process app. Gradio UI
  calls `processing.py` in-process, exactly as before any of this work started.
- `webui-backend.sh` (PHASE2.md) — a separate, additional way to run just the inference
  pipeline behind a FastAPI process, no Gradio attached.

These are not two halves of one running system yet — nothing spawns both together, and
the Gradio UI does not talk to the backend-only process over HTTP. Documenting this
honestly rather than writing launch docs that imply a collaboration that doesn't exist.

**Reinterpreted the "prove same output" requirement accordingly**: instead of proving two
processes-that-talk-to-each-other reproduce single-process output (there's no such
system yet), proved that the backend-only *extraction itself* didn't change generation
behavior — i.e., that `webui-backend.sh`'s output is identical to `webui.sh`'s for the
same request. That's the meaningful, honest version of this phase's verification goal
given what was actually built.

## Docs

Added a "Backend-only mode" subsection to [CLAUDE.md](CLAUDE.md)'s "Running the
Application" section, alongside the existing default launch instructions. States plainly
that this is an alternative mode today, not a collaborating one, and points at BFISO.md's
phase list and PHASE3.md's blocker for why the full split isn't a small remaining step.

## Equivalence verification

Fired the identical, fully deterministic request (fixed `seed=12345`, `subseed=999`,
`subseed_strength=0`, fixed sampler/steps/size) against both launch modes sequentially,
same machine, same checkpoint (`waiIllustriousSDXL_v170.safetensors`):

| | full `launch.py` (`--api`) | `webui-backend.sh` (`--nowebui --api`) |
|---|---|---|
| image SHA-256 | `49257ca8...562ec7` | `49257ca8...562ec7` |
| image bytes | 278332 | 278332 |
| full `info` dict (seed, subseed, sampler, cfg_scale, model hash, version, ...) | — | — |

First pass (subseed left unpinned/random) already showed byte-identical images despite
different random subseed values — expected, since `subseed_strength=0` means subseed has
zero effect on the actual pixels; still re-ran with subseed pinned too, for a completely
clean diff rather than leaving an asterisk in the result. Second pass:
`assert a == b` on the full result dict (image hash + every `info` field) — **passed,
zero differences**.

This confirms Phase 2's extraction (`--nowebui --api` mode) is a faithful, output-identical
alternative to the full launch for the same request — the backend-only path isn't
accidentally exercising a different code path or producing subtly different results.

## Where the whole BFISO effort (Phases 0–7) actually stands

**Done and verified**, each with a real test against a running process, not just written:
- Phase 1: venv/numpy independence is achievable (subprocess spike, numpy 1.26.4 vs 2.5.1
  simultaneously).
- Phase 2: the inference pipeline runs standalone, debugger-attachable, API-only.
- Phase 3: precisely how far torch-import coupling goes (577 files / 1306 blockers) —
  the finding that ruled out a mechanical fix and pointed at Phase 4's approach instead.
- Phase 4: a backend-served, torch-free script-UI schema exists and reconstructs 406/415
  real script args byte-for-byte.
- Phase 5: a real SSE progress endpoint exists and was proven against a live generation.
- Phase 6: the shipped frontend JS actually uses that SSE endpoint now, proven in a real
  browser.
- Phase 7 (this phase): the backend-only extraction is output-identical to the original.

**Not done — genuinely remaining work**, not just deferred paperwork:
- The Gradio frontend still runs entirely in-process; it does not call
  `webui-backend.sh` (or any separate backend) over HTTP. Making that real means finishing
  what PHASE4.md's schema builder started (wiring `build_controls_from_schema` into
  `modules/ui.py` in place of `script.ui()` calls) and resolving `modules/safe.py`'s real
  design question from PHASE3.md.
- No launch script spawns both processes together, because there's no wired collaboration
  yet for it to spawn.
- Only `progressbar.js`'s txt2img path was driven through a real browser in Phase 6;
  `extensions.js`/`textualInversion.js`'s call sites share the same code but weren't
  individually exercised.
- The frontend venv still has torch installed (it's the same `venv-gr3` as the backend
  today) — true numpy-version independence between the two processes hasn't been
  exercised end-to-end since Phase 1's isolated spike; it depends on the still-open work
  above.

The honest summary: this work built and verified every piece needed for the eventual
split — the wire contract, the backend extraction, the script-UI schema mechanism, the
progress channel — and proved each one works, but did not perform the actual cutover
(rewiring the Gradio process to be a thin HTTP client). That remains real, scoped,
identifiable future work, not an open-ended unknown.
