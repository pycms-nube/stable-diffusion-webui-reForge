# PHASE6 — Wire SSE Into the Frontend JS

Completes the item [PHASE5.md](PHASE5.md) explicitly deferred: the server-side SSE
endpoint existed and was verified, but nothing in the shipped UI consumed it yet.

## The WASM question

The user asked me to pay attention to the fact this JS predates WASM/modern browser
APIs and to look for improvements assuming client-side rendering and local processing.
Before touching code, checked whether there was already a relevant finding — there was:
claude-mem observation #96, "Gradio 3.x→4.x Migration Strategy: Decomposition Over
WASM/SSR" (a `WASM_SERVERR.md` study from an earlier session, no longer present in git
history — recorded in memory but apparently never committed). Its central finding:
**Gradio Lite (WASM/Pyodide) is architecturally infeasible for this app** — it needs
PyTorch and GPU access, neither available in a WASM/Pyodide sandbox. That finding was
about running the whole *inference backend* in-browser, a different question from
today's, but it's the right prior art to check before reaching for WASM as a lever.

For the actual code touched this phase (`progressbar.js`'s progress/live-preview path):
WASM doesn't fit either, for a narrower reason. The live preview is a base64-encoded
PNG/JPEG rendered via `new Image(); img.src = <data-uri>` — the browser's native image
decoder already handles this at (hardware-accelerated) native speed; there's no
CPU-bound JS work here for WASM to speed up. The actual legacy-code problem in this file
wasn't "too slow, needs WASM" — it was "two independent XHR-poll loops instead of one
push connection," a protocol problem, not a compute problem. So the real "written before
newer JS existed" issue here is the absence of `EventSource` (a fetch-adjacent, broadly
supported API), not the absence of WASM. Fixed that instead of forcing WASM in where the
evidence doesn't support it.

## What changed

**[javascript/progressbar.js](javascript/progressbar.js)**, rewritten:

- The old code ran **two** independent `setTimeout`-based polling loops per active task —
  `funProgress` (progress/ETA/title, `live_preview: false`) and `funLivePreview`
  (image frames, tracking `id_live_preview` across calls) — each issuing its own
  `XMLHttpRequest` `POST /internal/progress` every `live_preview_refresh_period`ms
  (500ms default), independently of each other.
- Replaced both with **one** native `EventSource` connection to
  `GET /internal/progress-stream?id_task=...&live_preview=<bool>`. The server-side
  `ProgressResponse` already carries both progress fields and an optional
  `live_preview` frame in the same payload (PHASE5.md), so one stream now covers what
  took two polling loops before — roughly half the request volume, and push instead of
  poll (lower latency, no wasted "nothing changed yet" round trips).
- `id_live_preview` bookkeeping moved server-side already (PHASE5.md's generator carries
  it forward internally) — the client no longer needs to track or resend it.
- Preserved `requestProgress()`'s exact public signature and behavior contract (calls
  `atEnd()` on completion, `onProgress(res)` on every update, same DOM elements/classes)
  so `javascript/ui.js`, `javascript/extensions.js`, and
  `javascript/textualInversion.js` — all of which call `requestProgress()` — needed zero
  changes.
- Modernized style while in the file anyway: `var` → `const`/`let`, and removed the
  `request()` raw-`XMLHttpRequest` helper entirely — confirmed via repo-wide grep it had
  no callers left anywhere (including extensions) once its only two call sites were
  replaced.
- Added bounded resilience the old code didn't have: `EventSource` auto-reconnects on
  transient drops by default (the old code was fail-fast — one XHR error killed the
  progress bar immediately). Capped at `MAX_CONSECUTIVE_ERRORS = 3` before giving up, so
  a single network hiccup doesn't kill the UI, but a truly dead connection still cleans
  up instead of hanging forever.
- Preserved the `inactivityTimeout` (default 40s) give-up behavior identically, evaluated
  per-message the same way the old code evaluated it per-poll-tick.

## Verification (real browser, real generation)

Launched the full webui (`launch.py`, no `--nowebui` — needed the actual Gradio UI this
time), opened it in the browser tool, typed a real prompt, and clicked the actual
Generate button (had to go through the DOM directly — a custom prompt-editor extension
intercepts synthetic keyboard/click events on the textbox in a way the accessibility-tree
click didn't trigger a submit; `document.querySelector('#txt2img_generate button').click()`
worked reliably).

- **Network tab**: exactly **one** request the whole generation —
  `GET /internal/progress-stream?id_task=task(...)&live_preview=true → 200 OK`. Zero
  `POST /internal/progress` calls — confirms the dual-polling path is genuinely gone,
  not just supplemented.
- **Live UI feedback**: browser tab title updated in real time from the SSE stream —
  observed `[40% ETA: 7s] Stable Diffusion` mid-generation, reverting to plain
  `Stable Diffusion` on completion (confirms `setTitle("")` cleanup ran).
- **Console**: no new errors. Only pre-existing `Wake Lock is not supported` warnings
  (same `console.error` call carried over unchanged from the original code — expected in
  this headless browser environment, unrelated to the rewrite).
- **Result**: 3 images rendered in `#txt2img_gallery` — the generation completed
  normally end to end through the real UI.

## What this phase did NOT do

- Did not touch `extensions.js`/`textualInversion.js`/`ui.js` — they call
  `requestProgress()` unchanged, and this phase didn't specifically exercise their code
  paths (only the txt2img path was driven through a real browser). The signature
  contract is unchanged, so no code-level risk, but their specific UI flows weren't
  individually re-tested.
- Did not add a long-poll fallback for environments where SSE might not traverse (e.g.
  certain buffering reverse proxies). This is a local-first tool typically run
  same-machine or same-LAN; if that changes, worth revisiting.
- Did not pursue any WASM-based change — assessed and rejected for stated reasons above,
  not silently skipped.
- Did not audit the rest of `javascript/` (contextMenus.js, extraNetworks.js, etc.) for
  similar modernization opportunities — scoped to the file this phase actually needed to
  touch for the SSE wiring.
