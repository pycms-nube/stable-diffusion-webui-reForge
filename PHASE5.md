# PHASE5 — SSE Progress Streaming

Executes [BFISO.md](BFISO.md) §6 Phase 5, resolving PHASE0.md Risk 1.

## Scope decision

Investigated what porting the "real SSE work" from `feat/user-session-jobqueue`
(PHASE0.md Risk 1) would actually mean. It turned out to be a full persistent job-queue
subsystem — `modules/job_queue/{db,models,runner,task_helpers}.py`, ~1000 lines: a
SQLite-backed task DB, cookie-based user sessions, a background task runner, task
history/replay. SSE is just that subsystem's delivery mechanism, not a standalone piece.
Porting it wholesale would be a large, mostly tangential feature relative to BFISO's
actual goal (numpy/frontend-backend process isolation) — not something to pull in as a
side effect of "add progress push."

**Decision (user-confirmed): build a minimal, purpose-built SSE endpoint** for
progress/live-preview instead — reusing the thread-safe-queue + `StreamingResponse`
pattern conceptually, without the DB, sessions, or task persistence.

## What changed

1. **[modules/progress.py](modules/progress.py)** — extracted `progressapi()`'s
   computation into `_compute_progress(req) -> ProgressResponse`, so the existing
   long-poll endpoint and the new SSE one share one implementation and can't drift
   apart. `progressapi()` itself is now a one-line wrapper; behavior is unchanged
   (verified — see below).
2. **New `GET /internal/progress-stream`** — an async generator
   (`_progress_event_generator`) that polls `_compute_progress()` server-side at the
   existing `shared.opts.live_preview_refresh_period` interval (the same 500ms default
   the current long-poll JS client already uses — no new setting invented) and yields
   `text/event-stream` frames. It's timer-based, not truly event-driven — a deliberate
   choice: it needed zero changes to the sampler or `shared_state.py`, only this one new
   endpoint. Stops on `completed=true`, on client disconnect (`request.is_disconnected()`
   checked each loop), or after `PROGRESS_STREAM_UNKNOWN_TASK_TIMEOUT` (300s) if the
   task id never becomes active/queued/completed, so a bad task id can't leave an
   orphaned generator polling forever.
   Uses `GET` with query params (`id_task`, `id_live_preview`, `live_preview`), not
   `POST` with a JSON body like `progressapi()` — the browser's native `EventSource` API
   only supports `GET` with no custom body, so the wire shape had to change for this
   endpoint specifically.
3. **Real pre-existing bug found and fixed**: [webui.py](webui.py)'s `api_only_worker()`
   (the `--nowebui --api` backend-only path used since Phase 2) never called
   `progress.setup_progress_api(app)` at all — only the full Gradio launch path
   (`webui_worker()`) did. This meant `/internal/progress` (the long-poll endpoint,
   predating this phase) was already silently missing from backend-only mode, not just
   the new SSE endpoint. Fixed by adding the same `progress.setup_progress_api(app)`
   call to `api_only_worker()`. Squarely in scope: BFISO.md §4 already specifies progress
   as part of the backend's HTTP contract, so the backend-only process needs to actually
   serve it.

## Verification (real generation, not a mock)

Launched a real backend (`webui-backend.sh` + `waiIllustriousSDXL_v170.safetensors`),
fired a real `POST /sdapi/v1/txt2img` (30 steps, `force_task_id` set to a known id) in
one thread, and connected to `GET /internal/progress-stream` for that same task id
concurrently in another. First run: 404 — caught the `api_only_worker()` gap above,
fixed it, restarted, re-ran.

```
[sse] connected, status=200
[sse] active=True queued=False completed=False progress=0.0 ...
[sse] active=True queued=False completed=False progress=0.0 ...
[sse] active=True queued=False completed=False progress=0.0 ...
[sse] active=True queued=False completed=False progress=0.1 ...
[sse] active=True queued=False completed=False progress=0.3 ...
[sse] active=True queued=False completed=False progress=0.4666666666666667 ...
[sse] active=True queued=False completed=False progress=0.6666666666666666 ...
[sse] active=True queued=False completed=False progress=0.8333333333333334 ...
[sse] active=True queued=False completed=False progress=0.9666666666666667 ...
[generation] status=200 took=8.6s
[sse] active=False queued=False completed=True progress=None textinfo='Waiting...'

Total SSE events received: 10
RESULT: PASS -- SSE stream delivered real, monotonically increasing progress for an
actual generation, ending in completed=true.
```

10 real events, progress monotonically non-decreasing from 0.0 to 0.967, terminated by a
clean `completed=true`, stream closed on its own. Asserted (not eyeballed): event count,
monotonicity, a final completed event, and that progress actually advanced.

**Regression check**: re-ran `POST /internal/progress` (the existing long-poll endpoint)
directly — identical response shape/behavior to before the refactor. Re-ran
`pytest test/test_txt2img.py test/test_img2img.py --no-server` — same 13 passed / 2
failed (pre-existing, documented in PHASE2.md) / 1 skipped as every prior phase's
baseline. No new failures.

## What this phase did NOT do

- Did not touch `shared_state.py` or the sampler — the polling-loop design was chosen
  specifically to avoid that, per the scope decision above.
- Did not wire the SSE endpoint into `javascript/progressbar.js` or any frontend client —
  this phase proved the server side works against a real generation via a Python test
  client, not that the Gradio UI's JS has been switched over to consume it. The existing
  long-poll path is untouched and still the one the shipped UI uses.
- Did not port any part of the job-queue subsystem (DB, sessions, task runner) — see
  the scope decision above.
