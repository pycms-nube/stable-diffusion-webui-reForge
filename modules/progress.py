import asyncio
import base64
import io
import time

import gradio as gr
from fastapi import Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import modules.shared as shared
from collections import OrderedDict
import string
import random
from typing import List, Optional

current_task = None
pending_tasks = OrderedDict()
finished_tasks = []
recorded_results = []
recorded_results_limit = 2

# Give up on an SSE stream whose id_task never becomes active/queued/completed
# (e.g. a bad or stale id), so a client that never disconnects can't leave a
# generator polling forever.
PROGRESS_STREAM_UNKNOWN_TASK_TIMEOUT = 300


def start_task(id_task):
    global current_task

    current_task = id_task
    pending_tasks.pop(id_task, None)


def finish_task(id_task):
    global current_task

    if current_task == id_task:
        current_task = None

    finished_tasks.append(id_task)
    if len(finished_tasks) > 16:
        finished_tasks.pop(0)

def create_task_id(task_type):
    N = 7
    res = ''.join(random.choices(string.ascii_uppercase +
    string.digits, k=N))
    return f"task({task_type}-{res})"

def record_results(id_task, res):
    recorded_results.append((id_task, res))
    if len(recorded_results) > recorded_results_limit:
        recorded_results.pop(0)


def add_task_to_queue(id_job):
    pending_tasks[id_job] = time.time()

class PendingTasksResponse(BaseModel):
    size: int = Field(title="Pending task size")
    tasks: List[str] = Field(title="Pending task ids")

class ProgressRequest(BaseModel):
    id_task: Optional[str] = Field(default=None, title="Task ID", description="id of the task to get progress for")
    id_live_preview: int = Field(default=-1, title="Live preview image ID", description="id of last received last preview image")
    live_preview: bool = Field(default=True, title="Include live preview", description="boolean flag indicating whether to include the live preview image")


class ProgressResponse(BaseModel):
    active: bool = Field(title="Whether the task is being worked on right now")
    queued: bool = Field(title="Whether the task is in queue")
    completed: bool = Field(title="Whether the task has already finished")
    progress: Optional[float] = Field(default=None, title="Progress", description="The progress with a range of 0 to 1")
    eta: Optional[float] = Field(default=None, title="ETA in secs")
    live_preview: Optional[str] = Field(default=None, title="Live preview image", description="Current live preview; a data: uri")
    id_live_preview: Optional[int] = Field(default=None, title="Live preview image ID", description="Send this together with next request to prevent receiving same image")
    textinfo: Optional[str] = Field(default=None, title="Info text", description="Info text used by WebUI.")


def setup_progress_api(app):
    app.add_api_route("/internal/pending-tasks", get_pending_tasks, methods=["GET"])
    app.add_api_route("/internal/progress-stream", progress_stream, methods=["GET"])
    return app.add_api_route("/internal/progress", progressapi, methods=["POST"], response_model=ProgressResponse)


def get_pending_tasks():
    pending_tasks_ids = list(pending_tasks)
    pending_len = len(pending_tasks_ids)
    return PendingTasksResponse(size=pending_len, tasks=pending_tasks_ids)


def _compute_progress(req: ProgressRequest) -> ProgressResponse:
    """Core progress computation, shared by the POST /internal/progress (long-poll)
    and GET /internal/progress-stream (SSE) endpoints so they can't drift apart."""
    active = req.id_task == current_task
    queued = req.id_task in pending_tasks
    completed = req.id_task in finished_tasks

    if not active:
        textinfo = "Waiting..."
        if queued:
            sorted_queued = sorted(pending_tasks.keys(), key=lambda x: pending_tasks[x])
            queue_index = sorted_queued.index(req.id_task)
            textinfo = "In queue: {}/{}".format(queue_index + 1, len(sorted_queued))
        return ProgressResponse(active=active, queued=queued, completed=completed, id_live_preview=-1, textinfo=textinfo)

    progress = 0

    job_count, job_no = shared.state.job_count, shared.state.job_no
    sampling_steps, sampling_step = shared.state.sampling_steps, shared.state.sampling_step

    if job_count > 0:
        progress += job_no / job_count
    if sampling_steps > 0 and job_count > 0:
        progress += 1 / job_count * sampling_step / sampling_steps

    progress = min(progress, 1)

    elapsed_since_start = time.time() - shared.state.time_start
    predicted_duration = elapsed_since_start / progress if progress > 0 else None
    eta = predicted_duration - elapsed_since_start if predicted_duration is not None else None

    live_preview = None
    id_live_preview = req.id_live_preview

    if shared.opts.live_previews_enable and req.live_preview:
        shared.state.set_current_image()
        if shared.state.id_live_preview != req.id_live_preview:
            image = shared.state.current_image
            if image is not None:
                buffered = io.BytesIO()

                if shared.opts.live_previews_image_format == "png":
                    # using optimize for large images takes an enormous amount of time
                    if max(*image.size) <= 256:
                        save_kwargs = {"optimize": True}
                    else:
                        save_kwargs = {"optimize": False, "compress_level": 1}

                else:
                    save_kwargs = {}

                image.save(buffered, format=shared.opts.live_previews_image_format, **save_kwargs)
                base64_image = base64.b64encode(buffered.getvalue()).decode('ascii')
                live_preview = f"data:image/{shared.opts.live_previews_image_format};base64,{base64_image}"
                id_live_preview = shared.state.id_live_preview

    return ProgressResponse(active=active, queued=queued, completed=completed, progress=progress, eta=eta, live_preview=live_preview, id_live_preview=id_live_preview, textinfo=shared.state.textinfo)


def progressapi(req: ProgressRequest):
    return _compute_progress(req)


async def _progress_event_generator(request: Request, req: ProgressRequest):
    """Server-side polling of the same state progressapi() reads, pushed as SSE
    instead of waiting for the client to re-request. Not event-driven -- it's a
    timer loop reusing _compute_progress() -- but that's a deliberate choice: it
    needed no changes to the sampler/shared_state at all, only this endpoint."""
    interval = max((shared.opts.live_preview_refresh_period or 500) / 1000, 0.1)
    unknown_since = None

    while True:
        if await request.is_disconnected():
            break

        response = _compute_progress(req)
        yield f"data: {response.model_dump_json()}\n\n"

        if response.completed:
            break

        if not response.active and not response.queued:
            unknown_since = unknown_since or time.time()
            if time.time() - unknown_since > PROGRESS_STREAM_UNKNOWN_TASK_TIMEOUT:
                break
        else:
            unknown_since = None

        # carry forward the live preview id we just sent, same as the long-poll
        # client does, so unchanged frames aren't re-encoded and re-sent
        if response.id_live_preview is not None:
            req.id_live_preview = response.id_live_preview

        await asyncio.sleep(interval)


def progress_stream(request: Request, id_task: str, id_live_preview: int = -1, live_preview: bool = True):
    """GET, not POST: browsers' native EventSource API only supports GET with no
    custom body, so params travel as a query string instead of the ProgressRequest
    JSON body progressapi() takes."""
    req = ProgressRequest(id_task=id_task, id_live_preview=id_live_preview, live_preview=live_preview)
    return StreamingResponse(_progress_event_generator(request, req), media_type="text/event-stream")


def restore_progress(id_task):
    while id_task == current_task or id_task in pending_tasks:
        time.sleep(0.1)

    res = next(iter([x[1] for x in recorded_results if id_task == x[0]]), None)
    if res is not None:
        return res

    return gr.update(), gr.update(), gr.update(), f"Couldn't restore progress for {id_task}: results either have been discarded or never were obtained"
