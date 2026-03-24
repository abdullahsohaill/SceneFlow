"""
SceneFlow API — FastAPI Application
=====================================
Exposes the three core REST endpoints:
  POST /api/v1/draft   → Generate a Scene DSL from a topic (LLM)
  POST /api/v1/render  → Queue a video rendering job (Celery)
  GET  /api/v1/jobs/{job_id}  → Check job status & get result URL
"""

import uuid
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    DraftRequest,
    DraftResponse,
    RenderRequest,
    RenderResponse,
    JobStatusResponse,
    ManimDraftResponse,
    ManimRenderRequest,
)
from .llm_engine import generate_draft, generate_manim_scenes
from .celery_worker import celery_app, render_video_task, render_manim_task

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import os

app = FastAPI(
    title="SceneFlow API",
    description=(
        "API-first engine that converts raw text into "
        "structured, multi-scene explainer videos."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins for dev (tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure output directory exists before mounting
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/tmp/sceneflow_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount outputs so frontend can load <video src="/outputs/...">
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# Mount frontend UI
app.mount("/static", StaticFiles(directory="static"), name="static")


# ══════════════════════════════════════════════
# Health Check / Root Index
# ══════════════════════════════════════════════

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Redirect to the simple frontend UI."""
    return RedirectResponse(url="/static/index.html")


# ══════════════════════════════════════════════
# POST /api/v1/draft
# ══════════════════════════════════════════════

@app.post(
    "/api/v1/draft",
    response_model=DraftResponse,
    tags=["drafting"],
    summary="Generate a Scene DSL from a topic",
)
async def create_draft(request: DraftRequest):
    """
    Accepts a topic, audience, and brand settings.
    Calls the LLM to produce a structured VideoDraft (Scene DSL).
    Returns the JSON blueprint — does NOT render video.
    """
    try:
        logger.info("Draft request: topic='%s', audience='%s'", request.topic, request.audience)

        draft = generate_draft(request)

        logger.info("Draft generated: %d scenes", len(draft.scenes))
        return DraftResponse(draft=draft)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Draft generation failed")
        raise HTTPException(status_code=500, detail=f"Draft generation failed: {str(e)}")


# ══════════════════════════════════════════════
# POST /api/v1/render
# ══════════════════════════════════════════════

@app.post(
    "/api/v1/render",
    response_model=RenderResponse,
    tags=["rendering"],
    summary="Queue a video rendering job",
)
async def render_video(request: RenderRequest):
    """
    Accepts a (possibly edited) VideoDraft JSON.
    Dispatches a background Celery task to render the video.
    Returns a job_id for polling.
    """
    try:
        job_id = str(uuid.uuid4())

        logger.info(
            "Render request: job_id='%s', scenes=%d",
            job_id,
            len(request.draft.scenes),
        )

        # Serialize the Pydantic model to a dict for Celery's JSON serializer
        draft_dict = request.draft.model_dump(mode="json")

        # Dispatch the async task
        render_video_task.apply_async(
            args=[draft_dict, job_id],
            task_id=job_id,
        )

        logger.info("Render job %s queued", job_id)
        return RenderResponse(job_id=job_id)

    except Exception as e:
        logger.exception("Failed to queue render job")
        raise HTTPException(status_code=500, detail=f"Failed to queue render: {str(e)}")


# ══════════════════════════════════════════════
# GET /api/v1/jobs/{job_id}
# ══════════════════════════════════════════════

@app.get(
    "/api/v1/jobs/{job_id}",
    response_model=JobStatusResponse,
    tags=["jobs"],
    summary="Check rendering job status",
)
async def get_job_status(job_id: str):
    """
    Poll the status of a rendering job.
    Returns status (queued/processing/completed/failed) and the result URL
    when complete.
    """
    # Query the Celery result backend
    result = celery_app.AsyncResult(job_id)

    # Map Celery states to our API states
    if result.state == "PENDING":
        status = "queued"
    elif result.state == "STARTED" or result.state == "RETRY":
        status = "processing"
    elif result.state == "SUCCESS":
        # The task returns a dict with "status" and "result_url" or "error"
        task_result = result.result or {}
        return JobStatusResponse(
            job_id=job_id,
            status=task_result.get("status", "completed"),
            result_url=task_result.get("result_url"),
            error=task_result.get("error"),
        )
    elif result.state == "FAILURE":
        return JobStatusResponse(
            job_id=job_id,
            status="failed",
            error=str(result.info),
        )
    else:
        status = "processing"

    return JobStatusResponse(job_id=job_id, status=status)


# ══════════════════════════════════════════════
# POST /api/v1/draft-manim
# ══════════════════════════════════════════════

@app.post(
    "/api/v1/draft-manim",
    response_model=ManimDraftResponse,
    tags=["manim"],
    summary="Generate Manim animation code from a topic",
)
async def create_manim_draft(request: DraftRequest):
    """
    Asks Gemini to generate Manim Python animation code for each scene.
    Returns narration text + Manim source code per scene.
    """
    try:
        logger.info("Manim draft request: topic='%s'", request.topic)
        draft = generate_manim_scenes(request)
        logger.info("Manim draft generated: %d scenes", len(draft.scenes))
        return ManimDraftResponse(draft=draft)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Manim draft generation failed")
        raise HTTPException(status_code=500, detail=f"Manim draft failed: {str(e)}")


# ══════════════════════════════════════════════
# POST /api/v1/render-manim
# ══════════════════════════════════════════════

@app.post(
    "/api/v1/render-manim",
    response_model=RenderResponse,
    tags=["manim"],
    summary="Queue Manim video rendering job",
)
async def render_manim_video(request: ManimRenderRequest):
    """
    Accepts a ManimDraft (narration + Manim code per scene).
    Dispatches a Celery task that renders animated videos.
    """
    try:
        job_id = str(uuid.uuid4())
        logger.info("Manim render: job_id='%s', scenes=%d", job_id, len(request.draft.scenes))

        draft_dict = request.draft.model_dump(mode="json")
        render_manim_task.apply_async(args=[draft_dict, job_id], task_id=job_id)

        return RenderResponse(job_id=job_id)
    except Exception as e:
        logger.exception("Failed to queue Manim render")
        raise HTTPException(status_code=500, detail=f"Failed to queue: {str(e)}")
