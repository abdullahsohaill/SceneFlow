"""
SceneFlow API — Celery Worker
==============================
Initializes the Celery app with a Redis broker and defines the
asynchronous video rendering task.
"""

import logging
import uuid

from celery import Celery

from .config import settings
from .schemas import ManimDraft
from .manim_pipeline import run_manim_pipeline

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Celery App Initialization
# ──────────────────────────────────────────────

celery_app = Celery(
    "sceneflow",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,       # Store task results in Redis too
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Prevent prefetching — each render task is heavy
    worker_prefetch_multiplier=1,
    # Long timeout for video rendering (10 minutes)
    task_time_limit=600,
    task_soft_time_limit=540,
)


@celery_app.task(bind=True, name="sceneflow.render_manim")
def render_manim_task(self, draft_dict: dict, job_id: str) -> dict:
    """
    Celery task that runs the Manim-based rendering pipeline.
    Each scene's LLM-generated Manim code is executed to produce
    animated .mp4 segments, overlaid with TTS audio, and concatenated.
    """
    try:
        from .schemas import ManimDraft
        from .manim_pipeline import run_manim_pipeline

        logger.info("Starting Manim render task for job %s", job_id)

        draft = ManimDraft.model_validate(draft_dict)
        result_path = run_manim_pipeline(draft, job_id)

        from pathlib import Path
        file_name = Path(result_path).name
        logger.info("Manim render completed for job %s → %s", job_id, result_path)
        return {
            "status": "completed",
            "result_url": f"/outputs/{job_id}/{file_name}",
        }

    except Exception as exc:
        logger.exception("Manim render failed for job %s", job_id)
        return {
            "status": "failed",
            "error": str(exc),
        }
