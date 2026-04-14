"""
SceneFlow API — Manim Video Pipeline
======================================
Orchestrates the Audio-First Multi-Agent rendering flow:
  1. Generate all TTS files.
  2. One single LLM Prompt with all scenes + audio durations.
  3. Single Manim execution for continuous seamless context.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
import subprocess

from mutagen.mp3 import MP3

from .config import settings
from .manim_renderer import render_manim_scene
from .schemas import DirectorStoryboard, DirectorScene
from .llm_engine import generate_full_manim_code

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# Step 1 — Audio Generation
# ══════════════════════════════════════════════

def generate_audio_step(scene: DirectorScene, work_dir: Path) -> tuple[Path, float]:
    """Generate TTS audio for a scene using edge-tts."""
    audio_path = work_dir / f"{scene.scene_id}.mp3"

    logger.info("Generating TTS for %s (%d chars)", scene.scene_id, len(scene.narration_text))

    cmd = [
        "edge-tts",
        "--voice", settings.TTS_VOICE,
        "--text", scene.narration_text,
        "--write-media", str(audio_path)
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    audio_info = MP3(str(audio_path))
    duration = audio_info.info.length

    logger.info("Audio for %s: %.2fs → %s", scene.scene_id, duration, audio_path)
    return audio_path, duration


# ══════════════════════════════════════════════
# Post-Processing
# ══════════════════════════════════════════════

def cleanup(work_dir: Path, final_video: Path, output_dir: Path, job_id: str) -> Path:
    """Move final video to output directory and clean up intermediates."""
    job_output_dir = output_dir / job_id
    job_output_dir.mkdir(parents=True, exist_ok=True)

    final_dest = job_output_dir / "final_video.mp4"
    shutil.move(str(final_video), str(final_dest))
    shutil.rmtree(str(work_dir), ignore_errors=True)

    logger.info("Cleaned up. Final output → %s", final_dest)
    return final_dest


# ══════════════════════════════════════════════
# Orchestrator — run_manim_pipeline
# ══════════════════════════════════════════════

def run_manim_pipeline(draft: DirectorStoryboard, job_id: str) -> str:
    """
    Monolithic Pipeline:
    1. Generate all TTS files sequentially.
    2. Pass entire storyboard and TTS metadata to LLM.
    3. Generate single ExplainerScene and render.
    """
    settings.validate()

    work_dir = Path(tempfile.mkdtemp(prefix=f"sceneflow_manim_{job_id}_"))
    scene_work_dir = work_dir / "render"
    scene_work_dir.mkdir(exist_ok=True)
    
    logger.info("Manim Monolithic Pipeline started for job %s", job_id)

    bg_color = "#0F172A"
    if draft.style_config.brand_colors:
        bg_color = draft.style_config.brand_colors[0]

    # --- Phase 1: Generate All Audio ---
    scenes_data = []

    for idx, scene in enumerate(draft.scenes):
        tts_path, tts_duration = generate_audio_step(scene, work_dir)
        
        scenes_data.append({
            "narration": scene.narration_text,
            "visuals": scene.visual_description,
            "audio_path": str(tts_path),
            "duration": tts_duration
        })

    # --- Phase 2: Monolithic Code Generation ---
    logger.info(f"Generating Code for {len(scenes_data)} scenes in ONE shot...")
    manim_code = generate_full_manim_code(
        scenes_data=scenes_data,
        bg_color=bg_color
    )

    # --- Phase 3: Single Render Execution ---
    logger.info(f"Starting Monolithic Rendering...")
    final_video_path = render_manim_scene(
        manim_code=manim_code,
        scene_class_name="ExplainerScene",
        work_dir=scene_work_dir,
        quality="l",
        narration_text="Combined Monolithic Narration",
    )

    # --- Phase 4: Finalize ---
    final_dest = cleanup(work_dir, final_video_path, Path(settings.OUTPUT_DIR), job_id)

    logger.info("Monolithic pipeline completed for job %s", job_id)
    return str(final_dest)
