"""
SceneFlow API — Manim Video Pipeline
======================================
Orchestrates the Audio-First Multi-Agent rendering flow:
  1. TTS audio generation  (edge-tts)
  2. LLM Animator generation (pass audio duration to sync)
  3. Manim rendering       (Code + Audio -> .mp4)
  4. Final concatenation    (ffmpeg concat demuxer)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from mutagen.mp3 import MP3

from .config import settings
from .manim_renderer import render_manim_scene
from .schemas import DirectorStoryboard, DirectorScene
from .llm_engine import generate_scene_manim_code

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# Step 1 — Audio Generation
# ══════════════════════════════════════════════

def generate_audio(scene: DirectorScene, work_dir: Path) -> tuple[Path, float]:
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
# Step 2 — Concatenate All Scenes
# ══════════════════════════════════════════════

def concatenate_scenes(scene_videos: list[Path], job_id: str, work_dir: Path) -> Path:
    """Concatenate individual scene MP4s into one final video."""
    concat_file = work_dir / "concat_list.txt"
    with open(concat_file, "w") as f:
        for video_path in scene_videos:
            f.write(f"file '{video_path}'\\n")

    output_path = work_dir / f"final_video_{job_id}.mp4"

    logger.info("Concatenating %d scenes → %s", len(scene_videos), output_path)

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    logger.info("Final video → %s", output_path)
    return output_path


# ══════════════════════════════════════════════
# Step 3 — Cleanup
# ══════════════════════════════════════════════

def cleanup(work_dir: Path, final_video: Path, output_dir: Path, job_id: str) -> Path:
    """Move final video to output directory and clean up intermediates."""
    job_output_dir = output_dir / job_id
    job_output_dir.mkdir(parents=True, exist_ok=True)

    final_dest = job_output_dir / final_video.name
    shutil.move(str(final_video), str(final_dest))
    shutil.rmtree(str(work_dir), ignore_errors=True)

    logger.info("Cleaned up. Final output → %s", final_dest)
    return final_dest


# ══════════════════════════════════════════════
# Orchestrator — run_manim_pipeline
# ══════════════════════════════════════════════

def run_manim_pipeline(draft: DirectorStoryboard, job_id: str) -> str:
    """
    Multi-Agent Audio-First Pipeline:
    1. Generate TTS per scene
    2. Feed audio length to Animator Agent for exact timing sync
    3. Render the code (Manim automatically merges the audio)
    4. Concat the clips.
    """
    settings.validate()

    work_dir = Path(tempfile.mkdtemp(prefix=f"sceneflow_manim_{job_id}_"))
    logger.info("Manim multi-agent pipeline started for job %s in %s", job_id, work_dir)

    scene_videos: list[Path] = []
    
    bg_color = "#0F172A"
    if draft.style_config.brand_colors:
        bg_color = draft.style_config.brand_colors[0]

    previous_code = ""

    for scene in draft.scenes:
        # 1. Generate Voice Audio (Get absolute duration)
        tts_path, tts_duration = generate_audio(scene, work_dir)

        # 2. Call Animator Agent (Context Chaining)
        logger.info(f"Agent 2 Context Chaining: Generating Manim Code for {scene.scene_id} using length {tts_duration:.2f}s")
        manim_code = generate_scene_manim_code(
            scene_plan=scene,
            bg_color=bg_color,
            audio_duration=tts_duration,
            audio_path=str(tts_path),
            previous_code=previous_code
        )
        
        # Save previous context state for the NEXT loop iteration
        previous_code = manim_code

        # 3. Compile Manim 
        scene_work_dir = work_dir / scene.scene_id
        scene_work_dir.mkdir(exist_ok=True)
        manim_video_path = render_manim_scene(
            manim_code=manim_code,
            scene_class_name="ExplainerScene",
            work_dir=scene_work_dir,
            quality="l",
            narration_text=scene.narration_text,
        )
        
        scene_videos.append(manim_video_path)

    # 4. Concatenate
    final_video = concatenate_scenes(scene_videos, job_id, work_dir)

    # 5. Cleanup
    final_dest = cleanup(work_dir, final_video, Path(settings.OUTPUT_DIR), job_id)

    logger.info("Manim pipeline completed for job %s → %s", job_id, final_dest)
    return str(final_dest)
