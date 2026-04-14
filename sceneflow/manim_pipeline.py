"""
SceneFlow API — Manim Video Pipeline
======================================
Orchestrates the Audio-First Multi-Agent rendering flow with Parallelism:
  1. (Sequential) Agent 1 & Agent 2: Storyboard & Code Generation (Context Chaining)
  2. (Parallel) Manim Rendering of all scenes simultaneously.
  3. (Parallel) Muxing each silent Manim video with its explicit TTS audio padded by frozen frames.
  4. (Sequential) Final concatenation.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
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
# Step 2 — Parallel Rendering & Muxing Helper
# ══════════════════════════════════════════════

def render_and_mux_task(scene_id: str, code: str, narration: str, audio_path: Path, work_dir: Path) -> Path:
    """Renders silent video and muxes it with audio explicitly to prevent desync."""
    scene_work_dir = work_dir / scene_id
    scene_work_dir.mkdir(exist_ok=True)
    
    logger.info(f"Parallel Render Started: {scene_id}")
    silent_video_path = render_manim_scene(
        manim_code=code,
        scene_class_name="ExplainerScene",
        work_dir=scene_work_dir,
        quality="l",
        narration_text=narration,
    )

    # Now we mux it using ffmpeg, padding video if it is too short!
    synced_video_path = scene_work_dir / f"{scene_id}_synced.mp4"
    logger.info(f"Muxing Audio/Video safely for: {scene_id}")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(silent_video_path),
        "-i", str(audio_path),
        "-filter_complex", "[0:v]tpad=stop_mode=clone:stop_duration=99[v]",
        "-map", "[v]",
        "-map", "1:a",
        "-c:a", "copy",
        "-shortest",
        str(synced_video_path)
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
    return synced_video_path


# ══════════════════════════════════════════════
# Step 3 — Post-Processing
# ══════════════════════════════════════════════

def concatenate_scenes(scene_videos: list[Path], job_id: str, work_dir: Path) -> Path:
    """Concatenate individual synced scene MP4s into one final video."""
    concat_file = work_dir / "concat_list.txt"
    with open(concat_file, "w") as f:
        for video_path in scene_videos:
            f.write(f"file '{video_path}'\n")

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
    Optimized Parallel Pipeline:
    1. (Sequential) Generate all TTS and LLM Code first.
    2. (Parallel) Spin up threads to render Manim clips AND mux them safely.
    3. (Sequential) Merge and Cleanup.
    """
    settings.validate()

    work_dir = Path(tempfile.mkdtemp(prefix=f"sceneflow_manim_{job_id}_"))
    logger.info("Manim Parallel Pipeline started for job %s", job_id)

    bg_color = "#0F172A"
    if draft.style_config.brand_colors:
        bg_color = draft.style_config.brand_colors[0]

    # --- Phase 1: Sequential Generation (Requires Context) ---
    generation_results = []
    previous_code = ""

    for scene in draft.scenes:
        # Rate limit protection for Gemini API
        if len(generation_results) > 0:
            time.sleep(2)

        # 1. TTS
        tts_path, tts_duration = generate_audio_step(scene, work_dir)

        # 2. LLM Code
        logger.info(f"Generating Code for {scene.scene_id} ({tts_duration:.2f}s)")
        manim_code = generate_scene_manim_code(
            scene_plan=scene,
            bg_color=bg_color,
            audio_duration=tts_duration,
            audio_path=str(tts_path),
            previous_code=previous_code
        )
        previous_code = manim_code
        
        generation_results.append({
            "id": scene.scene_id,
            "code": manim_code,
            "narration": scene.narration_text,
            "audio_path": tts_path
        })

    # --- Phase 2: Parallel Rendering & Muxing ---
    logger.info(f"Starting Parallel Rendering & Muxing for {len(generation_results)} scenes...")
    
    scene_videos: list[Path] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(render_and_mux_task, res["id"], res["code"], res["narration"], res["audio_path"], work_dir)
            for res in generation_results
        ]
        scene_videos = [f.result() for f in futures]

    # --- Phase 3: Finalize ---
    final_video = concatenate_scenes(scene_videos, job_id, work_dir)
    final_dest = cleanup(work_dir, final_video, Path(settings.OUTPUT_DIR), job_id)

    logger.info("Parallel pipeline completed for job %s", job_id)
    return str(final_dest)
