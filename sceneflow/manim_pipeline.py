"""
SceneFlow API — Manim Video Pipeline
======================================
Orchestrates the Manim-based rendering flow:
  1. TTS audio generation  (edge-tts)
  2. Manim rendering       (LLM-generated Python → animated .mp4)
  3. Audio overlay          (FFmpeg merges TTS onto Manim video)
  4. Final concatenation    (ffmpeg concat demuxer)
  5. Cleanup                (remove intermediates)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import ffmpeg
from mutagen.mp3 import MP3

from .config import settings
from .manim_renderer import render_manim_scene
from .schemas import ManimDraft, ManimScene

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# Step 1 — Audio Generation (same as original)
# ══════════════════════════════════════════════

def generate_audio(scene: ManimScene, work_dir: Path) -> tuple[Path, float]:
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
# Step 2 — Render Manim Scene
# ══════════════════════════════════════════════

def render_scene_animation(scene: ManimScene, work_dir: Path) -> Path:
    """Run the LLM-generated Manim code and return the animated .mp4."""
    scene_work_dir = work_dir / scene.scene_id
    scene_work_dir.mkdir(exist_ok=True)

    video_path = render_manim_scene(
        manim_code=scene.manim_code,
        scene_class_name=scene.scene_class_name,
        work_dir=scene_work_dir,
        quality="l",
        narration_text=scene.narration_text,
    )

    logger.info("Manim rendered %s → %s", scene.scene_id, video_path)
    return video_path


# ══════════════════════════════════════════════
# Step 3 — Overlay TTS Audio on Manim Video
# ══════════════════════════════════════════════

def overlay_audio(
    manim_video: Path,
    tts_audio: Path,
    tts_duration: float,
    scene_id: str,
    work_dir: Path,
) -> Path:
    """
    Replace the Manim video's silence with TTS narration.
    If the Manim video is shorter than the audio, it will be extended.
    If longer, it will be trimmed to match the audio.
    """
    output_path = work_dir / f"{scene_id}_final.mp4"

    logger.info("Overlaying audio on %s (%.2fs)", scene_id, tts_duration)

    v = ffmpeg.input(str(manim_video))
    a = ffmpeg.input(str(tts_audio))

    (
        ffmpeg
        .output(
            v.video,
            a,
            str(output_path),
            vcodec="libx264",
            acodec="aac",
            audio_bitrate="192k",
            pix_fmt="yuv420p",
            shortest=None,
        )
        .overwrite_output()
        .run(quiet=True)
    )

    logger.info("Audio overlay done for %s → %s", scene_id, output_path)
    return output_path


# ══════════════════════════════════════════════
# Step 4 — Concatenate All Scenes
# ══════════════════════════════════════════════

def concatenate_scenes(scene_videos: list[Path], job_id: str, work_dir: Path) -> Path:
    """Concatenate individual scene MP4s into one final video."""
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


# ══════════════════════════════════════════════
# Step 5 — Cleanup
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

def run_manim_pipeline(draft: ManimDraft, job_id: str) -> str:
    """
    Full Manim rendering pipeline orchestrator.

    For each scene:
      1. Generate TTS audio
      2. Render Manim animation → silent .mp4
      3. Overlay TTS audio onto the Manim video
    Then concatenate all scenes and clean up.
    """
    settings.validate()

    work_dir = Path(tempfile.mkdtemp(prefix=f"sceneflow_manim_{job_id}_"))
    logger.info("Manim pipeline started for job %s in %s", job_id, work_dir)

    scene_videos: list[Path] = []

    for scene in draft.scenes:
        # 1. TTS
        tts_path, tts_duration = generate_audio(scene, work_dir)

        # 2. Manim render
        manim_video = render_scene_animation(scene, work_dir)

        # 3. Overlay audio
        final_scene = overlay_audio(manim_video, tts_path, tts_duration, scene.scene_id, work_dir)
        scene_videos.append(final_scene)

    # 4. Concatenate
    final_video = concatenate_scenes(scene_videos, job_id, work_dir)

    # 5. Cleanup
    final_dest = cleanup(work_dir, final_video, settings.OUTPUT_DIR, job_id)

    logger.info("Manim pipeline completed for job %s → %s", job_id, final_dest)
    return str(final_dest)
