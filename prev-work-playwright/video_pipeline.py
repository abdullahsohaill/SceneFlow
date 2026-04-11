"""
SceneFlow API — Video Rendering Pipeline
==========================================
Orchestrates the full rendering flow:
  1. TTS audio generation  (edge-tts)
  2. HTML template rendering (Jinja2)
  3. Frame capture          (Playwright headless Chromium)
  4. Scene assembly          (ffmpeg-python: image + audio → mp4)
  5. Final concatenation     (ffmpeg concat demuxer)
  6. Cleanup                 (remove intermediates)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import ffmpeg
from jinja2 import Environment, FileSystemLoader
from mutagen.mp3 import MP3
from playwright.sync_api import sync_playwright

from .config import settings
from .schemas import Scene, StyleConfig, VideoDraft

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Resolve the templates directory (sibling to this file)
# ──────────────────────────────────────────────
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
jinja_env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))


# ══════════════════════════════════════════════
# Step 1 — Audio Generation
# ══════════════════════════════════════════════

def generate_audio(scene: Scene, work_dir: Path) -> tuple[Path, float]:
    """
    Call edge-tts CLI to generate narration audio for a single scene.
    This is completely free and requires no API keys.

    Returns
    -------
    (audio_path, duration_seconds)
    """
    audio_path = work_dir / f"{scene.scene_id}.mp3"

    logger.info("Generating TTS for %s (%d chars)", scene.scene_id, len(scene.narration_text))

    # We use subprocess to call the edge-tts CLI tool directly.
    # It's robust, async-safe, and very reliable.
    cmd = [
        "edge-tts",
        "--voice", settings.TTS_VOICE,
        "--text", scene.narration_text,
        "--write-media", str(audio_path)
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)

    # Read exact duration using mutagen
    audio_info = MP3(str(audio_path))
    duration = audio_info.info.length  # seconds (float)

    logger.info("Audio for %s: %.2fs → %s", scene.scene_id, duration, audio_path)
    return audio_path, duration


# ══════════════════════════════════════════════
# Step 2 — HTML Template Rendering
# ══════════════════════════════════════════════

def render_scene_html(scene: Scene, style_config: StyleConfig, work_dir: Path) -> Path:
    """
    Render a scene's visual elements into a standalone HTML file
    using a Jinja2 template and the brand style config.
    """
    template = jinja_env.get_template("base_scene.html")

    html_content = template.render(
        scene=scene,
        style=style_config,
    )

    html_path = work_dir / f"{scene.scene_id}.html"
    html_path.write_text(html_content, encoding="utf-8")

    logger.info("Rendered HTML for %s → %s", scene.scene_id, html_path)
    return html_path


# ══════════════════════════════════════════════
# Step 3 — Frame Capture (Playwright)
# ══════════════════════════════════════════════

def capture_frame(html_path: Path, work_dir: Path, scene_id: str) -> Path:
    """
    Open the HTML file in a headless Chromium browser at 1920×1080
    and take a screenshot.
    """
    screenshot_path = work_dir / f"{scene_id}.png"

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})

        # Navigate to the local HTML file
        page.goto(f"file://{html_path.resolve()}")

        # Small wait to let any CSS transitions settle
        page.wait_for_timeout(500)

        page.screenshot(path=str(screenshot_path), full_page=False)
        browser.close()

    logger.info("Captured frame for %s → %s", scene_id, screenshot_path)
    return screenshot_path


# ══════════════════════════════════════════════
# Step 4 — Scene Assembly (image + audio → mp4)
# ══════════════════════════════════════════════

def assemble_scene(scene_id: str, duration: float, work_dir: Path) -> Path:
    """
    Combine a static PNG frame and an MP3 audio track into a single
    scene MP4 using ffmpeg-python.

    The image is looped for the exact duration of the audio.
    """
    image_path = work_dir / f"{scene_id}.png"
    audio_path = work_dir / f"{scene_id}.mp3"
    output_path = work_dir / f"{scene_id}.mp4"

    logger.info("Assembling scene %s (%.2fs)", scene_id, duration)

    frames = int(duration * 25)
    video_input = (
        ffmpeg.input(str(image_path), loop=1, framerate=25)
        .filter("zoompan", z="min(zoom+0.0015,1.5)", d=frames, x="iw/2-(iw/zoom/2)", y="ih/2-(ih/zoom/2)", s="1920x1080")
    )
    audio_input = ffmpeg.input(str(audio_path))

    (
        ffmpeg
        .output(
            video_input,
            audio_input,
            str(output_path),
            vcodec="libx264",
            acodec="aac",
            audio_bitrate="192k",
            pix_fmt="yuv420p",
            shortest=None,
            t=duration,
        )
        .overwrite_output()
        .run(quiet=True)
    )

    logger.info("Assembled %s → %s", scene_id, output_path)
    return output_path


# ══════════════════════════════════════════════
# Step 5 — Final Concatenation
# ══════════════════════════════════════════════

def concatenate_scenes(scene_ids: list[str], job_id: str, work_dir: Path) -> Path:
    """
    Concatenate individual scene MP4s into a single final video
    using the FFmpeg concat demuxer.
    """
    concat_file = work_dir / "concat_list.txt"
    with open(concat_file, "w") as f:
        for sid in scene_ids:
            mp4_path = work_dir / f"{sid}.mp4"
            f.write(f"file '{mp4_path}'\n")

    output_path = work_dir / f"final_video_{job_id}.mp4"

    logger.info("Concatenating %d scenes → %s", len(scene_ids), output_path)

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
# Step 6 — Cleanup
# ══════════════════════════════════════════════

def cleanup(work_dir: Path, final_video: Path, output_dir: Path, job_id: str) -> Path:
    """
    Move the final video to the output directory and remove
    all intermediate files (HTML, PNG, MP3, individual MP4s).
    """
    job_output_dir = output_dir / job_id
    job_output_dir.mkdir(parents=True, exist_ok=True)

    final_dest = job_output_dir / final_video.name
    shutil.move(str(final_video), str(final_dest))

    shutil.rmtree(str(work_dir), ignore_errors=True)

    logger.info("Cleaned up. Final output → %s", final_dest)
    return final_dest


# ══════════════════════════════════════════════
# Orchestrator — run_pipeline
# ══════════════════════════════════════════════

def run_pipeline(draft: VideoDraft, job_id: str) -> str:
    """
    Full rendering pipeline orchestrator.
    """
    settings.validate()

    work_dir = Path(tempfile.mkdtemp(prefix=f"sceneflow_{job_id}_"))
    logger.info("Pipeline started for job %s in %s", job_id, work_dir)

    scene_ids: list[str] = []

    for scene in draft.scenes:
        _audio_path, duration = generate_audio(scene, work_dir)
        scene.estimated_duration = duration

        html_path = render_scene_html(scene, draft.style_config, work_dir)
        capture_frame(html_path, work_dir, scene.scene_id)
        assemble_scene(scene.scene_id, duration, work_dir)

        scene_ids.append(scene.scene_id)

    final_video = concatenate_scenes(scene_ids, job_id, work_dir)
    final_dest = cleanup(work_dir, final_video, settings.OUTPUT_DIR, job_id)

    logger.info("Pipeline completed for job %s → %s", job_id, final_dest)
    return str(final_dest)
