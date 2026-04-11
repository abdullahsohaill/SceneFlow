"""
SceneFlow API — Manim Renderer
================================
Executes LLM-generated Manim Python code to produce animated .mp4 videos.
If Manim fails, attempts to fix the code via LLM retry, and falls back to
a simple title-card scene as a last resort.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Fallback: generate a simple title-card scene
# ──────────────────────────────────────────────

FALLBACK_TEMPLATE = '''from manim import *

class ExplainerScene(Scene):
    def construct(self):
        self.camera.background_color = "#0F172A"

        title = Text("{title}", font_size=36, color=BLUE)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.scale(0.7).to_edge(UP))

        body = Text("{body}", font_size=24, color=WHITE, line_spacing=1.2)
        body.next_to(title, DOWN, buff=0.8)
        self.play(FadeIn(body))
        self.wait(3)
'''


def _write_and_render(code: str, scene_class_name: str, work_dir: Path, quality: str) -> Path:
    """Write code to file and invoke Manim CLI. Returns path to .mp4 or raises RuntimeError."""
    script_path = work_dir / f"{scene_class_name}.py"

    # Gemini JSON encodes newlines as literal \n in strings — unescape them
    clean_code = code.replace("\\n", "\n").replace("\\t", "\t")
    script_path.write_text(clean_code, encoding="utf-8")

    logger.info("Wrote Manim script to %s (%d lines)", script_path, clean_code.count("\n"))

    cmd = [
        "manim", "render",
        f"-q{quality}",
        "--format", "mp4",
        "--media_dir", str(work_dir / "media"),
        str(script_path),
        scene_class_name,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(work_dir),
    )

    if result.returncode != 0:
        error_msg = result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr
        raise RuntimeError(error_msg)

    media_dir = work_dir / "media" / "videos" / scene_class_name
    mp4_files = list(media_dir.rglob("*.mp4"))

    if not mp4_files:
        raise RuntimeError(f"No .mp4 found in {media_dir}")

    return mp4_files[0]


def _attempt_llm_fix(original_code: str, error_msg: str) -> str:
    """Ask Gemini to fix the broken Manim code."""
    try:
        from google import genai
        from google.genai import types
        from .config import settings

        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        prompt = f"""The following Manim Python code failed to render with this error:

ERROR:
{error_msg[:1000]}

ORIGINAL CODE:
{original_code}

Fix the code. Return ONLY the corrected Python code, nothing else.
Rules:
- Keep 'from manim import *' at the top
- Keep the class named 'ExplainerScene'
- Do NOT use SVGMobject, ImageMobject, or ThreeDScene
- Avoid complex numpy operations on Manim color objects
- Use simple shapes: Text, Rectangle, Circle, Arrow, etc.
"""
        response = client.models.generate_content(
            model="gemini-3-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3),
        )
        fixed = response.text.strip()
        # Strip markdown fences
        if fixed.startswith("```python"):
            fixed = fixed[9:]
        elif fixed.startswith("```"):
            fixed = fixed[3:]
        if fixed.endswith("```"):
            fixed = fixed[:-3]
        return fixed.strip()
    except Exception as e:
        logger.warning("LLM fix attempt failed: %s", e)
        return ""


def render_manim_scene(
    manim_code: str,
    scene_class_name: str,
    work_dir: Path,
    quality: str = "l",
    narration_text: str = "",
) -> Path:
    """
    Render LLM-generated Manim code with retry and fallback:
      1. Try the original code
      2. If it fails, ask the LLM to fix it
      3. If that also fails, render a simple fallback title card
    """

    # === Attempt 1: Original code ===
    last_error = ""
    try:
        video = _write_and_render(manim_code, scene_class_name, work_dir, quality)
        logger.info("Manim rendered (attempt 1) → %s", video)
        return video
    except RuntimeError as e:
        last_error = str(e)
        logger.warning("Manim attempt 1 failed: %s", last_error[:200])

    # === Attempt 2: LLM-corrected code ===
    retry_dir = work_dir / "retry"
    retry_dir.mkdir(exist_ok=True)

    fixed_code = _attempt_llm_fix(manim_code, last_error)
    if fixed_code:
        try:
            video = _write_and_render(fixed_code, scene_class_name, retry_dir, quality)
            logger.info("Manim rendered (attempt 2 — LLM fix) → %s", video)
            return video
        except RuntimeError as e2:
            logger.warning("Manim attempt 2 (LLM fix) also failed: %s", str(e2)[:200])

    # === Attempt 3: Fallback title card ===
    fallback_dir = work_dir / "fallback"
    fallback_dir.mkdir(exist_ok=True)

    # Extract a short title from narration
    title = narration_text[:50].replace('"', "'") if narration_text else "Scene"
    body = narration_text[50:120].replace('"', "'") if len(narration_text) > 50 else ""

    fallback_code = FALLBACK_TEMPLATE.format(title=title, body=body)
    try:
        video = _write_and_render(fallback_code, scene_class_name, fallback_dir, quality)
        logger.info("Manim rendered (fallback title card) → %s", video)
        return video
    except RuntimeError as e3:
        raise RuntimeError(f"All 3 render attempts failed. Last error: {e3}")
