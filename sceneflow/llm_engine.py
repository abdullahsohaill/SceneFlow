"""
SceneFlow API — LLM Engine
===========================
Multi-Agent Orchestration: 
1. Director Agent generates the multi-minute text storyboard.
2. Animator Agent generates Manim code frame-by-frame with audio sync & context passing.
"""

from google import genai
from google.genai import types, errors
import time

from .config import settings
from .schemas import DirectorStoryboard, DirectorScene, DraftRequest

def generate_with_retry(client, model, contents, config=None, retries=3, delay=2):
    """Refined helper to handle Google 503/429 rate limits with backoff."""
    for i in range(retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
        except (errors.ClientError, Exception) as e:
            if i == retries - 1: raise e
            time.sleep(delay * (i + 1))
    return None


# ──────────────────────────────────────────────────────────────
# Agent 1: The Director (Planner)
# ──────────────────────────────────────────────────────────────

DIRECTOR_SYSTEM_PROMPT = """\
You are SceneFlow-Director, an expert educational video planner.

Your job: Given a topic and audience, autonomously analyze the topic and determine the optimal number of scenes. 
IMPORTANT: For now, be extremely concise. Aim for 4 to 7 scenes total. Do not exceed 8 scenes. Focus on the most critical high-level points.
This is for a Manim-animated video.

For EVERY scene you determine is necessary, provide:
1. scene_id: "scene_1", "scene_2", etc.
2. narration_text: The spoken script for the narrator. Avoid markdown or complex symbols. Make it conversational.
3. visual_description: A clear, plain English description of exactly what should be drawn on the screen using shapes, text, lines, etc.
4. estimated_duration: A rough estimate of the duration in seconds.

Audience: {audience}

You must return a JSON object matching the requested array structure perfectly, containing exactly as many scenes as you mathematically deemed necessary.
"""

def generate_director_plan(request: DraftRequest) -> DirectorStoryboard:
    """Agent 1: Generates the conceptual sub-scene plan."""
    settings.validate()
    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    system_message = DIRECTOR_SYSTEM_PROMPT.format(
        audience=request.audience,
    )

    user_message = (
        f"Create an automated explainer video storyboard about: {request.topic}\n\n"
        f"Brand colors: {request.brand_colors}\n"
    )

    response = generate_with_retry(
        client=client,
        model="gemini-2.5-flash",
        contents=f"{system_message}\n\nUser Request:\n{user_message}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DirectorStoryboard,
            temperature=0.7,
        ),
    )

    json_text = response.text.strip()
    if json_text.startswith("```json"):
        json_text = json_text[7:]
    if json_text.endswith("```"):
        json_text = json_text[:-3]

    draft = DirectorStoryboard.model_validate_json(json_text.strip())
    
    # Inject styling
    draft.style_config.brand_colors = request.brand_colors
    draft.style_config.font_family = request.font_family
    if request.logo_url:
        draft.style_config.logo_url = request.logo_url

    return draft


# ──────────────────────────────────────────────────────────────
# Agent 2: The Animator (Code Generator)
# ──────────────────────────────────────────────────────────────

ANIMATOR_SYSTEM_PROMPT = """\
You are SceneFlow-Animator, an expert Manim Community programmer with exceptional design taste.

Your job: You are generating a SINGLE Python script containing ONE Manim `Scene` subclass that tells the ENTIRE video story natively.

CRITICAL RULES:
1. ALWAYS start with: `from manim import *`
2. Define exactly ONE Scene subclass named `ExplainerScene(Scene)`.
3. Use `self.camera.background_color = "{bg_color}"` at the beginning.
4. AUDIO SYNC IS MANDATORY: For each scene segment, you will be provided an `audio_path` and `audio_duration`.
   You must add the audio file like this exactly: `self.add_sound(r"{{audio_path}}")`
   Your animations (`self.play`, `self.wait`) for that segment MUST sum perfectly to the exact audio duration.
5. CONTINUITY: You are animating sequentially. Do not use `.clear()` needlessly; morph objects smoothly or fade old things out to keep the visual flow continuous.
6. AESTHETICS & DISPLAY: The current visuals are unappealing. You MUST make the video look extremely premium:
   - Do NOT use generic raw shapes. Use beautiful harmonious colors (e.g. pastel Blues, Purples, dynamic combinations).
   - Draw glowing edges, soft backgrounds, or gradients if possible.
   - TEXT FORMATTING: Extremely important. If you are drawing text on screen, ALWAYS use `font="sans-serif"` and clean scaling (`font_size=24` to 36). Avoid overlapping text. If a text string is long, use `Paragraph()` or chop it. Keep text centered, readable, and visually appealing.
   - Add micro-animations (smooth Create, FadeIn, Transform).

Return ONLY pure Python code. Do not output markdown code blocks.
"""

def generate_full_manim_code(
    scenes_data: list[dict], 
    bg_color: str,
) -> str:
    """Agent 2: Generates a monolithic Manim script spanning the entire video sequence."""
    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    system_message = ANIMATOR_SYSTEM_PROMPT.format(
        bg_color=bg_color,
    )

    user_message = "SCENE TIMELINE REQUIREMENTS:\n\n"
    for idx, s in enumerate(scenes_data):
        user_message += f"--- SEGMENT {idx + 1} ---\n"
        user_message += f"Narration Being Spoken: {s['narration']}\n"
        user_message += f"Visuals Requested: {s['visuals']}\n"
        user_message += f"Audio File to use: {s['audio_path'].replace(chr(92), chr(92)+chr(92))}\n"
        user_message += f"Audio Duration (Total wait/play time MUST match this): {s['duration']:.2f} seconds\n\n"

    response = generate_with_retry(
        client=client,
        model="gemini-2.5-flash",
        contents=f"{system_message}\n\nUser Request:\n{user_message}",
        config=types.GenerateContentConfig(temperature=0.2),
    )

    code = response.text.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
        
    return code.strip()

