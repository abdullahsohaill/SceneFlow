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
You are SceneFlow-Animator, an expert Manim Community programmer.

Your job: You are generating a single Python script for ONE specific scene in a larger sequence.

CRITICAL RULES:
1. ALWAYS start with: from manim import *
2. Define exactly ONE Scene subclass named "ExplainerScene".
3. Use self.camera.background_color = "{bg_color}" at the beginning.
4. AUDIO SYNC IS MANDATORY: You must add the audio file like this:
   `self.add_sound(r"{audio_path}")`
   Your animations (`self.play`, `self.wait`) must sum perfectly to the exact audio duration: {audio_duration:.2f} seconds.
5. CONTEXT CHAINING: If Previous Scene Code is provided, you must instantly recreate its final visual state at `run_time=0` (using `.add()`, `.set_opacity()`, etc).
6. TEXT FORMATTING: Extremely important. If you are drawing text on screen, ALWAYS use `font_size=24` or use `Paragraph` to prevent text from overflowing the screen. Keep text short and centered.

Return ONLY pure Python code. Do not output markdown code blocks.
"""

def generate_scene_manim_code(
    scene_plan: DirectorScene, 
    bg_color: str,
    audio_duration: float, 
    audio_path: str,
    previous_code: str = ""
) -> str:
    """Agent 2: Generates Manim Python for a single scene with Audio injection."""
    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    system_message = ANIMATOR_SYSTEM_PROMPT.format(
        bg_color=bg_color,
        audio_duration=audio_duration,
        audio_path=audio_path.replace("\\", "\\\\"),
    )

    user_message = f"""
SCENE TASK:
Render the following scene using Manim.
Narration Being Spoken: {scene_plan.narration_text}
Visuals Requested: {scene_plan.visual_description}

PREVIOUS SCENE CODE (For context chaining):
{previous_code if previous_code else "None. This is the first scene."}
"""

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
