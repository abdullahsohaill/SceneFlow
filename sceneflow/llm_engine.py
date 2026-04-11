"""
SceneFlow API — LLM Engine
===========================
Uses Google's Gemini SDK with Structured Outputs (Pydantic response_schema)
to convert a topic into a validated VideoDraft Scene DSL.
"""

from google import genai
from google.genai import types

from .config import settings
from .schemas import VideoDraft, DraftRequest


# ──────────────────────────────────────────────────────────────
# System prompt — instructs the LLM how to decompose a topic
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are SceneFlow, an expert at turning any topic into a structured explainer video script.

Your job:
1. Break the topic into {num_scenes} logical scenes, each with a unique scene_id
   (e.g. "scene_1", "scene_2", ...).
2. For each scene write clear, conversational narration_text (2–4 sentences)
   suitable for text-to-speech.  Avoid markdown or special characters in the
   narration — it will be spoken aloud.
3. For each scene pick ONE primary visual element from these types:
   • "title_text"    — content: {{"text": "..."}}
   • "bullet_points" — content: {{"items": ["...", "...", "..."]}}
   • "icon_diagram"  — content: {{"icons": [{{"label": "...", "emoji": "..."}}]}}
   • "code_block"    — content: {{"language": "...", "code": "..."}}
   • "flowchart"     — content: {{"steps": ["...", "...", "..."]}}
4. Set estimated_duration to a rough guess in seconds (5–15).
5. Use the brand colors, font, and logo provided in the style_config.

Audience: {audience}

Return ONLY valid JSON matching this exact structure:
```json
{{
  "scenes": [
    {{
      "scene_id": "scene_1",
      "narration_text": "clearly written spoken text...",
      "visuals": [
        {{
          "type": "title_text",
          "content": {{"text": "example"}}
        }}
      ],
      "estimated_duration": 5.0
    }}
  ],
  "style_config": {{}}
}}
```
"""


def generate_draft(request: DraftRequest) -> VideoDraft:
    """
    Call Gemini with structured output to produce a VideoDraft.

    Parameters
    ----------
    request : DraftRequest
        Contains topic, audience, brand_colors, and optional style settings.

    Returns
    -------
    VideoDraft
        A validated Pydantic model containing the full scene graph.
    """
    # Validate that we have an API key
    settings.validate()

    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    # Build the system prompt with the requested number of scenes & audience
    system_message = SYSTEM_PROMPT.format(
        num_scenes=request.num_scenes,
        audience=request.audience,
    )

    # User message describes what video to create
    user_message = (
        f"Create an explainer video about: {request.topic}\n\n"
        f"Style config:\n"
        f"  brand_colors: {request.brand_colors}\n"
        f"  font_family: {request.font_family}\n"
        f"  logo_url: {request.logo_url or 'none'}\n"
    )

    # ── Call Gemini with Pydantic structured output ──────────
    response = client.models.generate_content(
        model="gemini-3-flash",
        contents=f"{system_message}\n\nUser Request:\n{user_message}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VideoDraft,
            temperature=0.7,
        ),
    )

    json_text = response.text.strip()
    if json_text.startswith("```json"):
        json_text = json_text[7:]
    if json_text.endswith("```"):
        json_text = json_text[:-3]

    # Extract the parsed Pydantic object by validating the JSON response text
    draft = VideoDraft.model_validate_json(json_text.strip())

    # Inject the brand style config from the request (LLM might not echo it)
    draft.style_config.brand_colors = request.brand_colors
    draft.style_config.font_family = request.font_family
    if request.logo_url:
        draft.style_config.logo_url = request.logo_url

    return draft


# ──────────────────────────────────────────────────────────────
# Manim Code Generation
# ──────────────────────────────────────────────────────────────

MANIM_SYSTEM_PROMPT = """\
You are SceneFlow-Manim, an expert at creating educational Manim Community animations.

Your job: Given a topic and audience, produce {num_scenes} separate Manim scenes.
Each scene has narration_text (for TTS) and a complete, self-contained Python file
using the Manim Community library.

CRITICAL RULES for the Manim code:
1. Each scene MUST be a complete, self-contained Python file.
2. Always start with: from manim import *
3. Define exactly ONE Scene subclass named "ExplainerScene".
4. Use ONLY these well-tested Manim features:
   - Text("..."), MathTex("..."), Tex("...")
   - Rectangle, Circle, Square, Arrow, Line, Dot
   - VGroup for grouping objects
   - FadeIn, FadeOut, Write, Create, Transform, ReplacementTransform
   - self.play(...) for animations
   - self.wait(1) for pauses
   - .shift(), .scale(), .set_color(), .next_to(), .move_to()
   - UP, DOWN, LEFT, RIGHT, ORIGIN
   - color constants: BLUE, GREEN, RED, YELLOW, WHITE, PURPLE, ORANGE
5. DO NOT use: SVGMobject, ImageMobject, ThreeDScene, external files, internet.
6. Keep each scene focused: 3-6 animation steps, lasting about 10-20 seconds.
7. Make animations visually interesting: use color, movement, transforms.
8. Use self.camera.background_color = "{bg_color}" at the start of construct().

EXAMPLE of a valid Manim scene:
```python
from manim import *

class ExplainerScene(Scene):
    def construct(self):
        self.camera.background_color = "#0F172A"
        
        title = Text("How APIs Work", font_size=48, color=BLUE)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.scale(0.6).to_edge(UP))
        
        client = Rectangle(width=2, height=1, color=GREEN).shift(LEFT * 3)
        client_label = Text("Client", font_size=24).move_to(client)
        server = Rectangle(width=2, height=1, color=BLUE).shift(RIGHT * 3)
        server_label = Text("Server", font_size=24).move_to(server)
        
        self.play(FadeIn(client), FadeIn(client_label), FadeIn(server), FadeIn(server_label))
        
        arrow = Arrow(client.get_right(), server.get_left(), color=YELLOW)
        req_label = Text("Request", font_size=20, color=YELLOW).next_to(arrow, UP)
        self.play(Create(arrow), FadeIn(req_label))
        self.wait(1)
        
        resp_arrow = Arrow(server.get_left(), client.get_right(), color=GREEN).shift(DOWN * 0.5)
        resp_label = Text("Response", font_size=20, color=GREEN).next_to(resp_arrow, DOWN)
        self.play(Create(resp_arrow), FadeIn(resp_label))
        self.wait(2)
```

Return a JSON array of scene objects. Each object has:
- "scene_id": "scene_1", "scene_2", etc.
- "narration_text": conversational text for TTS (2-4 sentences, plain English, no markdown)
- "manim_code": the COMPLETE Python file as a string
- "scene_class_name": always "ExplainerScene"

Audience: {audience}

Return ONLY valid JSON like:
```json
[
  {{
    "scene_id": "scene_1",
    "narration_text": "spoken narration...",
    "manim_code": "from manim import *\\n\\nclass ExplainerScene(Scene):\\n    def construct(self):\\n        ...",
    "scene_class_name": "ExplainerScene"
  }}
]
```
"""


def generate_manim_scenes(request: DraftRequest) -> "ManimDraft":
    """
    Call Gemini to generate Manim Python code for each scene.

    Returns a ManimDraft with scene narration + Manim source code.
    """
    from .schemas import ManimDraft, ManimScene

    settings.validate()
    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    bg_color = "#0F172A"
    if request.brand_colors:
        bg_color = request.brand_colors[0]

    system_message = MANIM_SYSTEM_PROMPT.format(
        num_scenes=request.num_scenes,
        audience=request.audience,
        bg_color=bg_color,
    )

    user_message = (
        f"Create an animated explainer video about: {request.topic}\n\n"
        f"Brand colors: {request.brand_colors}\n"
        f"Number of scenes: {request.num_scenes}\n"
    )

    response = client.models.generate_content(
        model="gemini-3-flash",
        contents=f"{system_message}\n\nUser Request:\n{user_message}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.7,
        ),
    )

    json_text = response.text.strip()
    if json_text.startswith("```json"):
        json_text = json_text[7:]
    if json_text.endswith("```"):
        json_text = json_text[:-3]
    json_text = json_text.strip()

    import json
    scenes_data = json.loads(json_text)

    # Handle both {"scenes": [...]} and bare [...]
    if isinstance(scenes_data, dict) and "scenes" in scenes_data:
        scenes_list = scenes_data["scenes"]
    elif isinstance(scenes_data, list):
        scenes_list = scenes_data
    else:
        raise ValueError(f"Unexpected Gemini response format: {type(scenes_data)}")

    scenes = [ManimScene.model_validate(s) for s in scenes_list]

    return ManimDraft(
        scenes=scenes,
        style_config={
            "brand_colors": request.brand_colors,
            "font_family": request.font_family,
            "logo_url": request.logo_url,
        }
    )
