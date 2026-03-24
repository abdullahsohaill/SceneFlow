"""
SceneFlow API — Pydantic Schemas (The Scene DSL)
=================================================
These models define the structured JSON "Scene Graph" that sits at the heart
of SceneFlow.  The LLM generates a VideoDraft, the client can edit it, and
the rendering pipeline consumes it to produce the final .mp4.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
# Visual Element Types
# ──────────────────────────────────────────────────────────────

class TitleContent(BaseModel):
    text: str

class BulletContent(BaseModel):
    items: list[str]

class IconItem(BaseModel):
    label: str
    emoji: str

class IconDiagramContent(BaseModel):
    icons: list[IconItem]

class CodeBlockContent(BaseModel):
    language: str
    code: str

class FlowchartContent(BaseModel):
    steps: list[str]

class VisualElementTitle(BaseModel):
    type: str = Field(pattern="^title_text$")
    content: TitleContent

class VisualElementBullets(BaseModel):
    type: str = Field(pattern="^bullet_points$")
    content: BulletContent

class VisualElementIcons(BaseModel):
    type: str = Field(pattern="^icon_diagram$")
    content: IconDiagramContent

class VisualElementCode(BaseModel):
    type: str = Field(pattern="^code_block$")
    content: CodeBlockContent

class VisualElementFlowchart(BaseModel):
    type: str = Field(pattern="^flowchart$")
    content: FlowchartContent

from typing import Union
VisualElement = Union[
    VisualElementTitle, 
    VisualElementBullets, 
    VisualElementIcons, 
    VisualElementCode, 
    VisualElementFlowchart
]


class Scene(BaseModel):
    """
    One logical section of the explainer video.
    Contains narration text (for TTS) and a list of visual elements to render.
    """
    scene_id: str = Field(
        ...,
        description="Unique identifier for this scene, e.g. 'scene_1'."
    )
    narration_text: str = Field(
        ...,
        description="The full narration script the TTS engine will speak."
    )
    visuals: list[VisualElement] = Field(
        ...,
        description="Ordered list of visual elements displayed during this scene."
    )
    estimated_duration: float = Field(
        default=5.0,
        description="Estimated duration in seconds (overwritten by actual TTS audio length)."
    )


class StyleConfig(BaseModel):
    """Brand-level styling applied globally across all scenes."""
    brand_colors: list[str] = Field(
        default=["#4F46E5", "#10B981", "#F59E0B"],
        description="List of hex color codes for the brand palette."
    )
    font_family: str = Field(
        default="Inter, sans-serif",
        description="CSS font-family string."
    )
    logo_url: Optional[str] = Field(
        default=None,
        description="Optional URL to the brand logo (displayed in corner)."
    )
    background_color: str = Field(
        default="#0F172A",
        description="Background hex color for all scenes."
    )


class VideoDraft(BaseModel):
    """
    The complete Scene DSL document.
    This is what the LLM generates and what the rendering pipeline consumes.
    """
    scenes: list[Scene] = Field(
        ...,
        description="Ordered list of scenes making up the video."
    )
    style_config: StyleConfig = Field(
        default_factory=StyleConfig,
        description="Global brand styling applied to every scene."
    )


# ──────────────────────────────────────────────────────────────
# API Request / Response Models
# ──────────────────────────────────────────────────────────────

class DraftRequest(BaseModel):
    """POST /api/v1/draft — request body."""
    topic: str = Field(
        ...,
        description="The topic or concept to explain.",
        examples=["How REST APIs work"]
    )
    audience: str = Field(
        default="general",
        description="Target audience (e.g. 'beginners', 'developers', 'executives').",
        examples=["beginners"]
    )
    brand_colors: list[str] = Field(
        default=["#4F46E5", "#10B981", "#F59E0B"],
        description="Brand palette as hex codes.",
        examples=[["#4F46E5", "#10B981"]]
    )
    font_family: str = Field(
        default="Inter, sans-serif",
        description="CSS font-family for the video."
    )
    logo_url: Optional[str] = Field(
        default=None,
        description="Optional brand logo URL."
    )
    num_scenes: int = Field(
        default=4,
        ge=2,
        le=8,
        description="Number of scenes to generate (2–8)."
    )


class DraftResponse(BaseModel):
    """POST /api/v1/draft — response body."""
    draft: VideoDraft
    message: str = "Draft generated successfully. Edit and POST to /api/v1/render."


class RenderRequest(BaseModel):
    """POST /api/v1/render — request body (the possibly-edited draft)."""
    draft: VideoDraft


class RenderResponse(BaseModel):
    """POST /api/v1/render — response body."""
    job_id: str
    message: str = "Rendering job queued."


class JobStatusResponse(BaseModel):
    """GET /api/v1/jobs/{job_id} — response body."""
    job_id: str
    status: str = Field(
        ...,
        description="One of: queued, processing, completed, failed."
    )
    result_url: Optional[str] = Field(
        default=None,
        description="Path/URL to the final .mp4 when status is 'completed'."
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if status is 'failed'."
    )


# ──────────────────────────────────────────────────────────────
# Manim Pipeline Models
# ──────────────────────────────────────────────────────────────

class ManimScene(BaseModel):
    """One animation scene — narration for TTS + Manim Python code."""
    scene_id: str = Field(..., description="e.g. 'scene_1'")
    narration_text: str = Field(..., description="Spoken narration for TTS.")
    manim_code: str = Field(..., description="Complete Manim Python source code for this scene.")
    scene_class_name: str = Field(
        default="ExplainerScene",
        description="Name of the Scene subclass in the code."
    )


class ManimDraft(BaseModel):
    """The full draft containing multiple animated Manim scenes."""
    scenes: list[ManimScene]
    style_config: StyleConfig = Field(default_factory=StyleConfig)


class ManimDraftResponse(BaseModel):
    """Response for the Manim draft endpoint."""
    draft: ManimDraft
    message: str = "Manim draft generated. POST to /api/v1/render-manim to render."


class ManimRenderRequest(BaseModel):
    """Request body for the Manim render endpoint."""
    draft: ManimDraft
