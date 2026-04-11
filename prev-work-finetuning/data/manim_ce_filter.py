"""
SceneFlow Fine-Tuning — Manim CE Compatibility Filter
=====================================================
Filters Manim code examples to identify CE-compatible samples.
Uses AST parsing + regex patterns based on ManiBench's 145
documented GL→CE incompatibilities.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from enum import Enum


class Compatibility(Enum):
    CE_COMPATIBLE = "ce_compatible"
    GL_ONLY = "gl_only"
    NEEDS_CONVERSION = "needs_conversion"


@dataclass
class FilterResult:
    compatibility: Compatibility
    issues: list[str] = field(default_factory=list)
    score: float = 1.0  # 1.0 = fully compatible, 0.0 = fully incompatible


# ── GL-specific import patterns ──

GL_IMPORT_PATTERNS = [
    re.compile(r"from\s+manimlib"),
    re.compile(r"from\s+manim_imports_ext"),
    re.compile(r"import\s+manimlib"),
]

# ── Deprecated / GL-only class names ──

GL_ONLY_CLASSES = {
    "InteractiveScene",
    "GraphScene",
    "ThreeDScene",  # limited CE support
    "PiCreature",
    "Mortimer",
    "Randolph",
    "GlowDot",
    "DieFace",
}

# ── Deprecated / renamed animation names ──

GL_DEPRECATED_ANIMATIONS = {
    "ShowCreation": "Create",
    "FadeInFrom": "FadeIn",
    "FadeOutAndShift": "FadeOut",
    "FadeInFromDown": "FadeIn",
    "FadeInFromLarge": "FadeIn",
    "GrowArrow": "GrowFromPoint",
    "DrawBorderThenFill": "Create",
}

# ── CONFIG dict pattern (GL-style class config) ──

CONFIG_PATTERN = re.compile(
    r"^\s+CONFIG\s*=\s*\{", re.MULTILINE
)

# ── GL-specific method/attribute patterns ──

GL_METHOD_PATTERNS = [
    re.compile(r"\.apply_depth_test\s*\("),
    re.compile(r"\.set_shading\s*\("),
    re.compile(r"self\.frame\.reorient\s*\("),
    re.compile(r"self\.frame\.animate"),
    re.compile(r"\.get_shader_data\s*\("),
    re.compile(r"\.refresh_shader_data\s*\("),
    re.compile(r"Point3D\s*\("),
    re.compile(r"SurfaceMesh\s*\("),
]

# ── CE-specific positive indicators ──

CE_POSITIVE_PATTERNS = [
    re.compile(r"from\s+manim\s+import"),
    re.compile(r"class\s+\w+\(Scene\)"),
    re.compile(r"def\s+construct\s*\(\s*self\s*\)"),
]


def _check_imports(code: str) -> list[str]:
    """Check for GL-specific imports."""
    issues = []
    for pattern in GL_IMPORT_PATTERNS:
        if pattern.search(code):
            issues.append(f"GL import: {pattern.pattern}")
    return issues


def _check_ast(code: str) -> list[str]:
    """AST-based checks for GL-only class usage and CONFIG dicts."""
    issues = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        issues.append("SyntaxError: cannot parse code")
        return issues

    for node in ast.walk(tree):
        # Check class bases for GL-only classes
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = ""
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name in GL_ONLY_CLASSES:
                    issues.append(f"GL-only class: {base_name}")

        # Check for CONFIG = {} class attribute
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CONFIG":
                    issues.append("CONFIG dict (GL-style class configuration)")

        # Check for deprecated animation calls
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name in GL_DEPRECATED_ANIMATIONS:
                issues.append(
                    f"Deprecated animation: {func_name} → {GL_DEPRECATED_ANIMATIONS[func_name]}"
                )

    return issues


def _check_regex(code: str) -> list[str]:
    """Regex-based checks for GL-specific methods."""
    issues = []
    for pattern in GL_METHOD_PATTERNS:
        if pattern.search(code):
            issues.append(f"GL method: {pattern.pattern}")
    if CONFIG_PATTERN.search(code):
        issues.append("CONFIG dict pattern (regex)")
    return issues


def _has_ce_indicators(code: str) -> bool:
    """Check if code has positive CE indicators."""
    return any(p.search(code) for p in CE_POSITIVE_PATTERNS)


def filter_manim_code(code: str) -> FilterResult:
    """
    Analyze Manim code for CE compatibility.

    Returns a FilterResult with:
    - compatibility: ce_compatible, gl_only, or needs_conversion
    - issues: list of detected problems
    - score: 0.0-1.0 compatibility score
    """
    all_issues: list[str] = []

    all_issues.extend(_check_imports(code))
    all_issues.extend(_check_ast(code))
    all_issues.extend(_check_regex(code))

    has_ce = _has_ce_indicators(code)

    # Scoring: major issues (imports, GL-only classes) = instant GL_ONLY
    # Minor issues (deprecated animations) = needs_conversion
    major_issues = [i for i in all_issues if "GL import" in i or "GL-only class" in i]
    minor_issues = [i for i in all_issues if i not in major_issues]

    if major_issues:
        return FilterResult(
            compatibility=Compatibility.GL_ONLY,
            issues=all_issues,
            score=0.0,
        )
    elif minor_issues and len(minor_issues) > 2:
        return FilterResult(
            compatibility=Compatibility.NEEDS_CONVERSION,
            issues=all_issues,
            score=0.3,
        )
    elif minor_issues:
        # Few minor issues — likely convertible, still usable
        return FilterResult(
            compatibility=Compatibility.NEEDS_CONVERSION,
            issues=all_issues,
            score=0.7,
        )
    elif has_ce:
        return FilterResult(
            compatibility=Compatibility.CE_COMPATIBLE,
            issues=[],
            score=1.0,
        )
    else:
        # No GL issues but no CE indicators either — cautiously compatible
        return FilterResult(
            compatibility=Compatibility.CE_COMPATIBLE,
            issues=["No explicit CE indicators found"],
            score=0.8,
        )
