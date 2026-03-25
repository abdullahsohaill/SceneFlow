"""
SceneFlow Fine-Tuning — ManiBench Evaluation Runner
====================================================
Runs the ManiBench 12-problem benchmark on local models.
Computes all 4 metrics: Executability, VCER, Alignment, Coverage.

Usage:
    python -m evaluate.run_manibench --model outputs/qwen3b-manim-ft-merged --strategy zero_shot --trials 3
    python -m evaluate.run_manibench --model Qwen/Qwen2.5-Coder-3B-Instruct --problems MB-005 --trials 1 --skip-render
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .local_inference import LocalInference

logger = logging.getLogger(__name__)
console = Console()

# ── ManiBench dataset path ──
MANIBENCH_DATASET = Path(__file__).parent.parent / "ManiBench_Pilot_Dataset.json_ft"

# ── Prompt strategies ──

SYSTEM_PROMPTS = {
    "zero_shot": (
        "You are an expert Manim Community Edition (CE) programmer. "
        "Generate complete, executable Manim CE Python code. "
        "Use 'from manim import *'. Return ONLY Python code."
    ),
    "few_shot": (
        "You are an expert Manim Community Edition (CE) programmer. "
        "Here is an example of correct Manim CE code:\n\n"
        "```python\n"
        "from manim import *\n\n"
        "class ExampleScene(Scene):\n"
        "    def construct(self):\n"
        "        circle = Circle(color=BLUE)\n"
        "        self.play(Create(circle))\n"
        "        self.play(circle.animate.shift(RIGHT * 2))\n"
        "        label = MathTex(r'\\pi r^2')\n"
        "        label.next_to(circle, DOWN)\n"
        "        self.play(Write(label))\n"
        "        self.wait(1)\n"
        "```\n\n"
        "Generate complete, executable Manim CE code. Return ONLY Python code."
    ),
    "cot": (
        "You are an expert Manim Community Edition (CE) programmer. "
        "First, analyze what visual elements and animations are needed. "
        "Then write the complete Manim CE Python code. "
        "Use 'from manim import *'. Return ONLY Python code."
    ),
    "constraint": (
        "You are an expert Manim Community Edition (CE) programmer. "
        "Generate Manim CE code following these constraints:\n"
        "- Use 'from manim import *' (NOT manimlib)\n"
        "- Ensure proper timing with self.wait() calls\n"
        "- Use self.play() for all animations\n"
        "- Include proper color coding and labels\n"
        "- Return ONLY executable Python code."
    ),
    "version_aware": (
        "You are an expert Manim Community Edition (CE) programmer. "
        "CRITICAL: Use ONLY Manim CE syntax. The following are FORBIDDEN:\n"
        "- 'from manimlib import *' (use 'from manim import *')\n"
        "- CONFIG dicts (use __init__ parameters)\n"
        "- ShowCreation (use Create)\n"
        "- FadeInFrom (use FadeIn with shift=...)\n"
        "- GraphScene (use Scene with Axes)\n"
        "- InteractiveScene (use Scene)\n"
        "- self.frame (use self.camera.frame)\n"
        "Return ONLY executable Manim CE Python code."
    ),
}


# ── GL Detection Patterns (from ManiBench) ──

GL_DETECTION_PATTERNS = [
    re.compile(r"from\s+manimlib"),
    re.compile(r"import\s+manimlib"),
    re.compile(r"from\s+manim_imports_ext"),
    re.compile(r"\bCONFIG\s*=\s*\{"),
    re.compile(r"\bShowCreation\b"),
    re.compile(r"\bFadeInFrom\b"),
    re.compile(r"\bFadeOutAndShift\b"),
    re.compile(r"\bGraphScene\b"),
    re.compile(r"\bInteractiveScene\b"),
    re.compile(r"\bPiCreature\b"),
    re.compile(r"self\.frame\."),
    re.compile(r"\.apply_depth_test\b"),
    re.compile(r"\.set_shading\b"),
    re.compile(r"\bGlowDot\b"),
    re.compile(r"\bDieFace\b"),
]


# ── Metrics ──

@dataclass
class MetricResult:
    executability: float = 0.0
    vcer: float = 0.0
    alignment: float = 0.0
    coverage: float = 0.0


@dataclass
class TrialResult:
    problem_id: str = ""
    trial: int = 0
    strategy: str = ""
    model: str = ""
    generated_code: str = ""
    metrics: MetricResult = field(default_factory=MetricResult)
    error: str = ""
    duration_s: float = 0.0


def compute_executability(code: str, skip_render: bool = False) -> tuple[float, str]:
    """
    Check if code is executable.
    1. AST parse check
    2. (Optional) Sandboxed Manim render
    """
    # Step 1: AST parse
    try:
        ast.parse(code)
    except SyntaxError as e:
        return 0.0, f"SyntaxError: {e}"

    # Check for Scene subclass
    has_scene = bool(re.search(r"class\s+\w+\(.*Scene.*\)", code))
    if not has_scene:
        return 0.0, "No Scene subclass found"

    # Check for construct method
    has_construct = bool(re.search(r"def\s+construct\s*\(\s*self\s*\)", code))
    if not has_construct:
        return 0.0, "No construct() method found"

    if skip_render:
        # Static analysis only — passes syntax + structure checks
        return 1.0, ""

    # Step 2: Sandboxed Manim render
    try:
        with tempfile.TemporaryDirectory(prefix="manibench_") as tmpdir:
            script_path = Path(tmpdir) / "test_scene.py"
            script_path.write_text(code)

            # Extract scene class name
            match = re.search(r"class\s+(\w+)\(.*Scene.*\)", code)
            scene_name = match.group(1) if match else "Scene"

            cmd = [
                "manim", "render",
                "-ql",  # low quality for speed
                "--format", "mp4",
                "--media_dir", str(Path(tmpdir) / "media"),
                str(script_path),
                scene_name,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=tmpdir,
            )

            if result.returncode == 0:
                return 1.0, ""
            else:
                error = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
                return 0.0, f"Render failed: {error}"

    except subprocess.TimeoutExpired:
        return 0.0, "Render timeout (120s)"
    except Exception as e:
        return 0.0, f"Render error: {e}"


def compute_vcer(code: str) -> float:
    """
    Version-Conflict Error Rate.
    Counts GL-specific patterns found in the code.
    """
    conflicts = sum(1 for p in GL_DETECTION_PATTERNS if p.search(code))
    return conflicts / len(GL_DETECTION_PATTERNS) if conflicts > 0 else 0.0


def compute_alignment(code: str, required_events: list[dict]) -> float:
    """
    Alignment Score: weighted fraction of required visual events
    that are present in the code (keyword-based heuristic).
    """
    if not required_events:
        return 1.0

    total_weight = 0.0
    weighted_matches = 0.0

    for event in required_events:
        weight = event.get("weight", 1.0)
        total_weight += weight

        # Check if event keywords appear in code
        event_id = event.get("event_id", "")
        description = event.get("description", "")

        # Extract keywords from event description
        keywords = _extract_keywords(description)
        if any(kw.lower() in code.lower() for kw in keywords):
            weighted_matches += weight

    return weighted_matches / total_weight if total_weight > 0 else 0.0


def _extract_keywords(description: str) -> list[str]:
    """Extract meaningful keywords from an event description."""
    # Remove common stop words and extract key terms
    stop_words = {
        "the", "a", "an", "is", "are", "and", "or", "to", "in", "of",
        "for", "with", "that", "this", "it", "on", "at", "by", "from",
        "should", "must", "will", "be", "have", "has", "do", "does",
        "show", "display", "create", "animate", "render",
    }
    words = re.findall(r'\b[a-zA-Z]{3,}\b', description)
    return [w for w in words if w.lower() not in stop_words]


def compute_coverage(code: str, coverage_reqs: list[dict] | None = None) -> float:
    """
    Coverage Score: density of pedagogical elements.
    Sub-dimensions: Math, Visual, Numeric, Structural.
    """
    weights = {"math": 0.35, "visual": 0.30, "numeric": 0.20, "structural": 0.15}

    # Math annotation patterns
    math_patterns = [
        r"\bTex\b", r"\bMathTex\b", r"\bText\b", r"\bMathTable\b",
        r"\bDecimalNumber\b", r"\bInteger\b", r"\\\\frac", r"\\\\int",
        r"\\\\sum", r"\\\\lim", r"\\\\pi", r"\\\\theta",
    ]

    # Visual mapping patterns
    visual_patterns = [
        r"\.set_color\b", r"\.set_fill\b", r"\bArrow\b", r"\bDot\b",
        r"\bSurroundingRectangle\b", r"\bBrace\b", r"color\s*=",
        r"\bLine\b", r"\bDashedLine\b", r"\bCurvedArrow\b",
    ]

    # Numeric evidence patterns
    numeric_patterns = [
        r"\bDecimalNumber\b", r"\bInteger\b", r"\bValueTracker\b",
        r"\bNumberLine\b", r"\bAxes\b", r"\.plot\b", r"\bNumberPlane\b",
        r"\bBarChart\b",
    ]

    # Structural clarity patterns
    structural_patterns = [
        r"\bVGroup\b", r"\bGroup\b", r"\.arrange\b", r"\.wait\b",
        r"\bLaggedStart\b", r"\bSuccession\b", r"\bAnimationGroup\b",
        r"def\s+\w+\s*\(self", r"\bself\.play\b",
    ]

    all_pattern_sets = {
        "math": math_patterns,
        "visual": visual_patterns,
        "numeric": numeric_patterns,
        "structural": structural_patterns,
    }

    score = 0.0
    for dim, patterns in all_pattern_sets.items():
        matches = sum(1 for p in patterns if re.search(p, code))
        dim_score = min(1.0, matches / max(1, len(patterns) * 0.4))
        score += weights[dim] * dim_score

    return round(score, 4)


# ── Main Evaluation ──

def load_manibench_dataset(path: Path, problem_ids: list[str] | None = None) -> list[dict]:
    """Load ManiBench pilot dataset."""
    if not path.exists():
        console.print(f"[bold red]ManiBench dataset not found at {path}[/]")
        console.print("[yellow]Download it from: https://github.com/nabin2004/ManiBench[/]")
        console.print("[yellow]Place ManiBench_Pilot_Dataset.json in the finetuning/ directory[/]")
        raise FileNotFoundError(f"ManiBench dataset not found: {path}")

    with open(path) as f:
        data = json.load(f)

    # Handle different dataset formats
    problems = data if isinstance(data, list) else data.get("problems", data.get("benchmark", [data]))

    if problem_ids:
        problems = [p for p in problems if p.get("id", "") in problem_ids]

    return problems


def run_evaluation(
    model_path: str,
    strategy: str = "zero_shot",
    trials: int = 3,
    problem_ids: list[str] | None = None,
    skip_render: bool = False,
    output_dir: Path = Path("results"),
    load_in_8bit: bool = True,
) -> list[TrialResult]:
    """
    Run ManiBench evaluation on a local model.
    """
    console.print(f"\n[bold blue]🔬 ManiBench Evaluation[/]")
    console.print(f"  Model:    {model_path}")
    console.print(f"  Strategy: {strategy}")
    console.print(f"  Trials:   {trials}")
    console.print(f"  Render:   {'skip' if skip_render else 'enabled'}")

    # Load model
    engine = LocalInference(
        model_path=model_path,
        load_in_8bit=load_in_8bit,
    )

    # Load dataset
    problems = load_manibench_dataset(MANIBENCH_DATASET, problem_ids)
    console.print(f"  Problems: {len(problems)}")

    system_prompt = SYSTEM_PROMPTS.get(strategy, SYSTEM_PROMPTS["zero_shot"])
    results: list[TrialResult] = []

    for problem in problems:
        pid = problem.get("id", "unknown")
        prompt = problem.get("full_prompt", "")
        required_events = problem.get("required_visual_events", [])
        coverage_reqs = problem.get("coverage_requirements", [])

        if not prompt:
            console.print(f"[yellow]⚠ No prompt for {pid}, skipping[/]")
            continue

        for trial in range(1, trials + 1):
            console.print(f"\n[cyan]▶ {pid} — trial {trial}/{trials}[/]")

            t0 = time.time()
            try:
                response = engine.generate(prompt, system_prompt=system_prompt)
                code = engine.extract_code(response)
            except Exception as e:
                results.append(TrialResult(
                    problem_id=pid, trial=trial, strategy=strategy,
                    model=model_path, error=str(e),
                ))
                console.print(f"  [red]✗ Generation failed: {e}[/]")
                continue

            duration = time.time() - t0

            # Compute metrics
            exec_score, exec_error = compute_executability(code, skip_render)
            vcer_score = compute_vcer(code)
            align_score = compute_alignment(code, required_events)
            cov_score = compute_coverage(code, coverage_reqs)

            result = TrialResult(
                problem_id=pid,
                trial=trial,
                strategy=strategy,
                model=model_path,
                generated_code=code,
                metrics=MetricResult(
                    executability=exec_score,
                    vcer=vcer_score,
                    alignment=align_score,
                    coverage=cov_score,
                ),
                error=exec_error,
                duration_s=duration,
            )
            results.append(result)

            status = "✓" if exec_score > 0 else "✗"
            console.print(
                f"  [{('green' if exec_score > 0 else 'red')}]{status}[/] "
                f"Exec={exec_score:.0%} VCER={vcer_score:.2f} "
                f"Align={align_score:.2f} Cov={cov_score:.3f} "
                f"({duration:.1f}s)"
            )

    # Save results
    _save_results(results, model_path, strategy, output_dir)
    _print_summary(results)

    return results


def _save_results(results: list[TrialResult], model: str, strategy: str, output_dir: Path):
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(model).name if "/" not in model else model.replace("/", "_")

    result_path = output_dir / f"manibench_{model_name}_{strategy}_{run_id}.json"

    serializable = []
    for r in results:
        d = {
            "problem_id": r.problem_id,
            "trial": r.trial,
            "strategy": r.strategy,
            "model": r.model,
            "metrics": asdict(r.metrics),
            "error": r.error,
            "duration_s": r.duration_s,
            "generated_code": r.generated_code,
        }
        serializable.append(d)

    with open(result_path, "w") as f:
        json.dump(serializable, f, indent=2)

    console.print(f"\n[bold green]💾 Results saved to:[/] {result_path}")


def _print_summary(results: list[TrialResult]):
    """Print aggregate summary table."""
    if not results:
        return

    table = Table(title="\nManiBench Results Summary")
    table.add_column("Problem", style="cyan")
    table.add_column("Exec ↑", justify="right")
    table.add_column("VCER ↓", justify="right")
    table.add_column("Align ↑", justify="right")
    table.add_column("Cover ↑", justify="right")
    table.add_column("Trials", justify="right")

    # Group by problem
    from collections import defaultdict
    by_problem = defaultdict(list)
    for r in results:
        by_problem[r.problem_id].append(r)

    total_exec, total_vcer, total_align, total_cov, total_n = 0, 0, 0, 0, 0

    for pid in sorted(by_problem.keys()):
        trials = by_problem[pid]
        n = len(trials)
        avg_exec = sum(t.metrics.executability for t in trials) / n
        avg_vcer = sum(t.metrics.vcer for t in trials) / n
        avg_align = sum(t.metrics.alignment for t in trials) / n
        avg_cov = sum(t.metrics.coverage for t in trials) / n

        total_exec += avg_exec
        total_vcer += avg_vcer
        total_align += avg_align
        total_cov += avg_cov
        total_n += 1

        table.add_row(
            pid,
            f"{avg_exec:.0%}",
            f"{avg_vcer:.3f}",
            f"{avg_align:.3f}",
            f"{avg_cov:.3f}",
            str(n),
        )

    if total_n > 0:
        table.add_row(
            "[bold]Average",
            f"[bold]{total_exec / total_n:.0%}",
            f"[bold]{total_vcer / total_n:.3f}",
            f"[bold]{total_align / total_n:.3f}",
            f"[bold]{total_cov / total_n:.3f}",
            f"[bold]{sum(len(v) for v in by_problem.values())}",
        )

    console.print(table)


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(description="Run ManiBench evaluation on local models")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model path (HuggingFace name or local path)"
    )
    parser.add_argument(
        "--strategy", type=str, default="zero_shot",
        choices=list(SYSTEM_PROMPTS.keys()),
        help="Prompting strategy"
    )
    parser.add_argument(
        "--trials", type=int, default=3,
        help="Number of trials per problem"
    )
    parser.add_argument(
        "--problems", type=str, nargs="*", default=None,
        help="Specific problem IDs (e.g., MB-001 MB-005)"
    )
    parser.add_argument(
        "--skip-render", action="store_true",
        help="Skip Manim rendering (static analysis only)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--no-quantize", action="store_true",
        help="Load model in full precision (for small models or testing)"
    )
    args = parser.parse_args()

    run_evaluation(
        model_path=args.model,
        strategy=args.strategy,
        trials=args.trials,
        problem_ids=args.problems,
        skip_render=args.skip_render,
        output_dir=args.output,
        load_in_8bit=not args.no_quantize,
    )


if __name__ == "__main__":
    main()
