"""
SceneFlow Fine-Tuning — Dataset Preparation
============================================
Downloads BibbyResearch/3blue1brown-manim from HuggingFace,
filters for Manim CE compatibility, formats as chat-template
JSONL for SFT training.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.table import Table

from .manim_ce_filter import Compatibility, filter_manim_code

logger = logging.getLogger(__name__)
console = Console()

# ── Constants ──

DATASET_NAME = "BibbyResearch/3blue1brown-manim"
DEFAULT_OUTPUT_DIR = Path(__file__).parent
SYSTEM_PROMPT = (
    "You are an expert Manim Community Edition (CE) programmer. "
    "Generate clean, executable Manim CE Python code that creates "
    "mathematical animations. Always use 'from manim import *' and "
    "ensure all code is compatible with the latest Manim CE version. "
    "Return ONLY the Python code, no explanations."
)


def _format_chat_message(prompt: str, code: str) -> dict:
    """Format a single example as a chat-template message."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": code},
        ]
    }


def _extract_prompt_code(example: dict) -> tuple[str, str] | None:
    """
    Extract (prompt, code) from a dataset example.
    The BibbyResearch dataset has varying column names across sources.
    """
    # Try common column name patterns
    prompt_keys = ["prompt", "instruction", "input", "question", "description"]
    code_keys = ["code", "output", "response", "completion", "manim_code", "answer"]

    prompt = None
    code = None

    for k in prompt_keys:
        if k in example and example[k]:
            prompt = str(example[k]).strip()
            break

    for k in code_keys:
        if k in example and example[k]:
            code = str(example[k]).strip()
            break

    if not prompt or not code:
        return None

    # Skip very short or clearly broken examples
    if len(code) < 50 or len(prompt) < 10:
        return None

    return prompt, code


def prepare_dataset(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_examples: int | None = None,
    min_score: float = 0.5,
    val_ratio: float = 0.1,
) -> dict:
    """
    Full dataset preparation pipeline:
    1. Download from HuggingFace
    2. Filter for Manim CE compatibility
    3. Format as chat JSONL
    4. Train/val split
    """
    console.print(f"\n[bold blue]📦 Loading dataset:[/] {DATASET_NAME}")

    try:
        ds = load_dataset(DATASET_NAME, split="train")
    except Exception as e:
        console.print(f"[bold red]Error loading dataset:[/] {e}")
        console.print("[yellow]Trying without split specification...[/]")
        ds = load_dataset(DATASET_NAME)
        if isinstance(ds, dict):
            # Take the first available split
            split_name = list(ds.keys())[0]
            ds = ds[split_name]
            console.print(f"[green]Using split: {split_name}[/]")

    console.print(f"[green]Loaded {len(ds)} raw examples[/]")

    # ── Filter and format ──
    stats = {
        "total_raw": len(ds),
        "parse_failed": 0,
        "ce_compatible": 0,
        "needs_conversion": 0,
        "gl_only": 0,
        "below_threshold": 0,
        "kept": 0,
    }

    formatted = []

    for i, example in enumerate(ds):
        if max_examples and i >= max_examples * 5:
            # Look at 5x max to have enough after filtering
            break

        pair = _extract_prompt_code(example)
        if pair is None:
            stats["parse_failed"] += 1
            continue

        prompt, code = pair
        result = filter_manim_code(code)

        if result.compatibility == Compatibility.GL_ONLY:
            stats["gl_only"] += 1
            continue
        elif result.compatibility == Compatibility.NEEDS_CONVERSION:
            stats["needs_conversion"] += 1
            if result.score < min_score:
                stats["below_threshold"] += 1
                continue

        stats["ce_compatible"] += 1
        formatted.append(_format_chat_message(prompt, code))

        if max_examples and len(formatted) >= max_examples:
            break

    stats["kept"] = len(formatted)

    # ── Train/val split ──
    import random
    random.seed(42)
    random.shuffle(formatted)

    val_size = max(1, int(len(formatted) * val_ratio))
    train_data = formatted[val_size:]
    val_data = formatted[:val_size]

    # ── Write output ──
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl_ft"
    val_path = output_dir / "val.jsonl_ft"

    for path, data in [(train_path, train_data), (val_path, val_data)]:
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    stats["train_size"] = len(train_data)
    stats["val_size"] = len(val_data)

    # ── Report ──
    _print_report(stats, train_path, val_path)

    # Save stats
    stats_path = output_dir / "dataset_stats.json_ft"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def _print_report(stats: dict, train_path: Path, val_path: Path):
    """Print a nice summary table."""
    table = Table(title="Dataset Preparation Report")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green", justify="right")

    table.add_row("Total raw examples", str(stats["total_raw"]))
    table.add_row("Parse failures", str(stats["parse_failed"]))
    table.add_row("GL-only (excluded)", str(stats["gl_only"]))
    table.add_row("Needs conversion", str(stats["needs_conversion"]))
    table.add_row("Below threshold", str(stats["below_threshold"]))
    table.add_row("CE-compatible (kept)", str(stats["ce_compatible"]))
    table.add_row("─" * 20, "─" * 6)
    table.add_row("Train split", str(stats.get("train_size", 0)))
    table.add_row("Validation split", str(stats.get("val_size", 0)))

    console.print(table)
    console.print(f"\n[bold green]✅ Saved:[/] {train_path}")
    console.print(f"[bold green]✅ Saved:[/] {val_path}")


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(description="Prepare Manim SFT dataset")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for JSONL files"
    )
    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="Maximum examples to keep (after filtering)"
    )
    parser.add_argument(
        "--min-score", type=float, default=0.5,
        help="Minimum CE compatibility score (0.0-1.0)"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1,
        help="Validation set ratio"
    )
    args = parser.parse_args()

    prepare_dataset(
        output_dir=args.output,
        max_examples=args.max_examples,
        min_score=args.min_score,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
