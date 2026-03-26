"""
SceneFlow Fine-Tuning — Results Comparison
===========================================
Compares ManiBench evaluation results across multiple models
(e.g., base Qwen2.5-Coder vs fine-tuned version).

Usage:
    python -m evaluate.compare_results --results results/
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def load_results(results_dir: Path) -> dict[str, list[dict]]:
    """Load all result JSON files from a directory, grouped by model."""
    model_results = defaultdict(list)

    for f in sorted(results_dir.glob("manibench_*.json")):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        if data:
            model_name = data[0].get("model", f.stem)
            model_results[model_name].extend(data)

    return dict(model_results)


def compute_aggregates(results: list[dict]) -> dict:
    """Compute aggregate metrics from trial results."""
    if not results:
        return {}

    n = len(results)
    agg = {
        "n_trials": n,
        "executability": sum(r["metrics"]["executability"] for r in results) / n,
        "vcer": sum(r["metrics"]["vcer"] for r in results) / n,
        "alignment": sum(r["metrics"]["alignment"] for r in results) / n,
        "coverage": sum(r["metrics"]["coverage"] for r in results) / n,
    }

    # Per-problem breakdown
    by_problem = defaultdict(list)
    for r in results:
        by_problem[r["problem_id"]].append(r)

    agg["per_problem"] = {}
    for pid, trials in by_problem.items():
        pn = len(trials)
        agg["per_problem"][pid] = {
            "executability": sum(t["metrics"]["executability"] for t in trials) / pn,
            "vcer": sum(t["metrics"]["vcer"] for t in trials) / pn,
            "alignment": sum(t["metrics"]["alignment"] for t in trials) / pn,
            "coverage": sum(t["metrics"]["coverage"] for t in trials) / pn,
        }

    return agg


def print_comparison(all_results: dict[str, list[dict]]):
    """Print side-by-side comparison table."""
    aggregates = {}
    for model, results in all_results.items():
        aggregates[model] = compute_aggregates(results)

    # ── Summary table ──
    table = Table(title="\n📊 Model Comparison (ManiBench)")
    table.add_column("Model", style="cyan", max_width=40)
    table.add_column("Exec ↑", justify="right")
    table.add_column("VCER ↓", justify="right")
    table.add_column("Align ↑", justify="right")
    table.add_column("Cover ↑", justify="right")
    table.add_column("Trials", justify="right")

    for model in sorted(aggregates.keys()):
        agg = aggregates[model]
        # Shorten model name for display
        short_name = Path(model).name if "/" in model else model
        if len(short_name) > 40:
            short_name = short_name[:37] + "..."

        table.add_row(
            short_name,
            f"{agg['executability']:.0%}",
            f"{agg['vcer']:.3f}",
            f"{agg['alignment']:.3f}",
            f"{agg['coverage']:.3f}",
            str(agg["n_trials"]),
        )

    console.print(table)

    # ── Delta table (if exactly 2 models) ──
    if len(aggregates) == 2:
        models = sorted(aggregates.keys())
        base, ft = aggregates[models[0]], aggregates[models[1]]

        delta_table = Table(title=f"\n📈 Improvement: {Path(models[1]).name} vs {Path(models[0]).name}")
        delta_table.add_column("Metric", style="cyan")
        delta_table.add_column("Base", justify="right")
        delta_table.add_column("Fine-tuned", justify="right")
        delta_table.add_column("Δ", justify="right")

        for metric in ["executability", "vcer", "alignment", "coverage"]:
            b_val = base[metric]
            f_val = ft[metric]
            delta = f_val - b_val

            # Color delta: green if improvement, red if regression
            is_better = delta > 0 if metric != "vcer" else delta < 0
            color = "green" if is_better else "red"

            delta_table.add_row(
                metric.title(),
                f"{b_val:.3f}",
                f"{f_val:.3f}",
                f"[{color}]{delta:+.3f}[/]",
            )

        console.print(delta_table)

    # ── Per-problem grid ──
    all_problems = set()
    for agg in aggregates.values():
        all_problems.update(agg.get("per_problem", {}).keys())

    if all_problems:
        grid = Table(title="\n📋 Per-Problem Executability Grid")
        grid.add_column("Problem", style="cyan")
        for model in sorted(aggregates.keys()):
            short = Path(model).name if "/" in model else model
            grid.add_column(short[:20], justify="center")

        for pid in sorted(all_problems):
            row = [pid]
            for model in sorted(aggregates.keys()):
                pp = aggregates[model].get("per_problem", {}).get(pid, {})
                ex = pp.get("executability", 0)
                row.append(f"{'●' if ex > 0 else '○'} {ex:.0%}")
            grid.add_row(*row)

        console.print(grid)


def export_markdown(all_results: dict[str, list[dict]], output_path: Path):
    """Export comparison as Markdown table."""
    aggregates = {
        model: compute_aggregates(results)
        for model, results in all_results.items()
    }

    lines = ["# ManiBench Evaluation Results\n"]
    lines.append("| Model | Executability ↑ | VCER ↓ | Alignment ↑ | Coverage ↑ | Trials |")
    lines.append("|-------|:-:|:-:|:-:|:-:|:-:|")

    for model in sorted(aggregates.keys()):
        agg = aggregates[model]
        short = Path(model).name if "/" in model else model
        lines.append(
            f"| {short} | {agg['executability']:.1%} | {agg['vcer']:.3f} "
            f"| {agg['alignment']:.3f} | {agg['coverage']:.3f} | {agg['n_trials']} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"\n[bold green]📄 Markdown report:[/] {output_path}")


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(description="Compare ManiBench evaluation results")
    parser.add_argument(
        "--results", type=Path, default=Path("results"),
        help="Results directory"
    )
    parser.add_argument(
        "--export-md", type=Path, default=None,
        help="Export Markdown report to file"
    )
    args = parser.parse_args()

    all_results = load_results(args.results)

    if not all_results:
        console.print("[red]No results found in directory[/]")
        return

    console.print(f"[green]Found results for {len(all_results)} model(s)[/]")
    print_comparison(all_results)

    if args.export_md:
        export_markdown(all_results, args.export_md)


if __name__ == "__main__":
    main()
