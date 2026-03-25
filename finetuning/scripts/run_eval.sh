#!/bin/bash
# ──────────────────────────────────────────
# SceneFlow — ManiBench Evaluation Script
# ──────────────────────────────────────────
# Usage:
#   bash scripts/run_eval.sh <model_path> [strategy] [trials]
#
# Examples:
#   # Evaluate base model (static analysis only)
#   bash scripts/run_eval.sh Qwen/Qwen2.5-Coder-3B-Instruct zero_shot 3 --skip-render
#
#   # Evaluate fine-tuned model (with rendering)
#   bash scripts/run_eval.sh outputs/qwen3b-manim-ft-merged zero_shot 3
#
#   # Compare all results
#   python -m evaluate.compare_results --results results/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

MODEL=${1:?Usage: bash scripts/run_eval.sh <model_path> [strategy] [trials] [extra_args...]}
STRATEGY=${2:-zero_shot}
TRIALS=${3:-3}
shift 3 2>/dev/null || true

echo "╔══════════════════════════════════════╗"
echo "║   SceneFlow — ManiBench Evaluation   ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "Model:    $MODEL"
echo "Strategy: $STRATEGY"
echo "Trials:   $TRIALS"
echo ""

# Check ManiBench dataset
if [ ! -f "ManiBench_Pilot_Dataset.json_ft" ]; then
    echo "⚠ ManiBench dataset not found (Expected: ManiBench_Pilot_Dataset.json_ft)."
    echo "  I should have already pre-downloaded it for you."
    exit 1
fi

# Run evaluation
echo "🔬 Starting evaluation..."
python -m evaluate.run_manibench \
    --model "$MODEL" \
    --strategy "$STRATEGY" \
    --trials "$TRIALS" \
    "$@"

echo ""
echo "✅ Evaluation complete! Results saved in results/"
echo ""
echo "To compare multiple runs:"
echo "  python -m evaluate.compare_results --results results/"
