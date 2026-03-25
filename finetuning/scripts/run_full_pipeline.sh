#!/bin/bash
# ──────────────────────────────────────────
# SceneFlow — UNIFIED PIPE (TRAIN + EVAL)
# ──────────────────────────────────────────
# Usage: bash scripts/run_full_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════╗"
echo "║   SceneFlow — Full Manim Pipeline    ║"
echo "╚══════════════════════════════════════╝"
echo ""

# 1. Setup Environment
echo "📦 [1/7] Installing dependencies..."
pip install -r requirements.txt
mkdir -p results outputs
echo ""

# 2. Baseline Evaluation (Base Models)
echo "🔬 [2/7] Running Baseline Evaluation (3B-Instruct)..."
bash scripts/run_eval.sh Qwen/Qwen2.5-Coder-3B-Instruct zero_shot 3 --name "base_3b"
echo "🔬 [3/7] Running Baseline Evaluation (7B-Instruct)..."
bash scripts/run_eval.sh Qwen/Qwen2.5-Coder-7B-Instruct zero_shot 3 --name "base_7b"
echo ""

# 3. Fine-Tuning
echo "🚀 [4/7] Fine-Tuning Qwen2.5-Coder-3B..."
bash scripts/run_train.sh configs/qwen3b_qlora.yaml
echo "🚀 [5/7] Fine-Tuning Qwen2.5-Coder-7B..."
bash scripts/run_train.sh configs/qwen7b_qlora.yaml
echo ""

# 4. Post-Training Evaluation
echo "🔬 [6/7] Evaluating Fine-Tuned 3B..."
bash scripts/run_eval.sh outputs/qwen3b-manim-ft-merged zero_shot 3 --name "ft_3b"
echo "🔬 [7/7] Evaluating Fine-Tuned 7B..."
bash scripts/run_eval.sh outputs/qwen7b-manim-ft-merged zero_shot 3 --name "ft_7b"
echo ""

# 5. Final Comparison
echo "📊 [FINISH] Generating Final Comparison Report..."
python -m evaluate.compare_results --results results/
echo ""
echo "🎉 PIPELINE COMPLETE!"
echo "Summary results are saved in the 'results/' folder."
echo "You can now hand the codebase back for analysis."
