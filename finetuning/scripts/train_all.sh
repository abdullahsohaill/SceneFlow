#!/bin/bash
# ──────────────────────────────────────────
# SceneFlow — Bulk Fine-Tuning (3B + 7B)
# ──────────────────────────────────────────
# Usage: bash scripts/train_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════╗"
echo "║   SceneFlow — Bulk Manim Training    ║"
echo "╚══════════════════════════════════════╝"
echo ""

# 1. Setup Environment
echo "📦 Installing dependencies..."
pip install -r requirements.txt
echo ""

# 2. Train Qwen2.5-Coder-3B
echo "🚀 [1/2] Training Qwen2.5-Coder-3B..."
bash scripts/run_train.sh configs/qwen3b_qlora.yaml
echo "✅ 3B Training Complete!"
echo ""

# 3. Train Qwen2.5-Coder-7B
echo "🚀 [2/2] Training Qwen2.5-Coder-7B..."
bash scripts/run_train.sh configs/qwen7b_qlora.yaml
echo "✅ 7B Training Complete!"
echo ""

echo "🎉 ALL MODELS TRAINED SUCCESSFULLY!"
echo "Check the 'outputs/' directory for checkpoints."
echo ""
echo "Next step: Run evaluation using scripts/run_eval.sh"
