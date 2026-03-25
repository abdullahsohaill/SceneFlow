#!/bin/bash
# ──────────────────────────────────────────
# SceneFlow — Fine-Tuning Launch Script
# ──────────────────────────────────────────
# Usage:
#   bash scripts/run_train.sh configs/qwen3b_qlora.yaml
#   bash scripts/run_train.sh configs/qwen7b_qlora.yaml
#   bash scripts/run_train.sh configs/qwen3b_qlora.yaml --dry-run
#
# For multi-GPU:
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_train.sh configs/qwen7b_qlora.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

CONFIG=${1:?Usage: bash scripts/run_train.sh <config.yaml> [--dry-run]}
shift

echo "╔══════════════════════════════════════╗"
echo "║   SceneFlow — Manim Fine-Tuning      ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "Config: $CONFIG"
echo "Extra args: $@"
echo "Working dir: $SCRIPT_DIR"
echo ""

# Check if data exists
if [ ! -f "data/train.jsonl_ft" ]; then
    echo "⚠ Pre-downloaded training data (train.jsonl_ft) not found."
    python -m data.prepare_dataset
    echo ""
fi

# Count examples
TRAIN_COUNT=$(wc -l < data/train.jsonl_ft 2>/dev/null || echo "?")
VAL_COUNT=$(wc -l < data/val.jsonl_ft 2>/dev/null || echo "?")
echo "📊 Data: $TRAIN_COUNT train / $VAL_COUNT val examples"
echo ""

# Launch training
echo "🚀 Starting training..."
python -m train.finetune_qlora --config "$CONFIG" "$@"

echo ""
echo "✅ Training complete!"
