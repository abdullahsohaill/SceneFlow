# SceneFlow — Fine-Tuning & ManiBench Evaluation

Fine-tune **Qwen2.5-Coder-3B / 7B** (8-bit QLoRA) on the 3Blue1Brown Manim dataset and evaluate on the [ManiBench](https://github.com/nabin2004/ManiBench) benchmark.

## Quick Start (Lab GPU)

I have already **pre-downloaded and formatted** the training data and benchmark to save time!

```bash
# SINGLE COMMAND: Setup + Baseline + Train Both + Evaluate Both
cd finetuning
bash scripts/run_full_pipeline.sh
```

> This will take ~4-8 hours depends on the GPU. You can run it and leave it!

# 4. Evaluate on ManiBench
python -m evaluate.run_manibench --model outputs/qwen3b-manim-ft --strategy zero_shot --trials 3

# 5. Compare base vs fine-tuned
python -m evaluate.compare_results --results results/
```

## Directory Structure

```
finetuning/
├── configs/              # Training YAML configs
├── data/                 # Dataset preparation & filtering
├── train/                # QLoRA fine-tuning script
├── evaluate/             # ManiBench evaluation pipeline
├── scripts/              # Shell scripts for lab GPU runs
├── outputs/              # Trained model checkpoints (gitignored)
└── results/              # Evaluation results (gitignored)
```

## GPU Requirements

| Model | VRAM (8-bit QLoRA) | Estimated Time |
|-------|-------------------|----------------|
| Qwen2.5-Coder-3B | ~8 GB | ~1-2 hours |
| Qwen2.5-Coder-7B | ~16 GB | ~2-4 hours |
