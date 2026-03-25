"""
SceneFlow Fine-Tuning — QLoRA Training Script
==============================================
Fine-tunes Qwen2.5-Coder models using QLoRA (8-bit quantization)
on the prepared Manim CE training data.

Usage:
    python -m train.finetune_qlora --config configs/qwen3b_qlora.yaml
    python -m train.finetune_qlora --config configs/qwen7b_qlora.yaml --max-steps 5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from rich.console import Console
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

logger = logging.getLogger(__name__)
console = Console()


def load_config(config_path: str) -> dict:
    """Load training config from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_quantization(config: dict) -> BitsAndBytesConfig:
    """Create BitsAndBytesConfig from config."""
    quant_cfg = config["quantization"]

    if quant_cfg.get("load_in_8bit", False):
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(
                torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
            ),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )


def setup_lora(config: dict) -> LoraConfig:
    """Create LoRA config."""
    lora_cfg = config["lora"]
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=lora_cfg["task_type"],
        bias=lora_cfg["bias"],
    )


def load_model_and_tokenizer(config: dict, bnb_config: BitsAndBytesConfig):
    """Load quantized model and tokenizer."""
    model_name = config["model"]["name"]
    trust_remote = config["model"].get("trust_remote_code", True)

    console.print(f"\n[bold blue]🔧 Loading model:[/] {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Determine dtype from config
    train_cfg = config.get("training", {})
    dtype = torch.bfloat16 if train_cfg.get("bf16", False) else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=trust_remote,
        torch_dtype=dtype,
    )

    model.config.use_cache = False  # disable for training

    console.print(f"[green]✅ Model loaded: {model_name}[/]")
    _print_model_info(model)

    return model, tokenizer


def _print_model_info(model):
    """Print trainable parameter count."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(
        f"[dim]Parameters: {total:,} total, {trainable:,} trainable "
        f"({100 * trainable / total:.2f}%)[/]"
    )


def load_data(config: dict, tokenizer):
    """Load prepared JSONL datasets."""
    data_cfg = config["data"]
    train_path = data_cfg["train_file"]
    val_path = data_cfg["val_file"]

    console.print(f"\n[bold blue]📊 Loading data:[/]")
    console.print(f"  Train: {train_path}")
    console.print(f"  Val:   {val_path}")

    train_ds = load_dataset("json", data_files=train_path, split="train")
    val_ds = load_dataset("json", data_files=val_path, split="train")

    console.print(f"[green]  Train: {len(train_ds)} examples[/]")
    console.print(f"[green]  Val:   {len(val_ds)} examples[/]")

    return train_ds, val_ds



def get_text(example):
    """Top-level, purely picklable formatting function for Windows."""
    return example["text"]


def train(config: dict, max_steps: int | None = None, dry_run: bool = False):
    """Main training loop."""
    train_cfg = config["training"]

    # ── Setup ──
    bnb_config = setup_quantization(config)
    lora_config = setup_lora(config)

    model, tokenizer = load_model_and_tokenizer(config, bnb_config)
    model = get_peft_model(model, lora_config)
    _print_model_info(model)

    train_ds, val_ds = load_data(config, tokenizer)

    # ── Training arguments ──
    output_dir = train_cfg["output_dir"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=train_cfg["eval_strategy"],
        eval_steps=train_cfg["eval_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        optim=train_cfg["optim"],
        report_to=train_cfg["report_to"],
        max_steps=max_steps if max_steps else -1,
    )

    # ── Format dataset manually to avoid Windows dill/multiprocessing crashes ──
    from datasets import Dataset
    
    def _convert_to_text_ds(ds):
        texts = [tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False) for ex in ds]
        return Dataset.from_dict({"text": texts})

    train_ds = _convert_to_text_ds(train_ds)
    val_ds = _convert_to_text_ds(val_ds)

    # ── Trainer ──
    tokenizer.model_max_length = train_cfg["max_seq_length"]
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        formatting_func=get_text,
    )

    if dry_run:
        console.print("\n[bold yellow]🏃 Dry run — training for a few steps...[/]")
    else:
        console.print(f"\n[bold green]🚀 Starting training...[/]")
        console.print(f"[dim]Note: Training 3B model in 8-bit precision with batch size {train_cfg['per_device_train_batch_size']}[/]")

    # ── Train ──
    trainer.train()

    # ── Save ──
    console.print(f"\n[bold blue]💾 Saving LoRA adapter to:[/] {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metrics
    metrics = trainer.state.log_history
    metrics_path = Path(output_dir) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    console.print(f"[green]✅ Training complete![/]")

    # ── Merge (optional) ──
    merge_cfg = config.get("merge", {})
    if merge_cfg.get("enabled", False) and not dry_run:
        merge_path = merge_cfg["output_dir"]
        console.print(f"\n[bold blue]🔗 Merging LoRA into base model → {merge_path}[/]")
        _merge_and_save(config, output_dir, merge_path)

    return output_dir


def _merge_and_save(config: dict, adapter_path: str, output_path: str):
    """Merge LoRA adapter back into the base model and save."""
    model_name = config["model"]["name"]
    trust_remote = config["model"].get("trust_remote_code", True)

    # Load base model in matching precision for merging
    train_cfg = config.get("training", {})
    dtype = torch.bfloat16 if train_cfg.get("bf16", False) else torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=trust_remote,
        torch_dtype=dtype,
    )

    # Load and merge LoRA
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = model.merge_and_unload()

    # Save
    merged_model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)

    console.print(f"[green]✅ Merged model saved to: {output_path}[/]")


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Manim code generation")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Override max training steps (for testing)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run only a few steps for testing"
    )
    args = parser.parse_args()

    if args.dry_run and args.max_steps is None:
        args.max_steps = 5

    config = load_config(args.config)
    train(config, max_steps=args.max_steps, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
