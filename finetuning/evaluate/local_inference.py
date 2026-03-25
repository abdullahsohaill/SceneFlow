"""
SceneFlow Fine-Tuning — Local Model Inference
==============================================
Wraps transformers pipeline for running inference on local
(base or fine-tuned) Qwen2.5-Coder models.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


class LocalInference:
    """
    Local inference wrapper for Qwen2.5-Coder models.
    Supports both base HuggingFace models and fine-tuned local checkpoints.
    """

    def __init__(
        self,
        model_path: str,
        load_in_8bit: bool = True,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        device_map: str = "auto",
    ):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        logger.info("Loading model: %s", model_path)

        # Quantization config
        bnb_config = None
        if load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()

        logger.info("Model loaded successfully")

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Generate a response given a prompt.
        Uses the model's chat template for proper formatting.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the generated part (skip the input)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)

        return response.strip()

    def extract_code(self, response: str) -> str:
        """
        Extract Python code from model response.
        Handles markdown fences and raw code.
        """
        # Try to extract from markdown code blocks
        if "```python" in response:
            parts = response.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0]
                return code.strip()

        if "```" in response:
            parts = response.split("```")
            if len(parts) > 2:
                code = parts[1]
                # Remove language identifier if present
                if code.startswith(("python\n", "py\n")):
                    code = code.split("\n", 1)[1]
                return code.strip()

        # Assume the whole response is code
        return response.strip()
