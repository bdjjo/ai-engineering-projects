"""Load a self-contained checkpoint and sample a continuation with a fixed seed."""

from __future__ import annotations

from pathlib import Path

import torch

from gpt2_scratch.config import ModelConfig
from gpt2_scratch.model import GPT
from gpt2_scratch.runtime import load_checkpoint
from gpt2_scratch.tokenizer import Tokenizer, tokenizer_from_dict


def load_model(path: str | Path, device: torch.device) -> tuple[GPT, Tokenizer, dict]:
    checkpoint = load_checkpoint(path)
    model = GPT(ModelConfig(**checkpoint["config"]["model"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
    return model, tokenizer, checkpoint


def generate_text(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 0.8,
    top_k: int | None = 40,
    seed: int = 42,
) -> str:
    if not prompt:
        raise ValueError("prompt cannot be empty")
    device = next(model.parameters()).device
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    output = model.generate(ids, max_new_tokens, temperature, top_k, generator)
    return tokenizer.decode(output[0].tolist())
