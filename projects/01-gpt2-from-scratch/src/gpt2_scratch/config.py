"""Validated configuration, shared by training, checkpoints, and generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 65  # Replaced by the prepared tokenizer's actual vocabulary.
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True
    attention_backend: Literal["manual", "sdpa"] = "manual"

    def __post_init__(self) -> None:
        for name in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if self.attention_backend not in ("manual", "sdpa"):
            raise ValueError("attention_backend must be manual or sdpa")


@dataclass(frozen=True)
class TrainConfig:
    max_steps: int = 2500
    batch_size: int = 16
    learning_rate: float = 0.001
    min_lr: float = 0.0001
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_batches: int = 20
    log_interval: int = 100
    seed: int = 1337
    device: str = "auto"
    num_threads: int = 4

    def __post_init__(self) -> None:
        for name in (
            "max_steps",
            "batch_size",
            "eval_interval",
            "eval_batches",
            "log_interval",
            "num_threads",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.warmup_steps < self.max_steps:
            raise ValueError("warmup_steps must lie in [0, max_steps)")
        if not 0 < self.min_lr <= self.learning_rate:
            raise ValueError("require 0 < min_lr <= learning_rate")
        if self.grad_clip <= 0 or self.weight_decay < 0:
            raise ValueError("grad_clip must be positive and weight_decay nonnegative")
        if not (0 <= self.beta1 < 1 and 0 <= self.beta2 < 1):
            raise ValueError("AdamW beta values must lie in [0, 1)")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if set(payload) != {"model", "train"}:
            raise ValueError("config must contain exactly model and train objects")
        return cls(ModelConfig(**payload["model"]), TrainConfig(**payload["train"]))

    def to_dict(self) -> dict:
        return asdict(self)
