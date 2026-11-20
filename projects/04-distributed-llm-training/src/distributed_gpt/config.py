import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 128
    block_size: int = 32
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.0
    bias: bool = False

    def __post_init__(self):
        for name in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("Embedding width must be divisible by head count")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")


@dataclass
class TrainConfig:
    steps: int = 20
    batch_size: int = 2
    accumulation: int = 2
    learning_rate: float = 0.001
    weight_decay: float = 0.1
    warmup_steps: int = 2
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    save_every: int = 10
    log_every: int = 5
    measure_after: int = 3
    validation_batches: int = 4
    activation_checkpointing: bool = True
    precision: str = "fp32"
    seed: int = 1337

    def __post_init__(self):
        for name in (
            "steps",
            "batch_size",
            "accumulation",
            "save_every",
            "log_every",
            "validation_batches",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("learning_rate", "grad_clip"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.weight_decay < 0 or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("Invalid optimizer settings")
        if not 0 <= self.warmup_steps < self.steps or self.measure_after < 0:
            raise ValueError("Invalid warmup or measurement interval")
        if self.precision not in {"fp32", "fp16", "bf16", "auto"}:
            raise ValueError("Unknown precision")


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self):
        return asdict(self)


def load_config(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig(ModelConfig(**value["model"]), TrainConfig(**value["train"]))


def learning_rate(update, config):
    """Zero-based update index; the first warmup update has a nonzero learning rate."""
    if not 0 <= update < config.steps:
        raise ValueError("Update outside the configured schedule")
    if update < config.warmup_steps:
        return config.learning_rate * (update + 1) / max(1, config.warmup_steps)
    denominator = max(1, config.steps - config.warmup_steps - 1)
    ratio = (update - config.warmup_steps) / denominator
    return config.learning_rate * (
        config.min_lr_ratio + (1 - config.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * ratio))
    )
