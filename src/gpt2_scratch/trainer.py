"""Deterministic sampling, AdamW, warmup/cosine decay, and resumable checkpoints."""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import torch

from gpt2_scratch.config import ExperimentConfig, TrainConfig
from gpt2_scratch.data import load_prepared, sample_batch
from gpt2_scratch.model import GPT
from gpt2_scratch.runtime import (
    atomic_torch_save,
    configure_runtime,
    environment_info,
    load_checkpoint,
    resolve_device,
    source_hashes,
    write_json,
)
from gpt2_scratch.tokenizer import save_tokenizer


def learning_rate(step: int, config: TrainConfig) -> float:
    """LR for a one-based optimizer update; the first update has a nonzero LR."""
    if step < 1:
        raise ValueError("step is one-based")
    if config.warmup_steps and step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    progress = min(1.0, (step - config.warmup_steps) / (config.max_steps - config.warmup_steps))
    return config.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (
        config.learning_rate - config.min_lr
    )


def build_optimizer(model: GPT, config: TrainConfig) -> torch.optim.AdamW:
    # named_parameters/parameters deduplicates the tied embedding/output matrix.
    decay = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.ndim < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )


@torch.no_grad()
def estimate_losses(model: GPT, arrays: dict, config: TrainConfig, device: torch.device) -> dict:
    """Fixed evaluation windows; evaluating never advances the training RNG."""
    was_training = model.training
    model.eval()
    results = {}
    try:
        for offset, split in enumerate(("train", "val")):
            generator = torch.Generator().manual_seed(config.seed + 1000 + offset)
            total = 0.0
            for _ in range(config.eval_batches):
                x, y = sample_batch(
                    arrays[split], config.batch_size, model.config.block_size, generator, device
                )
                _, loss = model(x, y)
                total += float(loss)
            results[f"{split}_loss"] = total / config.eval_batches
        return results
    finally:
        model.train(was_training)


def train(
    config: ExperimentConfig,
    data_dir: str | Path,
    run_dir: str | Path,
    resume: str | Path | None = None,
    stop_after: int | None = None,
) -> dict:
    cfg = config.train
    configure_runtime(cfg.seed, cfg.num_threads)
    device = resolve_device(cfg.device)
    tokenizer, arrays, manifest = load_prepared(data_dir)
    model_cfg = replace(config.model, vocab_size=tokenizer.vocab_size)
    config = ExperimentConfig(model_cfg, cfg)
    if any(len(tokens) <= model_cfg.block_size for tokens in arrays.values()):
        raise ValueError("every training/validation split must exceed block_size")
    target_step = min(stop_after, cfg.max_steps) if stop_after is not None else cfg.max_steps
    if target_step <= 0:
        raise ValueError("stop_after must be positive")
    run_dir = Path(run_dir)
    if resume is None and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"{run_dir} is not empty; choose a new run or use --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    model = GPT(model_cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    train_rng = torch.Generator().manual_seed(cfg.seed + 1)
    step, best_step, best_val = 0, 0, float("inf")
    training_seconds, elapsed_previous = 0.0, 0.0
    history: list[dict] = []
    started = datetime.now(UTC).isoformat()
    if resume is not None:
        if Path(resume).resolve().parent != run_dir.resolve() or Path(resume).name != "last.pt":
            raise ValueError("resume from last.pt in the same run directory")
        checkpoint = load_checkpoint(resume)
        if checkpoint["config"] != config.to_dict() or checkpoint["data_manifest"] != manifest:
            raise ValueError("resume requires identical config, tokenizer, and dataset")
        if checkpoint["training_source_hashes"] != source_hashes():
            raise ValueError("training source changed; start a new run instead of exact resume")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]
        if target_step <= step:
            raise ValueError("requested stop step must be greater than checkpoint step")
        best_step, best_val = checkpoint["best_step"], checkpoint["best_val_loss"]
        history = checkpoint["history"]
        training_seconds = checkpoint["training_seconds"]
        elapsed_previous = checkpoint["elapsed_seconds"]
        started = checkpoint["started_utc"]
        train_rng.set_state(checkpoint["train_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    wall_start = time.perf_counter()
    metadata = {
        "run_name": run_dir.name,
        "started_utc": started,
        "config": config.to_dict(),
        "parameters": model.num_parameters,
        "environment": environment_info(device),
        "data_manifest": manifest,
        "training_source_hashes": source_hashes(),
    }
    write_json(run_dir / "metadata.json", metadata)
    save_tokenizer(tokenizer, run_dir / "tokenizer.json")
    # Rebuild the log from the checkpoint so interrupted trailing log lines are not duplicated.
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for row in history:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def record(row: dict) -> None:
        history.append(row)
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def save(path: Path) -> None:
        atomic_torch_save(
            {
                "format_version": 1,
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config.to_dict(),
                "tokenizer": tokenizer.to_dict(),
                "data_manifest": manifest,
                "best_step": best_step,
                "best_val_loss": best_val,
                "history": history,
                "torch_rng": torch.get_rng_state(),
                "train_rng": train_rng.get_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "training_seconds": training_seconds,
                "elapsed_seconds": elapsed_previous + time.perf_counter() - wall_start,
                "started_utc": started,
                "training_source_hashes": metadata["training_source_hashes"],
            },
            path,
        )

    def evaluate_and_save() -> None:
        nonlocal best_val, best_step
        losses = estimate_losses(model, arrays, cfg, device)
        if not all(math.isfinite(value) for value in losses.values()):
            raise FloatingPointError("nonfinite evaluation loss")
        record(
            {
                "event": "eval",
                "step": step,
                **losses,
                "val_perplexity": math.exp(losses["val_loss"]),
                "elapsed_seconds": elapsed_previous + time.perf_counter() - wall_start,
            }
        )
        print(
            f"step {step:5d} | train {losses['train_loss']:.4f} | val {losses['val_loss']:.4f}",
            flush=True,
        )
        if losses["val_loss"] < best_val:
            best_val, best_step = losses["val_loss"], step
            save(run_dir / "best.pt")
        save(run_dir / "last.pt")

    print(
        f"{model.num_parameters:,} parameters | {device} | {cfg.num_threads} threads | "
        f"{model_cfg.attention_backend} attention",
        flush=True,
    )
    if resume is None:
        evaluate_and_save()
    model.train()
    first_update = step + 1
    for step in range(first_update, target_step + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        update_start = time.perf_counter()
        x, y = sample_batch(
            arrays["train"], cfg.batch_size, model_cfg.block_size, train_rng, device
        )
        lr = learning_rate(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite training loss at update {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.grad_clip, error_if_nonfinite=True
        )
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        training_seconds += time.perf_counter() - update_start
        if step % cfg.log_interval == 0 or step == target_step:
            record(
                {
                    "event": "train",
                    "step": step,
                    "loss": float(loss.detach()),
                    "lr": lr,
                    "grad_norm_before_clip": float(grad_norm),
                    "training_tokens": step * cfg.batch_size * model_cfg.block_size,
                    "training_tokens_per_second": step
                    * cfg.batch_size
                    * model_cfg.block_size
                    / training_seconds,
                }
            )
            print(
                f"update {step:5d} | loss {float(loss.detach()):.4f} | lr {lr:.2e} | "
                f"{step * cfg.batch_size * model_cfg.block_size / training_seconds:.0f} tok/s",
                flush=True,
            )
        # A manual pause saves last.pt without adding extra model-selection evaluations.
        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            evaluate_and_save()
        elif step == target_step:
            save(run_dir / "last.pt")
    summary = {
        "status": "complete" if step == cfg.max_steps else "paused",
        "steps_completed": step,
        "best_step": best_step,
        "best_val_loss": best_val,
        "parameters": model.num_parameters,
        "training_tokens": step * cfg.batch_size * model_cfg.block_size,
        "training_seconds": training_seconds,
        "elapsed_seconds": elapsed_previous + time.perf_counter() - wall_start,
        "training_tokens_per_second": step
        * cfg.batch_size
        * model_cfg.block_size
        / training_seconds,
    }
    write_json(run_dir / "summary.json", summary)
    return summary
