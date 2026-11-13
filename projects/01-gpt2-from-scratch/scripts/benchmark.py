"""Measure update speed before committing to a long run; no model files are written."""

import argparse
import json
import time
from dataclasses import replace

import torch

from gpt2_scratch.config import ExperimentConfig
from gpt2_scratch.data import load_prepared, sample_batch
from gpt2_scratch.model import GPT
from gpt2_scratch.runtime import configure_runtime, resolve_device
from gpt2_scratch.trainer import build_optimizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="configs/char_cpu.json")
parser.add_argument("--data", default="data/processed/char")
parser.add_argument("--steps", type=int, default=20)
args = parser.parse_args()
if args.steps < 1:
    parser.error("--steps must be positive")
config = ExperimentConfig.load(args.config)
configure_runtime(config.train.seed, config.train.num_threads)
device = resolve_device(config.train.device)
tokenizer, arrays, _ = load_prepared(args.data, ("train",))
model = GPT(replace(config.model, vocab_size=tokenizer.vocab_size)).to(device)
optimizer = build_optimizer(model, config.train)
generator = torch.Generator().manual_seed(999)
elapsed = 0.0
for step in range(args.steps + 3):
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    x, y = sample_batch(
        arrays["train"], config.train.batch_size, model.config.block_size, generator, device
    )
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    if step >= 3:
        elapsed += time.perf_counter() - start
seconds_per_step = elapsed / args.steps
print(
    json.dumps(
        {
            "parameters": model.num_parameters,
            "device": str(device),
            "measured_steps_after_warmup": args.steps,
            "seconds_per_step": seconds_per_step,
            "tokens_per_second": config.train.batch_size
            * model.config.block_size
            / seconds_per_step,
            "estimated_training_minutes_excluding_evaluation": seconds_per_step
            * config.train.max_steps
            / 60,
        },
        indent=2,
    )
)
