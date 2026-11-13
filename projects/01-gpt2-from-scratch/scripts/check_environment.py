"""Show the active interpreter and compute backend before starting a run."""

import json
import sys

import torch

from gpt2_scratch.config import ModelConfig
from gpt2_scratch.model import GPT
from gpt2_scratch.runtime import configure_runtime, environment_info, resolve_device

configure_runtime(1337, 4)
device = resolve_device("auto")
print(f"Interpreter: {sys.executable}")
print(json.dumps(environment_info(device), indent=2))
model = GPT(ModelConfig(vocab_size=65, block_size=16, n_layer=1, n_head=2, n_embd=32)).to(device)
logits, loss = model(
    torch.zeros((2, 16), dtype=torch.long, device=device),
    torch.ones((2, 16), dtype=torch.long, device=device),
)
loss.backward()
print(f"Forward/backward OK | logits {tuple(logits.shape)} | loss {float(loss.detach()):.4f}")
