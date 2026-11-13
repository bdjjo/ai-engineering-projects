"""GPT-style decoder: explicit attention math, pre-normalization, and tied embeddings.

PyTorch supplies tensors, automatic differentiation, and primitive layers. No
pretrained model, nn.Transformer, Hugging Face model, or nanoGPT code is used.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from gpt2_scratch.config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.backend = config.attention_backend
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.ones(config.block_size, config.block_size, dtype=torch.bool).tril(),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        head_dim = width // self.n_head
        q, k, v = [
            tensor.reshape(batch, length, self.n_head, head_dim).transpose(1, 2)
            for tensor in (q, k, v)
        ]
        if self.backend == "manual":
            # [B,H,T,d] @ [B,H,d,T] -> [B,H,T,T]. Future positions get probability 0.
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
            scores = scores.masked_fill(~self.causal_mask[:length, :length], float("-inf"))
            weights = F.softmax(scores, dim=-1)
            weights = F.dropout(weights, p=self.dropout, training=self.training)
            attended = weights @ v
        else:
            # Optional equivalent kernel; the selected implementation depends on hardware.
            attended = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        attended = attended.transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.resid_dropout(self.proj(attended))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.expand = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.expand(x), approximate="tanh")))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attention = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.final_norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding.weight
        for name, param in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(param, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self, ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if ids.ndim != 2 or not 0 < ids.shape[1] <= self.config.block_size:
            raise ValueError("input must have shape [batch, time], with 1 <= time <= block_size")
        if targets is not None and targets.shape != ids.shape:
            raise ValueError("targets must have the same shape as input")
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.dropout(self.token_embedding(ids) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        # Always return [B,T,V]; a single consistent contract also enables causality tests.
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 40,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError("provide a nonempty prompt of shape [batch, time]")
        if max_new_tokens < 0 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(
                "max_new_tokens must be nonnegative; temperature must be finite and > 0"
            )
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive or None")
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                logits, _ = self(ids[:, -self.config.block_size :])
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    values, indices = torch.topk(logits, min(top_k, logits.shape[-1]))
                    filtered = torch.full_like(logits, float("-inf"))
                    logits = filtered.scatter(-1, indices, values)
                next_id = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)
                ids = torch.cat((ids, next_id), dim=1)
            return ids
        finally:
            self.train(was_training)
