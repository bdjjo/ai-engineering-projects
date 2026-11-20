"""Pre-normalized GPT with causal SDPA and tied token/output embeddings."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.projection = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [
            t.view(batch, length, self.n_head, width // self.n_head).transpose(1, 2)
            for t in (q, k, v)
        ]
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        return self.projection(y.transpose(1, 2).contiguous().view(batch, length, width))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
        )

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._initialize)
        self.lm_head.weight = self.token_embedding.weight
        for block in self.blocks:
            for weight in (block.attention.projection.weight, block.mlp[2].weight):
                nn.init.normal_(weight, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, targets=None):
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= self.config.block_size:
            raise ValueError("Tokens must be [batch, sequence] within the model context")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.dropout(self.token_embedding(tokens) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = (
            None
            if targets is None
            else F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        )
        return logits, loss

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def parameter_count_formula(config):
    d, layers = config.n_embd, config.n_layer
    return (
        (config.vocab_size + config.block_size) * d
        + layers * 12 * d * d
        + (4 * layers + 2) * d
        + (layers * 9 * d if config.bias else 0)
    )


def optimizer_groups(model, weight_decay):
    """Capture original dimensionality BEFORE FSDP turns parameters into local 1D shards."""
    decay, no_decay = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
