"""Full-split scoring and explicit statistical baselines on the same targets."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from gpt2_scratch.model import GPT
from gpt2_scratch.tokenizer import BPETokenizer, Tokenizer


def metrics_from_nll(nll: float, targets: int, scored_bytes: int) -> dict:
    loss = nll / targets
    return {
        "cross_entropy_nats": loss,
        "perplexity": math.exp(loss),
        "bits_per_token": loss / math.log(2),
        "bits_per_byte": nll / (math.log(2) * scored_bytes),
        "scored_tokens": targets,
        "scored_bytes": scored_bytes,
    }


def baseline_nll(train: torch.Tensor, tokens: torch.Tensor, vocab_size: int) -> dict[str, float]:
    """Fit add-one unigram/bigram models on training tokens only."""
    unigram_counts = torch.bincount(train, minlength=vocab_size).double() + 1.0
    unigram = unigram_counts / unigram_counts.sum()
    bigram_counts = (
        torch.bincount(train[:-1] * vocab_size + train[1:], minlength=vocab_size * vocab_size)
        .reshape(vocab_size, vocab_size)
        .double()
        + 1.0
    )
    bigram = bigram_counts / bigram_counts.sum(dim=-1, keepdim=True)
    return {
        "uniform": (len(tokens) - 1) * math.log(vocab_size),
        "unigram": float(-unigram[tokens[1:]].log().sum()),
        "bigram": float(-bigram[tokens[:-1], tokens[1:]].log().sum()),
    }


@torch.inference_mode()
def full_split_nll(model: GPT, tokens: torch.Tensor, batch_size: int = 16) -> tuple[float, int]:
    """Score every next token once, resetting context at each nonoverlapping block.

    Short final sequences are right-padded and ignored in the loss. Causal attention
    prevents this right padding from changing earlier predictions.
    """
    if len(tokens) < 2 or batch_size < 1:
        raise ValueError("need at least two tokens and a positive batch size")
    was_training = model.training
    model.eval()
    block = model.config.block_size
    device = next(model.parameters()).device
    starts = list(range(0, len(tokens) - 1, block))
    nll, count = 0.0, 0
    try:
        for offset in range(0, len(starts), batch_size):
            group = starts[offset : offset + batch_size]
            length = min(block, len(tokens) - 1 - group[0])
            x = torch.zeros((len(group), length), dtype=torch.long)
            y = torch.full_like(x, -100)
            for row, start in enumerate(group):
                size = min(block, len(tokens) - 1 - start)
                x[row, :size] = tokens[start : start + size]
                y[row, :size] = tokens[start + 1 : start + size + 1]
                count += size
            logits, _ = model(x.to(device))
            loss = F.cross_entropy(
                logits.reshape(-1, model.config.vocab_size),
                y.to(device).reshape(-1),
                reduction="sum",
                ignore_index=-100,
            )
            nll += float(loss)
    finally:
        model.train(was_training)
    return nll, count


def evaluate_split(
    model: GPT,
    tokenizer: Tokenizer,
    train: torch.Tensor,
    tokens: torch.Tensor,
    split_bytes: int,
    batch_size: int = 16,
) -> dict:
    first = int(tokens[0])
    first_bytes = (
        len(tokenizer.vocab[first])
        if isinstance(tokenizer, BPETokenizer)
        else len(tokenizer.decode([first]).encode("utf-8"))
    )
    scored_bytes = split_bytes - first_bytes
    nll, count = full_split_nll(model, tokens, batch_size)
    if count != len(tokens) - 1:
        raise RuntimeError("evaluation did not cover every target exactly once")
    values = baseline_nll(train, tokens, tokenizer.vocab_size)
    values["gpt"] = nll
    return {name: metrics_from_nll(value, count, scored_bytes) for name, value in values.items()}
