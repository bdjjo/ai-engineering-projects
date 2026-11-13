"""DPO from sequence log probabilities, with explicit causal response masking."""

import math

from torch.nn import functional as F


def sequence_logps(logits, labels):
    if logits.shape[:2] != labels.shape or labels.shape[1] < 2:
        raise ValueError("logits and labels must align and contain at least two positions")
    targets = labels[:, 1:]
    mask = targets != -100
    counts = mask.sum(-1)
    if (counts == 0).any():
        raise ValueError("every sequence must contain a scored response token")
    losses = F.cross_entropy(
        logits[:, :-1].float().transpose(1, 2), targets, reduction="none", ignore_index=-100
    )
    return -(losses * mask).sum(-1), counts


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.1):
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    if not (chosen.shape == rejected.shape == ref_chosen.shape == ref_rejected.shape):
        raise ValueError("all log probability vectors must have the same shape")
    margin = beta * ((chosen - ref_chosen.detach()) - (rejected - ref_rejected.detach()))
    return -F.logsigmoid(margin).mean(), margin.detach()
