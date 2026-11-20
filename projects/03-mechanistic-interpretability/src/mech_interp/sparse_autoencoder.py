"""ReLU sparse autoencoder with constrained decoder directions."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SAEConfig:
    d_model: int = 768
    hidden_dim: int = 1024
    l1_coefficient: float = 0.01
    lr: float = 0.0003
    batch_size: int = 128
    steps: int = 500
    eval_every: int = 50
    seed: int = 1337

    def __post_init__(self):
        for name in ("d_model", "hidden_dim", "batch_size", "steps", "eval_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.l1_coefficient < 0 or self.lr <= 0:
            raise ValueError("Invalid L1 coefficient or learning rate")


class SparseAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.W_dec = nn.Parameter(torch.randn(config.hidden_dim, config.d_model))
        self.normalize_decoder()
        self.W_enc = nn.Parameter(self.W_dec.detach().T.clone())
        self.b_enc = nn.Parameter(torch.zeros(config.hidden_dim))
        self.b_dec = nn.Parameter(torch.zeros(config.d_model))

    def encode(self, x):
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def reconstruct(self, features):
        return features @ self.W_dec + self.b_dec

    def forward(self, x):
        features = self.encode(x)
        reconstructed = self.reconstruct(features)
        # Both terms are summed across dimensions, then averaged across tokens.
        mse = (reconstructed - x).square().sum(-1).mean()
        l1 = features.sum(-1).mean()
        loss = mse + self.config.l1_coefficient * l1
        return loss, reconstructed, features

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.div_(self.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-8))

    @torch.no_grad()
    def project_decoder_gradient(self):
        if self.W_dec.grad is not None:
            parallel = (self.W_dec.grad * self.W_dec).sum(-1, keepdim=True)
            self.W_dec.grad.sub_(parallel * self.W_dec)


@torch.inference_mode()
def evaluate(sae, activations, scale, mean, batch_size=256):
    """Global held-out FVU denominator uses the training per-dimension mean."""
    device = next(sae.parameters()).device
    mean = mean.to(device)
    squared_error = total_variance = l1 = l0 = 0.0
    fired = torch.zeros(sae.config.hidden_dim, dtype=torch.bool, device=device)
    for start in range(0, len(activations), batch_size):
        x = activations[start : start + batch_size].to(device).float() / scale
        _, reconstructed, features = sae(x)
        squared_error += (reconstructed - x).square().sum().item()
        total_variance += (x - mean).square().sum().item()
        l1 += features.sum().item()
        l0 += (features > 0).sum().item()
        fired |= (features > 0).any(0)
    n = len(activations)
    if n == 0:
        raise ValueError("Evaluation set is empty")
    return {
        "mse_per_dimension": squared_error / (n * sae.config.d_model),
        "l1_per_token": l1 / n,
        "l0": l0 / n,
        "inactive_fraction": (~fired).float().mean().item(),
        "variance_explained": 1 - squared_error / total_variance if total_variance > 0 else None,
        "tokens": n,
    }
