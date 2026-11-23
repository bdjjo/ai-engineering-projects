"""Frozen transformer features with a learned Bradley-Terry scalar reward head."""

import copy

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from cai.io import fresh_directory, read_json, write_json
from cai.modeling import encode_response, hub_args, pad_batch


def bradley_terry_loss(chosen, rejected):
    if chosen.shape != rejected.shape or chosen.numel() == 0:
        raise ValueError("Reward vectors must match and be nonempty")
    return -F.logsigmoid(chosen - rejected).mean()


def last_nonpad(hidden, mask):
    if hidden.shape[:2] != mask.shape or not mask.bool().any(-1).all():
        raise ValueError("Each sequence needs at least one non-padding token")
    indices = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    positions = indices.masked_fill(~mask.bool(), -1).max(-1).values
    return hidden[torch.arange(len(hidden), device=hidden.device), positions]


class RewardModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.head = nn.Linear(backbone.config.hidden_size, 1, bias=False)

    @torch.no_grad()
    def features(self, batch):
        self.backbone.eval()
        mask = batch["attention_mask"]
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        hidden = self.backbone(
            input_ids=batch["input_ids"], attention_mask=mask, position_ids=positions
        ).last_hidden_state
        return last_nonpad(hidden, mask).float()

    def forward(self, batch):
        return self.head(self.features(batch)).squeeze(-1)


def encode_pairs(pairs, tokenizer, max_length):
    return [
        {
            side: encode_response(tokenizer, p["prompt"], p[side], max_length)
            for side in ("chosen", "rejected")
        }
        for p in pairs
    ]


@torch.no_grad()
def cache_features(model, pairs, tokenizer, max_length, batch_size=4):
    encoded = encode_pairs(pairs, tokenizer, max_length)
    device = next(model.parameters()).device
    result = {}
    for side in ("chosen", "rejected"):
        features = []
        for start in range(0, len(encoded), batch_size):
            batch = pad_batch(
                [r[side] for r in encoded[start : start + batch_size]],
                tokenizer.pad_token_id,
                device,
            )
            features.append(model.features(batch))
        result[side] = torch.cat(features)
    return result


@torch.no_grad()
def head_metrics(head, features):
    chosen = head(features["chosen"]).squeeze(-1)
    rejected = head(features["rejected"]).squeeze(-1)
    return {
        "loss": bradley_terry_loss(chosen, rejected).item(),
        "accuracy": (chosen > rejected).float().mean().item(),
        "ties": (chosen == rejected).sum().item(),
        "margin": (chosen - rejected).mean().item(),
        "pairs": len(chosen),
    }


def fit_reward(model, tokenizer, train, validation, out, config, model_spec):
    out = fresh_directory(out)
    train_features = cache_features(model, train, tokenizer, config["max_length"])
    val_features = cache_features(model, validation, tokenizer, config["max_length"])
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=config["reward_lr"])
    history, best_loss, best_state, best_step = [], float("inf"), None, None
    for step in range(config["reward_steps"] + 1):
        if step > 0:
            chosen = model.head(train_features["chosen"]).squeeze(-1)
            rejected = model.head(train_features["rejected"]).squeeze(-1)
            loss = bradley_terry_loss(chosen, rejected)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
            optimizer.step()
        metrics = {
            "step": step,
            "train": head_metrics(model.head, train_features),
            "validation": head_metrics(model.head, val_features),
        }
        history.append(metrics)
        if metrics["validation"]["loss"] < best_loss:
            best_loss, best_step = metrics["validation"]["loss"], step
            best_state = copy.deepcopy(model.head.state_dict())
    model.head.load_state_dict(best_state)
    with torch.no_grad():
        scores = model.head(torch.cat(list(train_features.values()))).squeeze(-1)
    normalization = {
        "mean": scores.mean().item(),
        "std": max(scores.std(unbiased=False).item(), 1e-6),
    }
    summary = {
        "backbone": model_spec,
        "frozen_backbone": True,
        "selected_step": best_step,
        "normalization": normalization,
        "history": history,
        "validation": head_metrics(model.head, val_features),
        "length_baseline_accuracy": sum(len(p["chosen"]) > len(p["rejected"]) for p in validation)
        / len(validation),
    }
    torch.save({k: v.cpu() for k, v in best_state.items()}, out / "head.pt")
    write_json(out / "summary.json", summary)
    return summary


def load_reward(path, device):
    from pathlib import Path

    path = Path(path)
    summary = read_json(path / "summary.json")
    backbone = AutoModel.from_pretrained(**hub_args(summary["backbone"]), torch_dtype=torch.float32)
    reward = RewardModel(backbone).to(device).eval()
    reward.head.load_state_dict(
        torch.load(path / "head.pt", map_location=device, weights_only=True)
    )
    return reward, summary


@torch.no_grad()
def score_texts(reward, tokenizer, prompts, answers, max_length, normalization):
    rows = [
        encode_response(tokenizer, p, a, max_length) for p, a in zip(prompts, answers, strict=True)
    ]
    batch = pad_batch(rows, tokenizer.pad_token_id, next(reward.parameters()).device)
    scores = reward(batch)
    return ((scores - normalization["mean"]) / normalization["std"]).clamp(-5, 5)
