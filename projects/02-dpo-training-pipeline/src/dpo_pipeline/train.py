"""Explicit loops; cache the frozen SFT reference before the first DPO update."""

import json
import math
import random
from pathlib import Path

import torch

from .data import batch
from .objective import dpo_loss, sequence_logps


def score(model, item, pad_id):
    tensors = batch([item], pad_id, next(model.parameters()).device)
    labels = tensors.pop("labels")
    return sequence_logps(model(**tensors).logits, labels)


@torch.no_grad()
def reference_scores(model, rows, pad_id):
    model.eval()
    return [
        (score(model, r["chosen"], pad_id)[0].item(), score(model, r["rejected"], pad_id)[0].item())
        for r in rows
    ]


@torch.no_grad()
def evaluate(model, rows, pad_id, refs=None, beta=0.1):
    model.eval()
    losses, margins, likelihood_wins = [], [], []
    for i, row in enumerate(rows):
        chosen, count = score(model, row["chosen"], pad_id)
        if refs is None:
            losses.append(float(-chosen / count))
        else:
            rejected, _ = score(model, row["rejected"], pad_id)
            rc, rr = [chosen.new_tensor([v]) for v in refs[i]]
            loss, margin = dpo_loss(chosen, rejected, rc, rr, beta)
            losses.append(float(loss))
            margins.append(float(margin))
            likelihood_wins.append(float(chosen > rejected))
    result = {"loss": sum(losses) / len(losses), "examples": len(rows)}
    if refs is not None:
        result.update(
            reward_margin=sum(margins) / len(margins),
            reward_accuracy=sum(x > 0 for x in margins) / len(margins),
            likelihood_preference_accuracy=sum(likelihood_wins) / len(likelihood_wins),
        )
    return result


def fit(model, rows, val, pad_id, out, *, steps, lr, accumulation=4, beta=0.1, dpo=False):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"nonempty stage directory: {out}; use a new run")
    out.mkdir(parents=True, exist_ok=True)
    # Dropout is disabled in both policy and reference, including LoRA (configured at zero).
    model.eval()
    refs = reference_scores(model, rows, pad_id) if dpo else None
    val_refs = reference_scores(model, val, pad_id) if dpo else None
    initial = evaluate(model, val, pad_id, val_refs, beta)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    device = next(model.parameters()).device
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and next(model.parameters()).dtype == torch.float16
    )
    rng = random.Random(1337)
    history = [{"step": 0, "validation": initial}]
    best = initial["loss"]
    model.save_pretrained(out / "best")
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accumulation):
            i = rng.randrange(len(rows))
            row = rows[i]
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()
            ):
                chosen, count = score(model, row["chosen"], pad_id)
                if dpo:
                    rejected, _ = score(model, row["rejected"], pad_id)
                    rc, rr = [chosen.new_tensor([v]) for v in refs[i]]
                    loss, _ = dpo_loss(chosen, rejected, rc, rr, beta)
                else:
                    loss = (-chosen / count).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            total += loss.item() / accumulation
            scaler.scale(loss / accumulation).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        if step % max(1, steps // 10) == 0 or step == steps:
            metrics = evaluate(model, val, pad_id, val_refs, beta)
            history.append({"step": step, "train_loss": total, "validation": metrics})
            print(json.dumps(history[-1]), flush=True)
            if metrics["loss"] < best:
                best = metrics["loss"]
                model.save_pretrained(out / "best")
            model.save_pretrained(out / "last")
            (out / "metrics.json").write_text(json.dumps(history, indent=2))
    assert math.isfinite(best)
    return history
