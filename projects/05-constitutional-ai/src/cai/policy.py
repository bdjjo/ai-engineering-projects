"""Response-only SFT and on-policy REINFORCE with a frozen reference penalty."""

import copy
import random

import torch

from cai.io import fresh_directory, write_json
from cai.modeling import encode_response, generate, pad_batch, sequence_logps
from cai.reward import score_texts


def reinforce_loss(logps, ref_logps, rewards, baseline, kl_coef):
    if kl_coef < 0 or not (logps.shape == ref_logps.shape == rewards.shape):
        raise ValueError("Invalid KL coefficient or unaligned rollout scores")
    sampled_kl = logps.detach() - ref_logps.detach()
    utility = rewards.detach() - kl_coef * sampled_kl
    advantage = utility - baseline
    return -(advantage.detach() * logps).mean(), utility, sampled_kl


@torch.no_grad()
def sft_metrics(model, rows, pad_id):
    model.eval()
    total_logp, total_tokens = 0.0, 0
    for row in rows:
        batch = pad_batch([row], pad_id, next(model.parameters()).device)
        logps, counts = sequence_logps(model, batch)
        total_logp += logps.sum().item()
        total_tokens += counts.sum().item()
    return {"nll": -total_logp / total_tokens, "response_tokens": total_tokens}


def fit_sft(model, tokenizer, train, validation, out, config):
    out = fresh_directory(out)
    device = next(model.parameters()).device

    def encode(pairs):
        return [
            encode_response(tokenizer, p["prompt"], p["chosen"], config["max_length"])
            for p in pairs
        ]

    train_rows, val_rows = encode(train), encode(validation)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config["sft_lr"]
    )
    rng = random.Random(config["seed"])
    history, best, best_step = [], float("inf"), None
    for step in range(config["sft_steps"] + 1):
        if step:
            # eval disables dropout; gradients remain enabled for a deterministic objective.
            model.eval()
            rows = [rng.choice(train_rows) for _ in range(config["batch_size"])]
            logps, counts = sequence_logps(model, pad_batch(rows, tokenizer.pad_token_id, device))
            loss = -logps.sum() / counts.sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if step % config["eval_every"] == 0 or step == config["sft_steps"]:
            metrics = sft_metrics(model, val_rows, tokenizer.pad_token_id)
            history.append({"step": step, **metrics})
            if metrics["nll"] < best:
                best, best_step = metrics["nll"], step
                model.save_pretrained(out / "best")
    summary = {
        "selected_step": best_step,
        "validation_nll": best,
        "history": history,
        "objective": "response-only supervised fine-tuning on AI-preferred answers",
    }
    tokenizer.save_pretrained(out / "best")
    write_json(out / "summary.json", summary)
    return summary


@torch.no_grad()
def policy_reward(model, tokenizer, reward, rows, config, normalization):
    scores, truncated = [], 0
    for i, row in enumerate(rows):
        rollout = generate(
            model,
            tokenizer,
            row["prompt"],
            config["rollout_tokens"],
            seed=config["seed"] + i,
            max_context=config["max_context"],
        )
        truncated += rollout["hit_token_limit"]
        score = score_texts(
            reward,
            tokenizer,
            [row["prompt"]],
            [rollout["text"]],
            config["max_length"],
            normalization,
        )
        scores.append(score.item())
    return {"reward": sum(scores) / len(scores), "count": len(scores), "truncated": truncated}


def fit_rl(model, tokenizer, reward, train, validation, out, config, normalization):
    out = fresh_directory(out)
    reference = copy.deepcopy(model).requires_grad_(False).eval()
    reward.requires_grad_(False).eval()
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config["rl_lr"]
    )
    rng = random.Random(config["seed"])
    history, baseline, best, best_step = [], 0.0, -float("inf"), None
    for step in range(config["rl_steps"] + 1):
        update = {}
        if step:
            prompts = [rng.choice(train)["prompt"] for _ in range(config["batch_size"])]
            rollouts = [
                generate(
                    model,
                    tokenizer,
                    p,
                    config["rollout_tokens"],
                    seed=config["seed"] + step * config["batch_size"] + i,
                    sample=True,
                    max_context=config["max_context"],
                )
                for i, p in enumerate(prompts)
            ]
            batch = pad_batch(rollouts, tokenizer.pad_token_id, device)
            with torch.no_grad():
                ref_logps, _ = sequence_logps(reference, batch)
                scores = score_texts(
                    reward,
                    tokenizer,
                    prompts,
                    [r["text"] for r in rollouts],
                    config["max_length"],
                    normalization,
                )
            model.eval()
            logps, _ = sequence_logps(model, batch)
            loss, utility, sampled_kl = reinforce_loss(
                logps, ref_logps, scores, baseline, config["kl_coef"]
            )
            if not torch.isfinite(loss):
                raise ValueError("Non-finite policy loss")
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            if not torch.isfinite(torch.tensor(grad_norm)):
                raise ValueError("Non-finite policy gradient")
            optimizer.step()
            # The current rollout never contributes to its own baseline.
            baseline = 0.9 * baseline + 0.1 * utility.mean().item()
            update = {
                "loss": loss.item(),
                "sampled_sequence_kl": sampled_kl.mean().item(),
                "reward": scores.mean().item(),
                "baseline_after": baseline,
                "gradient_norm": grad_norm,
                "truncated": sum(r["hit_token_limit"] for r in rollouts),
            }
        entry = {"step": step, **update}
        if step % config["eval_every"] == 0 or step == config["rl_steps"]:
            metrics = policy_reward(model, tokenizer, reward, validation, config, normalization)
            entry["validation"] = metrics
            if metrics["reward"] > best:
                best, best_step = metrics["reward"], step
                model.save_pretrained(out / "best")
        history.append(entry)
        write_json(out / "history.json", history)
    summary = {
        "selected_step": best_step,
        "selection_metric": "validation learned reward",
        "validation_reward": best,
        "history": history,
        "objective": "on-policy sequence REINFORCE with sampled reference KL penalty",
        "warning": "Learned reward is a proxy, not evidence of safety or helpfulness.",
    }
    tokenizer.save_pretrained(out / "best")
    write_json(out / "summary.json", summary)
    return summary
