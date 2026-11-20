"""Bidirectional head-output patching on position-aligned IOI prompt pairs.

This is an activation-patching study, not a replication of the full IOI circuit.
"""

from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import partial

import torch


@dataclass(frozen=True)
class IOIPair:
    clean: str
    corrupt: str
    recipient: str
    subject: str


def make_prompts():
    pairs = []
    for a, b in [("Mary", "John"), ("Sarah", "Tom"), ("Emma", "James"), ("Alice", "Bob")]:
        for place, item in [("store", "drink"), ("park", "book")]:
            for first, second in [(a, b), (b, a)]:
                pairs.append(
                    IOIPair(
                        clean=f"When {first} and {second} went to the {place}, "
                        f"{second} gave a {item} to",
                        corrupt=f"When {second} and {first} went to the {place}, "
                        f"{first} gave a {item} to",
                        recipient=first,
                        subject=second,
                    )
                )
    return pairs


def logit_difference(logits, answer_ids):
    """Per-prompt logit(recipient) - logit(subject), fixed across both conditions."""
    if answer_ids.ndim != 2 or answer_ids.shape[1] != 2:
        raise ValueError("answer_ids must have shape [batch, 2]")
    if torch.any(answer_ids[:, 0] == answer_ids[:, 1]):
        raise ValueError("Recipient and subject must be different tokens")
    last = logits[:, -1, :]
    chosen = last.gather(-1, answer_ids.to(last.device))
    return chosen[:, 0] - chosen[:, 1]


def patch_head(value, hook, *, source, head):
    result = value.clone()
    result[:, :, head, :] = source[hook.name][:, :, head, :]
    return result


def prepare_batches(model, prompts, batch_size):
    """Bucket by token length; no padding and no reading a padding position as END."""
    if batch_size < 1 or not prompts:
        raise ValueError("Need a positive batch size and at least one prompt")
    buckets = defaultdict(list)
    for index, pair in enumerate(prompts):
        clean = model.to_tokens(pair.clean, prepend_bos=True)
        corrupt = model.to_tokens(pair.corrupt, prepend_bos=True)
        if clean.shape != corrupt.shape:
            raise ValueError("Clean/corrupt token positions are not aligned")
        answer = [
            model.to_single_token(" " + pair.recipient),
            model.to_single_token(" " + pair.subject),
        ]
        if answer[0] == answer[1]:
            raise ValueError("Answers must be distinct single tokens")
        buckets[clean.shape[1]].append((index, clean, corrupt, answer))
    for records in buckets.values():
        for offset in range(0, len(records), batch_size):
            chunk = records[offset : offset + batch_size]
            yield (
                [r[0] for r in chunk],
                torch.cat([r[1] for r in chunk]),
                torch.cat([r[2] for r in chunk]),
                torch.tensor([r[3] for r in chunk], device=chunk[0][1].device),
            )


@torch.inference_mode()
def run_activation_patching(model, prompts=None, batch_size=4):
    prompts = prompts or make_prompts()
    n = len(prompts)
    shape = (model.cfg.n_layers, model.cfg.n_heads, n)
    damage = torch.empty(shape)
    recovery = torch.empty(shape)
    clean_scores, corrupt_scores = torch.empty(n), torch.empty(n)
    control_errors = []
    for indices, clean, corrupt, answers in prepare_batches(model, prompts, batch_size):
        clean_logits, clean_cache = model.run_with_cache(
            clean, names_filter=lambda name: name.endswith("attn.hook_z")
        )
        clean_score = logit_difference(clean_logits, answers)
        del clean_logits
        corrupt_logits, corrupt_cache = model.run_with_cache(
            corrupt, names_filter=lambda name: name.endswith("attn.hook_z")
        )
        corrupt_score = logit_difference(corrupt_logits, answers)
        del corrupt_logits
        clean_scores[indices] = clean_score.cpu()
        corrupt_scores[indices] = corrupt_score.cpu()
        # Identical-source patch must reproduce the clean run.
        self_logits = model.run_with_hooks(
            clean,
            fwd_hooks=[("blocks.0.attn.hook_z", partial(patch_head, source=clean_cache, head=0))],
        )
        error = (logit_difference(self_logits, answers) - clean_score).abs().max().item()
        control_errors.append(error)
        if error > 1e-4:
            raise RuntimeError(f"Self-patch control failed: {error}")
        del self_logits
        for layer in range(model.cfg.n_layers):
            name = f"blocks.{layer}.attn.hook_z"
            for head in range(model.cfg.n_heads):
                logits = model.run_with_hooks(
                    clean, fwd_hooks=[(name, partial(patch_head, source=corrupt_cache, head=head))]
                )
                damage[layer, head, indices] = (
                    clean_score - logit_difference(logits, answers)
                ).cpu()
                del logits
                logits = model.run_with_hooks(
                    corrupt, fwd_hooks=[(name, partial(patch_head, source=clean_cache, head=head))]
                )
                recovery[layer, head, indices] = (
                    logit_difference(logits, answers) - corrupt_score
                ).cpu()
                del logits
            print(
                f"Batch {indices[0] + 1}-{indices[-1] + 1}: layer {layer + 1} complete", flush=True
            )
        del clean_cache, corrupt_cache
    gap = (clean_scores - corrupt_scores).mean().item()
    mean_damage = damage.mean(-1)
    ranking = []
    for flat in mean_damage.flatten().argsort(descending=True)[:10].tolist():
        layer, head = divmod(flat, model.cfg.n_heads)
        ranking.append(
            {
                "layer": layer,
                "head": head,
                "damage": mean_damage[layer, head].item(),
                "recovery": recovery[layer, head].mean().item(),
            }
        )
    return {
        "protocol": "All-position z patching; swapped-name corruption; fixed clean answer IDs",
        "prompts": [asdict(pair) for pair in prompts],
        "n_prompts": n,
        "clean_logit_diff": clean_scores.mean().item(),
        "corrupt_logit_diff": corrupt_scores.mean().item(),
        "clean_pairwise_accuracy": (clean_scores > 0).float().mean().item(),
        "baseline_gap": gap,
        "self_patch_max_error": max(control_errors),
        "damage": mean_damage.tolist(),
        "recovery": recovery.mean(-1).tolist(),
        "normalized_recovery": (recovery.mean(-1) / gap).tolist() if gap > 1e-6 else None,
        "per_prompt_clean": clean_scores.tolist(),
        "per_prompt_corrupt": corrupt_scores.tolist(),
        "per_prompt_damage": damage.tolist(),
        "per_prompt_recovery": recovery.tolist(),
        "top_heads_by_damage": ranking,
    }


def plot_effects(result, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    arrays = [np.array(result[key]) for key in ("damage", "recovery")]
    bound = max(float(abs(a).max()) for a in arrays) or 1.0
    for ax, array, title in zip(
        axes, arrays, ["Corrupt → clean: damage", "Clean → corrupt: recovery"]
    ):
        plot = ax.imshow(array, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
        ax.set(xlabel="Head (zero-based)", ylabel="Layer (zero-based)", title=title)
        ax.set_xticks(range(array.shape[1]))
        ax.set_yticks(range(array.shape[0]))
    fig.colorbar(plot, ax=axes, label="Logit difference change; positive = larger effect")
    fig.savefig(path, dpi=160)
    plt.close(fig)
