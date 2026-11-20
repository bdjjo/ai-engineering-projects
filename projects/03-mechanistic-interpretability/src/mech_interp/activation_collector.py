"""Bounded activation collection from distinct source splits; no padding/BOS samples."""

import hashlib

import torch


def text_hash(text):
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


@torch.inference_mode()
def collect_split(model, records, n_tokens, hook_name, max_length=128, excluded=None):
    if n_tokens < 2 or not 2 <= max_length <= model.cfg.n_ctx:
        raise ValueError("Invalid token budget or context length")
    blocked = set(excluded or [])
    accepted = set()
    chunks = []
    count = 0
    for record in records:
        text = record["text"].strip()
        key = text_hash(text)
        if len(text) < 80 or key in blocked or key in accepted:
            continue
        # Bound the tokenizer input as well as the forward context.
        tokens = model.to_tokens(text[: max_length * 16], prepend_bos=True)[:, :max_length]
        if tokens.shape[1] < 5:
            continue
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name], return_type=None)
        acts = cache[hook_name][0, 1:].detach().to(device="cpu", dtype=torch.float16)
        take = min(n_tokens - count, len(acts))
        chunks.append(acts[:take].clone())
        accepted.add(key)
        count += take
        if count >= n_tokens:
            break
    if count < n_tokens:
        raise ValueError(f"Source exhausted: collected {count} of {n_tokens} requested tokens")
    result = torch.cat(chunks)
    if not torch.isfinite(result).all():
        raise ValueError("Collected activations contain non-finite values")
    return result, sorted(accepted)


def collect_dataset(
    model,
    provenance,
    train_tokens=8192,
    val_tokens=2048,
    layer=8,
    max_length=128,
    corpus="wikitext",
    dataset_revision=None,
):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    if not 0 <= layer < model.cfg.n_layers:
        raise ValueError("Layer is outside the model")
    hook_name = f"blocks.{layer}.hook_resid_post"
    if corpus == "wikitext":
        dataset_id, subset = "Salesforce/wikitext", "wikitext-2-raw-v1"
        revision = dataset_revision or HfApi().dataset_info(dataset_id).sha
        train = load_dataset(dataset_id, subset, revision=revision, split="train", streaming=True)
        val = load_dataset(
            dataset_id, subset, revision=revision, split="validation", streaming=True
        )
        split_note = "Official train/validation splits; exact normalized-text duplicates excluded"
    elif corpus == "openwebtext":
        dataset_id, subset = "stas/openwebtext-10k", None
        revision = dataset_revision or HfApi().dataset_info(dataset_id).sha

        def partition(validation):
            source = load_dataset(dataset_id, revision=revision, split="train", streaming=True)
            for row in source:
                if (int(text_hash(row["text"])[:8], 16) % 10 == 0) == validation:
                    yield row

        train, val = partition(False), partition(True)
        split_note = "Normalized-text SHA256 mod 10: 0 validation, 1-9 training"
    else:
        raise ValueError("Unknown corpus")
    train_acts, train_hashes = collect_split(model, train, train_tokens, hook_name, max_length)
    print(f"Collected {len(train_acts):,} training tokens", flush=True)
    val_acts, val_hashes = collect_split(
        model, val, val_tokens, hook_name, max_length, excluded=train_hashes
    )
    return {
        "train": train_acts,
        "validation": val_acts,
        "metadata": {
            "model": provenance,
            "hook_name": hook_name,
            "layer": layer,
            "dataset": dataset_id,
            "dataset_config": subset,
            "dataset_revision": revision,
            "split_protocol": split_note,
            "train_text_hashes": train_hashes,
            "validation_text_hashes": val_hashes,
            "train_tokens": len(train_acts),
            "validation_tokens": len(val_acts),
            "max_length": max_length,
            "storage_dtype": "float16",
            "sampling": "Sequential nonempty text records, prefix only; BOS excluded",
        },
    }
