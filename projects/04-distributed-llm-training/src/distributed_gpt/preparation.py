"""Bounded streaming tokenization with explicit dataset/tokenizer provenance."""

import hashlib
from pathlib import Path

import numpy as np

from .data import file_hash, write_json


def normalized_hash(text):
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def prepare_text(
    out,
    source="wikitext",
    train_tokens=131072,
    val_tokens=16384,
    dataset_revision=None,
    tokenizer_revision=None,
):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose a new data directory")
    if min(train_tokens, val_tokens) < 2:
        raise ValueError("Token budgets must be at least two")
    tokenizer_id = "openai-community/gpt2"
    tokenizer_revision = tokenizer_revision or HfApi().model_info(tokenizer_id).sha
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=tokenizer_revision)
    if len(tokenizer) > 65536:
        raise ValueError("Tokenizer vocabulary does not fit uint16")
    dataset_id = "Salesforce/wikitext" if source == "wikitext" else "stas/openwebtext-10k"
    subset = "wikitext-2-raw-v1" if source == "wikitext" else None
    dataset_revision = dataset_revision or HfApi().dataset_info(dataset_id).sha
    out.mkdir(parents=True, exist_ok=True)
    seen = set()
    statistics = {}
    for split, budget in [("train", train_tokens), ("validation", val_tokens)]:
        source_split = split if source == "wikitext" else "train"
        records = load_dataset(
            dataset_id, subset, revision=dataset_revision, split=source_split, streaming=True
        )
        written, hashes = 0, []
        path = out / f"{split}.bin"
        with path.open("wb") as handle:
            for record in records:
                text = record["text"].strip()[:50000]
                key = normalized_hash(text)
                if len(text) < 80 or key in seen:
                    continue
                if source == "openwebtext":
                    if (int(key[:8], 16) % 10 == 0) != (split == "validation"):
                        continue
                ids = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
                ids = ids[: budget - written]
                np.asarray(ids, dtype="<u2").tofile(handle)
                written += len(ids)
                hashes.append(key)
                seen.add(key)
                if written == budget:
                    break
        if written < budget:
            raise ValueError(f"{split} exhausted after {written} of {budget} requested tokens")
        statistics[split] = {
            "tokens": written,
            "records": len(hashes),
            "text_sha256": hashes,
            "file_sha256": file_hash(path),
        }
        print(f"Prepared {split}: {written:,} tokens", flush=True)
    manifest = {
        "dataset": dataset_id,
        "dataset_config": subset,
        "dataset_revision": dataset_revision,
        "tokenizer": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "vocab_size": len(tokenizer),
        "dtype": "little-endian uint16",
        "split_protocol": "Official splits"
        if source == "wikitext"
        else "Text SHA256 mod 10: bucket 0 validation, others training",
        "processing": "Sequential records, minimum 80 characters, at most 50000 characters "
        "per record, EOS separators, exact normalized-text duplicates excluded",
        "splits": statistics,
    }
    write_json(out / "manifest.json", manifest)
    return manifest
