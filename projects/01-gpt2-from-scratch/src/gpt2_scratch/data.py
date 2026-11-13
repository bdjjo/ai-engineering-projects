"""Leakage-aware data preparation and vectorized next-token batches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from gpt2_scratch.tokenizer import BPETokenizer, CharTokenizer, load_tokenizer, save_tokenizer


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_text(
    text: str, train_fraction: float = 0.9, val_fraction: float = 0.05
) -> dict[str, str]:
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1 - train_fraction:
        raise ValueError("train, validation, and test fractions must all be positive")
    train_end = int(len(text) * train_fraction)
    val_end = int(len(text) * (train_fraction + val_fraction))
    splits = {"train": text[:train_end], "val": text[train_end:val_end], "test": text[val_end:]}
    if any(len(part) < 2 for part in splits.values()):
        raise ValueError("each split needs at least two characters")
    return splits


def prepare_data(
    input_path: str | Path, output_dir: str | Path, kind: str = "char", vocab_size: int = 512
) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty; choose a new directory")
    raw = Path(input_path).read_bytes()
    text = raw.decode("utf-8")  # Read bytes to avoid platform-dependent newline translation.
    splits = split_text(text)
    if kind == "char":
        tokenizer = CharTokenizer(splits["train"])
    elif kind == "bpe":
        tokenizer = BPETokenizer.fit(splits["train"], vocab_size)
    else:
        raise ValueError("tokenizer must be char or bpe")
    arrays = {
        name: np.asarray(tokenizer.encode(part), dtype=np.int32) for name, part in splits.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_tokenizer(tokenizer, output_dir / "tokenizer.json")
    manifest = {
        "format_version": 1,
        "source_name": Path(input_path).name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw),
        "source_characters": len(text),
        "tokenizer": kind,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_sha256": sha256_file(output_dir / "tokenizer.json"),
        "split_policy": "contiguous 90/5/5 characters; fit tokenizer on train only",
        "splits": {},
    }
    offset = 0
    for name, part in splits.items():
        np.save(output_dir / f"{name}.npy", arrays[name], allow_pickle=False)
        manifest["splits"][name] = {
            "start_character": offset,
            "end_character": offset + len(part),
            "characters": len(part),
            "bytes": len(part.encode("utf-8")),
            "tokens": len(arrays[name]),
            "characters_per_token": len(part) / len(arrays[name]),
            "array_sha256": sha256_file(output_dir / f"{name}.npy"),
        }
        offset += len(part)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_prepared(directory: str | Path, splits: tuple[str, ...] = ("train", "val")) -> tuple:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format_version"] != 1:
        raise ValueError("unsupported data format")
    if sha256_file(directory / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise ValueError("tokenizer hash does not match manifest")
    tokenizer = load_tokenizer(directory / "tokenizer.json")
    arrays = {}
    for split in splits:
        path = directory / f"{split}.npy"
        if sha256_file(path) != manifest["splits"][split]["array_sha256"]:
            raise ValueError(f"{split} array hash does not match manifest")
        array = np.load(path, allow_pickle=False)
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer) or len(array) < 2:
            raise ValueError("token array must be a one-dimensional integer array")
        if array.min() < 0 or array.max() >= tokenizer.vocab_size:
            raise ValueError("token array contains an invalid token")
        arrays[split] = torch.from_numpy(array.astype(np.int64))
    return tokenizer, arrays, manifest


def sample_batch(
    tokens: torch.Tensor,
    batch_size: int,
    block_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(tokens) <= block_size:
        raise ValueError("split must contain more tokens than block_size")
    starts = torch.randint(len(tokens) - block_size, (batch_size,), generator=generator)
    positions = starts[:, None] + torch.arange(block_size)[None, :]
    return tokens[positions].to(device), tokens[positions + 1].to(device)
