"""Memmapped token blocks and deterministic, disjoint rank batches."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class TokenDataset:
    def __init__(self, path, block_size, vocab_size):
        path = Path(path)
        if block_size < 1 or path.stat().st_size % 2:
            raise ValueError("Invalid block size or truncated uint16 token file")
        self.tokens = np.memmap(path, dtype="<u2", mode="r")
        self.block_size = block_size
        if len(self) < 1:
            raise ValueError("Dataset is too small for one input/target block")
        for offset in range(0, len(self.tokens), 1_000_000):
            if self.tokens[offset : offset + 1_000_000].max() >= vocab_size:
                raise ValueError("Token ID exceeds model vocabulary")

    def __len__(self):
        return (len(self.tokens) - 1) // self.block_size

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.block_size
        chunk = torch.from_numpy(self.tokens[start : start + self.block_size + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]


class RankBatchStream:
    def __init__(self, dataset, batch_size, rank=0, world_size=1, seed=1337):
        if batch_size < 1 or world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid batch size or distributed rank")
        self.dataset, self.batch_size = dataset, batch_size
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.batches_per_epoch = len(dataset) // (batch_size * world_size)
        if not self.batches_per_epoch:
            raise ValueError("Dataset cannot supply one complete global batch")
        self.epoch = None
        self.order = None

    def indices(self, microstep):
        if microstep < 0:
            raise ValueError("Negative microstep")
        epoch, within_epoch = divmod(microstep, self.batches_per_epoch)
        if epoch != self.epoch:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            self.order = torch.randperm(len(self.dataset), generator=generator)
            self.epoch = epoch
        start = (within_epoch * self.world_size + self.rank) * self.batch_size
        return self.order[start : start + self.batch_size].tolist()

    def batch(self, microstep, device):
        examples = [self.dataset[i] for i in self.indices(microstep)]
        x = torch.stack([pair[0] for pair in examples]).to(device)
        y = torch.stack([pair[1] for pair in examples]).to(device)
        return x, y


def prepare_synthetic(out, train_tokens=16385, val_tokens=4097, vocab_size=128, seed=1337):
    """Predictable token cycles for systems tests; not natural-language training data."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose a new synthetic data directory")
    if not 2 <= vocab_size <= 65536 or min(train_tokens, val_tokens) < 2:
        raise ValueError("Invalid synthetic data settings")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    hashes = {}
    for split, length in [("train", train_tokens), ("validation", val_tokens)]:
        starts = rng.integers(0, vocab_size, size=(length + 31) // 32)
        ids = ((starts[:, None] + np.arange(32)) % vocab_size).reshape(-1)[:length]
        path = out / f"{split}.bin"
        ids.astype("<u2").tofile(path)
        hashes[split] = file_hash(path)
    manifest = {
        "source": "Synthetic modular token sequences; systems validation only",
        "seed": seed,
        "vocab_size": vocab_size,
        "dtype": "little-endian uint16",
        "train_tokens": train_tokens,
        "validation_tokens": val_tokens,
        "sha256": hashes,
    }
    write_json(out / "manifest.json", manifest)
    return manifest
