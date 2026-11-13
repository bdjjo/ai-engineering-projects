"""Character encoding and a small, actual byte-pair encoder written from first principles.

BPE starts with all 256 UTF-8 byte values and repeatedly merges the most frequent
adjacent pair. A simple, documented regex prevents merges across word boundaries.
It is deliberately not compatible with the original GPT-2 tokenizer vocabulary.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Protocol

PRETOKEN_PATTERN = r"\s+|\w+|[^\w\s]+"
PRETOKENIZER = re.compile(PRETOKEN_PATTERN, flags=re.UNICODE)


class Tokenizer(Protocol):
    @property
    def vocab_size(self) -> int: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def to_dict(self) -> dict: ...


class CharTokenizer:
    """A transparent baseline. Fit only on training text; reject unseen characters."""

    def __init__(self, text: str) -> None:
        if not text:
            raise ValueError("cannot fit a character vocabulary on empty text")
        self.characters = sorted(set(text))
        self.stoi = {char: idx for idx, char in enumerate(self.characters)}

    @property
    def vocab_size(self) -> int:
        return len(self.characters)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[char] for char in text]
        except KeyError as exc:
            raise ValueError(
                f"unknown character {exc.args[0]!r}; use BPE for arbitrary UTF-8"
            ) from exc

    def decode(self, ids: list[int]) -> str:
        if any(idx < 0 or idx >= self.vocab_size for idx in ids):
            raise ValueError("token ID outside the vocabulary")
        return "".join(self.characters[idx] for idx in ids)

    def to_dict(self) -> dict:
        return {"version": 1, "kind": "char", "characters": self.characters}


def merge_pair(ids: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    """Replace nonoverlapping occurrences, from left to right (aaa -> [aa, a])."""
    result: list[int] = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return tuple(result)


class BPETokenizer:
    """Byte BPE with deterministic tie-breaking and no external tokenizer library."""

    def __init__(self, merges: list[tuple[int, int]] | None = None) -> None:
        self.merges = list(merges or [])
        self.vocab = {i: bytes([i]) for i in range(256)}
        for new_id, pair in enumerate(self.merges, start=256):
            if pair[0] not in self.vocab or pair[1] not in self.vocab:
                raise ValueError("merge must refer only to earlier tokens")
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def fit(cls, text: str, vocab_size: int = 512) -> BPETokenizer:
        if not text or vocab_size < 256:
            raise ValueError("BPE needs nonempty training text and vocab_size >= 256")
        # Counting distinct pieces once makes the educational algorithm practical.
        pieces: Counter[tuple[int, ...]] = Counter(
            tuple(piece.encode("utf-8")) for piece in PRETOKENIZER.findall(text)
        )
        merges: list[tuple[int, int]] = []
        for new_id in range(256, vocab_size):
            counts: Counter[tuple[int, int]] = Counter()
            for piece, frequency in pieces.items():
                for pair in zip(piece, piece[1:], strict=False):
                    counts[pair] += frequency
            if not counts:
                break
            # Most frequent; lexicographically smallest pair breaks ties reproducibly.
            pair = min(counts, key=lambda p: (-counts[p], p))
            if counts[pair] < 2:
                break
            merges.append(pair)
            updated: Counter[tuple[int, ...]] = Counter()
            for piece, frequency in pieces.items():
                updated[merge_pair(piece, pair, new_id)] += frequency
            pieces = updated
        return cls(merges)

    def encode(self, text: str) -> list[int]:
        cache: dict[str, tuple[int, ...]] = {}
        result: list[int] = []
        for piece in PRETOKENIZER.findall(text):
            if piece not in cache:
                ids = tuple(piece.encode("utf-8"))
                # Apply learned merges in their training order, never refit at inference.
                for new_id, pair in enumerate(self.merges, start=256):
                    ids = merge_pair(ids, pair, new_id)
                    if len(ids) < 2:
                        break
                cache[piece] = ids
            result.extend(cache[piece])
        return result

    def decode(self, ids: list[int]) -> str:
        try:
            raw = b"".join(self.vocab[idx] for idx in ids)
        except KeyError as exc:
            raise ValueError("token ID outside the vocabulary") from exc
        # Arbitrary sampled bytes may not form valid UTF-8; exact encode/decode does.
        return raw.decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "kind": "bpe",
            "pattern": PRETOKEN_PATTERN,
            "merges": [list(pair) for pair in self.merges],
        }


def tokenizer_from_dict(payload: dict) -> Tokenizer:
    if payload.get("version") != 1:
        raise ValueError("unsupported tokenizer version")
    if payload["kind"] == "char":
        chars = payload["characters"]
        if not chars or chars != sorted(set(chars)) or any(len(c) != 1 for c in chars):
            raise ValueError("invalid character vocabulary")
        return CharTokenizer("".join(chars))
    if payload["kind"] == "bpe":
        if payload.get("pattern") != PRETOKEN_PATTERN:
            raise ValueError("unsupported BPE pretokenization pattern")
        return BPETokenizer([tuple(pair) for pair in payload["merges"]])
    raise ValueError("unknown tokenizer kind")


def save_tokenizer(tokenizer: Tokenizer, path: str | Path) -> None:
    Path(path).write_text(json.dumps(tokenizer.to_dict(), indent=2), encoding="utf-8")


def load_tokenizer(path: str | Path) -> Tokenizer:
    return tokenizer_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
