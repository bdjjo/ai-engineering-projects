"""Split before generation; reject overlap and keep failed feedback out of training."""

import unicodedata

from cai.constitution import get_principle
from cai.io import digest, read_json


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def validate_prompts(rows):
    if not rows:
        raise ValueError("Prompt dataset is empty")
    ids, texts, families = set(), set(), {}
    for row in rows:
        if not isinstance(row.get("id"), str) or not row["id"].strip() or row["id"] in ids:
            raise ValueError("Prompt IDs must be unique nonempty strings")
        if row.get("split") not in {"train", "validation", "test"}:
            raise ValueError("Every prompt needs a train, validation, or test split")
        if row.get("kind") not in {"adversarial", "benign"}:
            raise ValueError("Unknown prompt kind")
        get_principle(row["principle_id"])
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError("Prompt text cannot be empty")
        text = normalized(row["prompt"])
        if text in texts:
            raise ValueError("Duplicate normalized prompt, possibly leaking across splits")
        family = row.get("family", row["id"])
        if family in families and families[family] != row["split"]:
            raise ValueError("A prompt family cannot cross splits")
        families[family] = row["split"]
        ids.add(row["id"])
        texts.add(text)
    return rows


def accepted_pairs(records):
    pairs = []
    for row in records:
        if row["status"] != "accepted":
            continue
        if row["split"] == "test":
            raise ValueError("Test data must never become training preferences")
        original, revised = row["original"], row["revised"]
        if normalized(original) == normalized(revised):
            raise ValueError("Identical responses cannot form a strict preference")
        winner = row["judgment"]["winner"]
        if winner not in {"original", "revised"}:
            raise ValueError("Accepted pairs need a strict preference")
        pairs.append(
            {
                **{k: row[k] for k in ("id", "prompt", "split", "kind", "principle_id")},
                "chosen": row[winner],
                "rejected": row["original" if winner == "revised" else "revised"],
                "winner": winner,
                "record_sha256": digest(row),
            }
        )
    return pairs


def partition_pairs(pairs):
    if not pairs:
        raise ValueError("No accepted preferences; inspect the generation audit before training")
    validate_prompts(pairs)
    if any(p["split"] == "test" for p in pairs):
        raise ValueError("Test preferences are not allowed in model fitting")
    for p in pairs:
        if not all(isinstance(p.get(k), str) and p[k].strip() for k in ("chosen", "rejected")):
            raise ValueError("Preference responses cannot be empty")
        if normalized(p["chosen"]) == normalized(p["rejected"]):
            raise ValueError("Identical preference responses")
    train = [p for p in pairs if p["split"] == "train"]
    validation = [p for p in pairs if p["split"] == "validation"]
    if not train or not validation:
        raise ValueError("Need accepted preferences in BOTH train and validation splits")
    return train, validation


def load_feedback_bundle(path, rows, config):
    """Explicitly import earlier feedback with immutable upstream provenance."""
    from pathlib import Path

    path = Path(path)
    manifest = read_json(path / "manifest.json")
    if manifest["config"] != config or manifest["data_sha256"] != digest(rows):
        raise ValueError("Imported feedback config or prompt split differs from this run")
    records = read_json(path / "feedback/records.json")
    expected = [row for row in rows if row["split"] != "test"]
    if len(records) != len(expected) or any(
        any(record.get(k) != value for k, value in row.items())
        for record, row in zip(records, expected, strict=True)
    ):
        raise ValueError("Imported feedback does not cover the expected prompts in order")
    pairs = read_json(path / "feedback/pairs.json")
    if accepted_pairs(records) != pairs:
        raise ValueError("Imported preference pairs do not match their generation records")
    audit = read_json(path / "feedback/audit.json")
    lineage = {
        "manifest": manifest,
        "records_sha256": digest(records),
        "pairs_sha256": digest(pairs),
        "audit_sha256": digest(audit),
    }
    return lineage, {"records.json": records, "pairs.json": pairs, "audit.json": audit}
