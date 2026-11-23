"""Export a small, auditable result bundle, excluding model weights and environment files."""

import argparse
import shutil
from pathlib import Path

from cai.io import fresh_directory, read_json, write_json


def export_report(run, out):
    run, out = Path(run), fresh_directory(out)
    names = [
        "manifest.json",
        "prompts.json",
        "feedback/audit.json",
        "feedback/records.json",
        "feedback/pairs.json",
        "reward/summary.json",
        "sft/summary.json",
        "rl/summary.json",
        "evaluation-sft/summary.json",
        "evaluation-sft/responses.json",
        "evaluation-rl/summary.json",
        "evaluation-rl/responses.json",
    ]
    copied = []
    for name in names:
        source = run / name
        if source.exists():
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied.append(name)
    if "manifest.json" not in copied:
        raise ValueError("Run manifest is missing")
    feedback = read_json(run / "feedback/audit.json") if "feedback/audit.json" in copied else None
    index = {
        "files": copied,
        "feedback": feedback,
        "training_complete": all(
            f"{stage}/summary.json" in copied for stage in ("reward", "sft", "rl")
        ),
        "evaluation_complete": all(
            f"evaluation-{stage}/summary.json" in copied for stage in ("sft", "rl")
        ),
        "interpretation": "Completion and learned reward do not establish alignment improvement.",
    }
    write_json(out / "index.json", index)
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(export_report(args.run, args.out))
