"""Aggregate complete blinded human ratings; ties count as half a win."""

import argparse
import json
import random
from pathlib import Path


def summarize(ratings, key):
    if not ratings or len({r["id"] for r in ratings}) != len(ratings):
        raise ValueError("ratings must be nonempty with unique IDs")
    if {r["id"] for r in ratings} != set(key):
        raise ValueError("rate every generated comparison exactly once")
    values = []
    for r in ratings:
        if r["winner"] not in ("A", "B", "tie"):
            raise ValueError("each winner must be A, B, or tie")
        values.append(
            0.5
            if r["winner"] == "tie"
            else float(key[r["id"]][0 if r["winner"] == "A" else 1] == "dpo")
        )
    rng = random.Random(1337)
    boot = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(2000))
    return {
        "examples": len(values),
        "dpo_wins": values.count(1.0),
        "sft_wins": values.count(0.0),
        "ties": values.count(0.5),
        "tie_adjusted_dpo_win_rate": sum(values) / len(values),
        "bootstrap_95_percent_interval": [boot[49], boot[1949]],
        "note": "Prompt bootstrap; does not capture rater bias or disagreement.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ratings", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    result = summarize(json.loads(Path(a.ratings).read_text()), json.loads(Path(a.key).read_text()))
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
