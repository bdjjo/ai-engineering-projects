"""Export small auditable metrics and a plot, without weights or token files."""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from distributed_gpt.data import write_json  # noqa: E402


def export(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Report destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        key: json.loads((source / f"{key}.json").read_text())
        for key in ("metadata", "summary", "history")
    }
    if not payload["summary"]["completed"]:
        raise ValueError("Cannot export an incomplete run")
    write_json(destination / "results.json", payload)
    history = payload["history"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    steps = [row["step"] for row in history]
    axes[0].plot(steps, [row["loss"] for row in history], color="#176b75", linewidth=2)
    axes[0].set(xlabel="Optimizer attempt", ylabel="Training cross-entropy (nats)")
    axes[1].plot(steps, [row["tokens_per_second"] for row in history], color="#a64b26")
    axes[1].set(xlabel="Optimizer attempt", ylabel="Global tokens / second")
    meta = payload["metadata"]
    hardware = meta["gpu_name"] or "CPU"
    fig.suptitle(f"{meta['strategy'].upper()} · {meta['world_size']} rank(s) · {hardware}")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(destination / "training.png", dpi=150)
    plt.close(fig)
    if (source / "profiler.txt").exists():
        shutil.copyfile(source / "profiler.txt", destination / "profiler.txt")
    return payload["summary"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.run, args.out), indent=2))
