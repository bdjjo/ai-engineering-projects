"""Render held-out SAE diagnostics from a recorded training history."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", default="outputs/sae/history.json")
    parser.add_argument("--out", default="outputs/sae/training.png")
    args = parser.parse_args()
    rows = json.loads(Path(args.history).read_text())
    steps = [r["step"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for ax, key, label in zip(
        axes,
        ["mse_per_dimension", "l0", "variance_explained"],
        ["Reconstruction MSE", "Active features per token (L0)", "Variance explained"],
    ):
        ax.plot(steps, [r["validation"][key] for r in rows], color="#2356a8", marker=".")
        ax.set(xlabel="Training update", ylabel=label)
        ax.grid(alpha=0.2)
    fig.suptitle("Held-out SAE diagnostics · fixed training budget")
    fig.savefig(args.out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
