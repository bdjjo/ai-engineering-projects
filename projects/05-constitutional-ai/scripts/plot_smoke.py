"""Plot engineering diagnostics from a completed synthetic smoke run."""

import argparse
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt  # noqa: E402

from cai.io import read_json  # noqa: E402


def plot(source, out):
    report = read_json(source)
    reward, sft = report["reward"]["history"], report["sft"]["history"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    for split, color in (("train", "#167d9a"), ("validation", "#e07a36")):
        axes[0].plot(
            [r["step"] for r in reward],
            [r[split]["loss"] for r in reward],
            label=split,
            color=color,
            linewidth=2,
        )
    axes[0].set(title="Reward head learns the fixture preference", ylabel="Pairwise loss")
    axes[0].legend(frameon=False)
    axes[1].plot(
        [r["step"] for r in sft], [r["nll"] for r in sft], color="#167d9a", marker="o", linewidth=2
    )
    axes[1].set(title="SFT learns the fixture answer", ylabel="Validation response NLL")
    for ax in axes:
        ax.set_xlabel("Optimizer updates")
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "CPU integration check · random model and scripted feedback\n"
        "These curves do not measure real-world safety",
        fontsize=12,
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="reports/cpu-smoke.json")
    parser.add_argument("--out", default="reports/cpu-training.png")
    args = parser.parse_args()
    plot(args.report, args.out)
