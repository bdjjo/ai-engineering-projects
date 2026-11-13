"""Export measured run evidence, fixed-seed samples, and publication-ready plots."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".local" / "matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gpt2_scratch.generation import generate_text, load_model  # noqa: E402
from gpt2_scratch.runtime import configure_runtime, resolve_device, write_json  # noqa: E402

SAMPLE_PLAN = [
    {"prompt": "ROMEO:\n", "seed": 42, "temperature": 0.8, "top_k": 40, "tokens": 300},
    {"prompt": "First Citizen:\n", "seed": 43, "temperature": 0.8, "top_k": 40, "tokens": 300},
    {"prompt": "To be, or not to be", "seed": 44, "temperature": 0.8, "top_k": 40, "tokens": 300},
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def style_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#25354A",
            "text.color": "#152A42",
            "xtick.color": "#435167",
            "ytick.color": "#435167",
            "axes.titleweight": "bold",
            "figure.facecolor": "#F7F9FC",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#D2DAE6",
            "grid.color": "#E7ECF2",
            "grid.linewidth": 0.7,
            "savefig.facecolor": "#F7F9FC",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    configure_runtime(1337, 4)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    runs = []
    table = []
    for path in map(Path, args.runs):
        destination = output / path.name
        destination.mkdir(exist_ok=True)
        summary = read_json(path / "summary.json")
        if summary["status"] != "complete":
            raise ValueError(f"{path.name} is not complete")
        for name in (
            "metadata.json",
            "summary.json",
            "metrics.jsonl",
            "tokenizer.json",
            "evaluation-both.json",
        ):
            shutil.copyfile(path / name, destination / name)
        metadata = read_json(path / "metadata.json")
        evaluation = read_json(path / "evaluation-both.json")
        events = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
        model, tokenizer, checkpoint = load_model(path / "best.pt", resolve_device("cpu"))
        samples = []
        for plan in SAMPLE_PLAN:
            text = generate_text(
                model,
                tokenizer,
                plan["prompt"],
                plan["tokens"],
                plan["temperature"],
                plan["top_k"],
                plan["seed"],
            )
            samples.append({**plan, "text": text})
        write_json(
            destination / "samples.json",
            {
                "policy": "all three predefined prompts and seeds; no rejected samples or editing",
                "checkpoint_step": checkpoint["step"],
                "samples": samples,
            },
        )
        (destination / "samples.txt").write_text(
            "\n\n".join(
                f"SAMPLE {i + 1} | seed={sample['seed']} | temperature=0.8 | top_k=40\n"
                + sample["text"]
                for i, sample in enumerate(samples)
            )
            + "\n",
            encoding="utf-8",
        )
        runs.append(
            {
                "name": path.name,
                "summary": summary,
                "metadata": metadata,
                "evaluation": evaluation,
                "events": events,
            }
        )
        for split, models in evaluation["splits"].items():
            for name, metrics in models.items():
                table.append({"run": path.name, "split": split, "model": name, **metrics})
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    write_json(output / "sample_plan.json", {"samples": SAMPLE_PLAN})
    style_plots()
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 4.7), squeeze=False)
    for ax, run in zip(axes[0], runs, strict=True):
        events = [event for event in run["events"] if event["event"] == "eval"]
        steps = [event["step"] for event in events]
        ax.plot(
            steps,
            [event["train_loss"] for event in events],
            color="#008C82",
            linewidth=2.4,
            marker="o",
            markersize=4,
            label="Training estimate",
        )
        ax.plot(
            steps,
            [event["val_loss"] for event in events],
            color="#D58023",
            linewidth=2.4,
            marker="o",
            markersize=4,
            label="Validation estimate",
        )
        ax.set(
            title=f"{run['name']} · {run['summary']['parameters']:,} parameters",
            xlabel="Optimizer updates",
            ylabel="Cross-entropy (nats / token)",
        )
        ax.grid(axis="y")
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Learning from random initialization", fontsize=18, fontweight="bold", y=1.01)
    fig.text(
        0.5,
        -0.015,
        "Fixed sampled windows · seed 1337 · CPU float32 · token units differ across panels",
        ha="center",
        fontsize=10,
        color="#53647A",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "training-curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 4.7), squeeze=False)
    comparison_max = 1.17 * max(
        metrics["bits_per_byte"]
        for run in runs
        for metrics in run["evaluation"]["splits"]["test"].values()
    )
    for ax, run in zip(axes[0], runs, strict=True):
        scores = run["evaluation"]["splits"]["test"]
        names = ["uniform", "unigram", "bigram", "gpt"]
        values = [scores[name]["bits_per_byte"] for name in names]
        bars = ax.bar(
            np.arange(4), values, color=["#C6D0DF", "#A6B6CC", "#7B93B3", "#008C82"], width=0.62
        )
        ax.set_xticks(np.arange(4), ["Uniform", "Unigram", "Bigram", "GPT"])
        ax.set(title=run["name"], ylabel="Bits per scored UTF-8 byte (lower is better)")
        ax.set_ylim(0, comparison_max)
        ax.bar_label(bars, fmt="%.2f", padding=5, fontsize=11)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    fig.suptitle(
        "Held-out test: compare against simple baselines", fontsize=17, fontweight="bold", y=1.01
    )
    fig.text(
        0.5,
        -0.015,
        "Same test text · every target scored once · 128-token context resets · one training seed",
        ha="center",
        fontsize=10,
        color="#53647A",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "test-baselines.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    rows = [
        "# Measured experiment summary",
        "",
        "Generated from completed local runs by `scripts/export_report.py`.",
        "",
        "| Run | Parameters | Updates | Best sampled val loss | Test loss | "
        "Test perplexity | Test bits/byte | Wall minutes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        summary = run["summary"]
        test = run["evaluation"]["splits"]["test"]["gpt"]
        rows.append(
            f"| {run['name']} | {summary['parameters']:,} | {summary['steps_completed']:,} | "
            f"{summary['best_val_loss']:.4f} | {test['cross_entropy_nats']:.4f} | "
            f"{test['perplexity']:.3f} | {test['bits_per_byte']:.3f} | "
            f"{summary['elapsed_seconds'] / 60:.2f} |"
        )
    rows += [
        "",
        "Perplexities use different token units and must not be compared directly.",
        "These runs are a descriptive comparison, not a controlled tokenizer ablation.",
        "",
        "![Training curves](figures/training-curves.png)",
        "",
        "![Test baselines](figures/test-baselines.png)",
        "",
    ]
    (output / "README.md").write_text("\n".join(rows), encoding="utf-8")
    print(f"Exported {len(runs)} runs to {output}")


if __name__ == "__main__":
    main()
