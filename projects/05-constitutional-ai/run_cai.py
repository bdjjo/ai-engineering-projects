"""Run an auditable Constitutional AI experiment one stage at a time or end to end."""

import argparse

from cai.pipeline import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["collect", "reward", "sft", "rl", "evaluate", "all"])
    parser.add_argument("--config", default="configs/colab-pilot.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--prompts")
    parser.add_argument("--feedback-from", help="Import completed feedback into a new training run")
    args = parser.parse_args()
    run(args.config, args.out, args.stage, args.device, args.prompts, args.feedback_from)
