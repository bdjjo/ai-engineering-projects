"""Train a small GPT from random weights; run with --help for all options."""

import argparse
import json
from dataclasses import replace

from gpt2_scratch.config import ExperimentConfig
from gpt2_scratch.runtime import cli_error
from gpt2_scratch.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/char_cpu.json")
    parser.add_argument("--data", default="data/processed/char")
    parser.add_argument("--out", default="runs/char-cpu")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume", help="Path to last.pt in --out; all settings must match")
    parser.add_argument(
        "--stop-after", type=int, help="Pause after this update without changing LR schedule"
    )
    args = parser.parse_args()
    try:
        config = ExperimentConfig.load(args.config)
        if args.device:
            config = replace(config, train=replace(config.train, device=args.device))
        print(
            json.dumps(train(config, args.data, args.out, args.resume, args.stop_after), indent=2)
        )
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        cli_error(exc)


if __name__ == "__main__":
    main()
