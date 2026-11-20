"""Train GPT locally or launch with torchrun for DDP/FSDP."""

import argparse
from dataclasses import replace

from distributed_gpt.config import load_config
from distributed_gpt.trainer import run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cpu-smoke.json")
    parser.add_argument("--data", default="data/synthetic")
    parser.add_argument("--out", default="outputs/cpu-smoke")
    parser.add_argument("--strategy", choices=["single", "ddp", "fsdp"], default="single")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--peak-tflops", type=float)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.batch_size is not None:
        config.train = replace(config.train, batch_size=args.batch_size)
    run_training(
        config,
        args.data,
        args.out,
        args.strategy,
        args.device,
        args.threads,
        args.resume,
        args.stop_after,
        args.profile,
        args.peak_tflops,
    )


if __name__ == "__main__":
    main()
