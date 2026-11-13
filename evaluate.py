"""Evaluate a chosen checkpoint on all validation/test targets, alongside baselines."""

import argparse
import json
from pathlib import Path

from gpt2_scratch.data import load_prepared
from gpt2_scratch.evaluation import evaluate_split
from gpt2_scratch.generation import load_model
from gpt2_scratch.runtime import cli_error, configure_runtime, resolve_device, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=["val", "test", "both"], default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out")
    args = parser.parse_args()
    try:
        configure_runtime(1337, args.threads)
        splits = ("val", "test") if args.split == "both" else (args.split,)
        tokenizer, arrays, manifest = load_prepared(args.data, ("train", *splits))
        model, checkpoint_tokenizer, checkpoint = load_model(
            args.checkpoint, resolve_device(args.device)
        )
        if (
            manifest != checkpoint["data_manifest"]
            or tokenizer.to_dict() != checkpoint_tokenizer.to_dict()
        ):
            raise ValueError("checkpoint and prepared data do not match")
        result = {
            "checkpoint_step": checkpoint["step"],
            "tokenizer": manifest["tokenizer"],
            "protocol": (
                "every target once; context resets every block_size tokens; first token excluded"
            ),
            "block_size": model.config.block_size,
            "smoothing_alpha": 1.0,
            "splits": {
                split: evaluate_split(
                    model,
                    tokenizer,
                    arrays["train"],
                    arrays[split],
                    manifest["splits"][split]["bytes"],
                    args.batch_size,
                )
                for split in splits
            },
        }
        output = args.out or Path(args.checkpoint).parent / f"evaluation-{args.split}.json"
        write_json(output, result)
        print(json.dumps(result, indent=2))
    except (ValueError, FileNotFoundError) as exc:
        cli_error(exc)


if __name__ == "__main__":
    main()
