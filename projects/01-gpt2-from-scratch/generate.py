"""Generate text from a saved model. Output includes the original prompt."""

import argparse
from pathlib import Path

from gpt2_scratch.generation import generate_text, load_model
from gpt2_scratch.runtime import cli_error, configure_runtime, resolve_device, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="ROMEO:\n")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--out", help="Optional .txt output; sampling metadata goes beside it as .json"
    )
    args = parser.parse_args()
    try:
        configure_runtime(args.seed, args.threads)
        model, tokenizer, checkpoint = load_model(args.checkpoint, resolve_device(args.device))
        text = generate_text(
            model, tokenizer, args.prompt, args.tokens, args.temperature, args.top_k, args.seed
        )
        print(text)
        if args.out:
            output = Path(args.out)
            if output.suffix != ".txt":
                raise ValueError("--out must have a .txt extension")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
            write_json(
                output.with_suffix(".json"),
                {
                    "prompt": args.prompt,
                    "max_new_tokens": args.tokens,
                    "seed": args.seed,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "checkpoint_step": checkpoint["step"],
                    "tokenizer": checkpoint["tokenizer"]["kind"],
                    "device": args.device,
                },
            )
    except (ValueError, FileNotFoundError) as exc:
        cli_error(exc)


if __name__ == "__main__":
    main()
