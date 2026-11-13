"""Fit a tokenizer on train only, then write independently encoded splits."""

import argparse
import json

from gpt2_scratch.data import prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/shakespeare.txt")
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--out", help="Default: data/processed/<tokenizer>")
    args = parser.parse_args()
    manifest = prepare_data(
        args.input, args.out or f"data/processed/{args.tokenizer}", args.tokenizer, args.vocab_size
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
