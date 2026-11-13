"""Print one input/target example to make next-token prediction tangible."""

import argparse

from gpt2_scratch.data import load_prepared

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", default="data/processed/char")
parser.add_argument("--tokens", type=int, default=16)
args = parser.parse_args()
tokenizer, arrays, _ = load_prepared(args.data, ("train",))
ids = arrays["train"][: args.tokens + 1].tolist()
print("Input IDs: ", ids[:-1])
print("Target IDs:", ids[1:])
print("Input text: ", repr(tokenizer.decode(ids[:-1])))
print("Target text:", repr(tokenizer.decode(ids[1:])))
print("Each target is the next token after the corresponding input position.")
