"""Collect bounded layer-8 GPT-2 residual activations into a local cache."""

import argparse
from pathlib import Path

import torch

from mech_interp.activation_collector import collect_dataset
from mech_interp.runtime import configure, load_model, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/activations.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--revision")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--train-tokens", type=int, default=8192)
    parser.add_argument("--val-tokens", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--corpus", choices=["wikitext", "openwebtext"], default="wikitext")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    path = Path(args.out)
    if path.exists():
        raise FileExistsError(f"Cache already exists: {path}")
    configure(threads=args.threads)
    model, provenance = load_model(args.device, args.revision)
    bundle = collect_dataset(
        model,
        provenance,
        args.train_tokens,
        args.val_tokens,
        args.layer,
        args.max_length,
        args.corpus,
        args.dataset_revision,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)
    write_json(path.with_suffix(".json"), bundle["metadata"])
    print(f"Saved {bundle['train'].shape} training activations to {path}")


if __name__ == "__main__":
    main()
