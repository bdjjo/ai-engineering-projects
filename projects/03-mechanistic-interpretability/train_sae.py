"""Train a sparse autoencoder from a collected activation cache."""

import argparse
import json
from pathlib import Path

from mech_interp.sparse_autoencoder import SAEConfig
from mech_interp.training import train_sae


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", default="data/activations.pt")
    parser.add_argument("--config", default="configs/cpu-reference.json")
    parser.add_argument("--out", default="outputs/sae")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    config = SAEConfig(**json.loads(Path(args.config).read_text()))
    train_sae(args.activations, config, args.out, args.device, args.threads)


if __name__ == "__main__":
    main()
