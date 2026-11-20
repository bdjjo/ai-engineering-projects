"""Count large model parameters on the meta device without allocating their weights."""

import argparse
import json

import torch

from distributed_gpt.config import load_config
from distributed_gpt.model import GPT, parameter_count_formula
from distributed_gpt.performance import flops_per_token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fsdp-1b.json")
    args = parser.parse_args()
    config = load_config(args.config)
    with torch.device("meta"):
        model = GPT(config.model)
    count = model.parameter_count()
    assert count == parameter_count_formula(config.model)
    print(
        json.dumps(
            {
                "parameter_count": count,
                "model_flops_per_token_estimate": flops_per_token(config.model),
                "fp32_parameter_bytes": count * 4,
                "unsharded_fp32_weights_gradients_adam_bytes_estimate": count * 16,
                "note": "Excludes activations, temporary buffers, and allocator overhead",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
