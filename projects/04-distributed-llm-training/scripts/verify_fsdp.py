"""Small two-GPU FSDP save/resume test; requires Linux and two CUDA devices."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from verify_distributed import compare, read_state

from distributed_gpt.config import load_config
from distributed_gpt.data import prepare_synthetic, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/fsdp-verification")
    args = parser.parse_args()
    if os.name == "nt" or torch.cuda.device_count() < 2:
        raise SystemExit("This check needs Linux, NCCL, and at least two CUDA GPUs")
    project = Path(__file__).resolve().parents[1]
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    prepare_synthetic(root / "data")
    config = load_config(project / "configs/cpu-smoke.json")
    config.train.precision = "auto"
    config.train.batch_size = 1
    write_json(root / "config.json", config.to_dict())
    env = dict(os.environ, PYTHONPATH=str(project / "src"))
    for name, extra in [
        ("full", []),
        ("first", ["--stop-after", "10"]),
        ("resumed", ["--resume", str(root / "first/checkpoints/step-000010")]),
    ]:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                str(project / "train_distributed.py"),
                "--strategy",
                "fsdp",
                "--device",
                "cuda",
                "--config",
                str(root / "config.json"),
                "--data",
                str(root / "data"),
                "--out",
                str(root / name),
                *extra,
            ],
            check=True,
            env=env,
            timeout=300,
        )
    full = read_state(root / "full/checkpoints/step-000020", config)
    resumed = read_state(root / "resumed/checkpoints/step-000020", config)
    difference = compare(full, resumed, exact=False)
    metadata = json.loads((root / "full/metadata.json").read_text())
    assert metadata["effective_sharding"] == "FULL_SHARD"
    write_json(
        root / "verification.json",
        {
            "passed": True,
            "model_and_optimizer_resume_max_absolute_difference": difference,
            "world_size": 2,
            "precision": metadata["precision"],
            "scope": "Tiny-model FSDP recovery only; not a billion-parameter benchmark",
        },
    )


if __name__ == "__main__":
    main()
