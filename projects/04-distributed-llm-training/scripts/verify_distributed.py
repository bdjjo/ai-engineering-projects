"""Real two-rank CPU DDP check, equal global batch, and distributed resume comparison."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict

from distributed_gpt.checkpoint import OPTIONS
from distributed_gpt.config import load_config
from distributed_gpt.data import prepare_synthetic, write_json
from distributed_gpt.model import GPT, optimizer_groups

PROJECT = Path(__file__).resolve().parents[1]


def read_state(path, config):
    model = GPT(config.model)
    optimizer = torch.optim.AdamW(optimizer_groups(model, config.train.weight_decay), foreach=False)
    model_state, optim_state = get_state_dict(model, optimizer, options=OPTIONS)
    state = {"model": model_state, "optimizer": optim_state}
    dcp.load(state, checkpoint_id=path, no_dist=True)
    return state


def compare(a, b, exact):
    maximum = 0.0
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0 if exact else 2e-6, rtol=0 if exact else 2e-4)
        return (a - b).abs().max().item() if a.numel() else 0.0
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            maximum = max(maximum, compare(a[key], b[key], exact))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            maximum = max(maximum, compare(x, y, exact))
    else:
        assert a == b, (a, b)
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/distributed-verification")
    args = parser.parse_args()
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    prepare_synthetic(root / "data")
    config_path = PROJECT / "configs/cpu-smoke.json"
    config = load_config(config_path)
    env = dict(os.environ, PYTHONPATH=str(PROJECT / "src"), OMP_NUM_THREADS="1", USE_LIBUV="0")

    def launch(name, distributed=False, extra=()):
        command = [sys.executable]
        if distributed:
            if os.name == "nt":
                command += [str(PROJECT / "scripts/launch_cpu_ddp.py")]
            else:
                command += [
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node=2",
                    str(PROJECT / "train_distributed.py"),
                    "--strategy",
                    "ddp",
                ]
        else:
            command += [str(PROJECT / "train_distributed.py")]
        command += [
            "--config",
            str(config_path),
            "--data",
            str(root / "data"),
            "--out",
            str(root / name),
            "--device",
            "cpu",
            "--batch-size",
            "1" if distributed else "2",
            *extra,
        ]
        subprocess.run(command, check=True, env=env, timeout=240)

    launch("single")
    launch("ddp", True)
    launch("ddp-first", True, ["--stop-after", "10"])
    launch("ddp-resumed", True, ["--resume", str(root / "ddp-first/checkpoints/step-000010")])
    states = {
        name: read_state(root / name / "checkpoints/step-000020", config)
        for name in ("single", "ddp", "ddp-resumed")
    }
    report = {
        "single_vs_ddp_max_absolute_tensor_difference": compare(
            states["single"], states["ddp"], False
        ),
        "ddp_resume_max_absolute_tensor_difference": compare(
            states["ddp"], states["ddp-resumed"], True
        ),
        "global_tokens_per_step": 128,
        "scope": "CPU FP32, two Gloo ranks, equal global batch; model AND AdamW state compared",
        "multi_gpu_fsdp_verified": False,
        "passed": True,
    }
    write_json(root / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
