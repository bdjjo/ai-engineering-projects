import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from distributed_gpt.config import ExperimentConfig, TrainConfig
from distributed_gpt.data import prepare_synthetic
from distributed_gpt.trainer import run_training


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_cuda_amp_checkpoint_training(tmp_path):
    prepare_synthetic(tmp_path / "data")
    config = ExperimentConfig(train=TrainConfig(steps=5, warmup_steps=1, precision="auto"))
    summary = run_training(config, tmp_path / "data", tmp_path / "run", device="cuda")
    assert summary["completed"]
    assert summary["cuda_peak_memory"]["max_rank_allocated_bytes"] > 0
    assert summary["final_validation_loss"] < summary["initial_validation_loss"]


@pytest.mark.multigpu
@pytest.mark.skipif(
    os.name == "nt" or torch.cuda.device_count() < 2, reason="Linux and two CUDA devices required"
)
def test_real_fsdp_resume(tmp_path):
    project = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, str(project / "scripts/verify_fsdp.py"), "--out", str(tmp_path / "fsdp")],
        check=True,
        timeout=1000,
    )
