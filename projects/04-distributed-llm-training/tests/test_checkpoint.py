import copy
import json
from dataclasses import replace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict

from distributed_gpt.checkpoint import OPTIONS, CheckpointManager, load_checkpoint
from distributed_gpt.config import ExperimentConfig, ModelConfig, TrainConfig
from distributed_gpt.data import prepare_synthetic
from distributed_gpt.distributed import Context, make_scaler
from distributed_gpt.model import GPT, optimizer_groups
from distributed_gpt.trainer import run_training


def objects(config=None):
    model = GPT(config or ModelConfig())
    optimizer = torch.optim.AdamW(optimizer_groups(model, 0.1), lr=0.001, foreach=False)
    scaler = make_scaler("single", "fp32", torch.device("cpu"))
    return model, optimizer, scaler


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


def test_async_snapshot_is_immutable_and_restores_optimizer_rng(tmp_path):
    model, optimizer, scaler = objects()
    x, y = torch.randint(0, 128, (2, 2, 16))
    model(x, y)[1].backward()
    optimizer.step()
    expected = copy.deepcopy(get_state_dict(model, optimizer, options=OPTIONS))
    rng = torch.get_rng_state().clone()
    context = Context(0, 1, torch.device("cpu"))
    manager = CheckpointManager(context, tmp_path, {"example": 1})
    manager.save(model, optimizer, scaler, 1, 2)
    # Mutate immediately, while serialization may still be running.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(10)
        for state in optimizer.state.values():
            state["exp_avg"].add_(3)
    torch.rand(20)
    manager.wait()
    new_model, new_opt, new_scaler = objects()
    step = load_checkpoint(
        tmp_path / "step-000001", new_model, new_opt, new_scaler, context, {"example": 1}, 2
    )
    assert step == 1
    assert_tree_equal(expected, get_state_dict(new_model, new_opt, options=OPTIONS))
    assert torch.equal(torch.get_rng_state(), rng)
    with pytest.raises(ValueError, match="matching"):
        load_checkpoint(
            tmp_path / "step-000001", new_model, new_opt, new_scaler, context, {"example": 2}, 2
        )
    with pytest.raises(ValueError, match="completion"):
        load_checkpoint(tmp_path / "partial", new_model, new_opt, new_scaler, context, {}, 2)


def checkpoint_state(path, config):
    model, optimizer, _ = objects(config)
    model_state, optimizer_state = get_state_dict(model, optimizer, options=OPTIONS)
    state = {"model": model_state, "optimizer": optimizer_state}
    dcp.load(state, checkpoint_id=path, no_dist=True)
    return state


def test_interrupted_training_exact_recovery_with_dropout(tmp_path):
    config = ExperimentConfig(
        ModelConfig(dropout=0.2), TrainConfig(steps=6, warmup_steps=1, save_every=3, log_every=6)
    )
    prepare_synthetic(tmp_path / "data")
    full = run_training(config, tmp_path / "data", tmp_path / "full", threads=1)
    run_training(config, tmp_path / "data", tmp_path / "first", threads=1, stop_after=3)
    resumed = run_training(
        config,
        tmp_path / "data",
        tmp_path / "resumed",
        threads=1,
        resume=tmp_path / "first/checkpoints/step-000003",
    )
    assert full["final_validation_loss"] == resumed["final_validation_loss"]
    assert_tree_equal(
        checkpoint_state(tmp_path / "full/checkpoints/step-000006", config.model),
        checkpoint_state(tmp_path / "resumed/checkpoints/step-000006", config.model),
    )
    assert full["performance"]["mfu_percent"] is None
    assert full["cuda_peak_memory"] is None
    assert not full["gpu_mfu_measured"]
    with pytest.raises(FileExistsError):
        run_training(config, tmp_path / "data", tmp_path / "full", threads=1)
    altered = replace(config, train=replace(config.train, batch_size=1))
    with pytest.raises(ValueError, match="matching"):
        run_training(
            altered,
            tmp_path / "data",
            tmp_path / "bad",
            threads=1,
            resume=tmp_path / "first/checkpoints/step-000003",
        )


def test_profiler_exports_trace(tmp_path):
    prepare_synthetic(tmp_path / "data")
    config = ExperimentConfig(train=TrainConfig(steps=5, warmup_steps=1))
    summary = run_training(config, tmp_path / "data", tmp_path / "profile", threads=1, profile=True)
    assert summary["profile_enabled"]
    trace = json.loads((tmp_path / "profile/trace-rank0.json").read_text())
    assert trace["traceEvents"]
    assert (tmp_path / "profile/profiler.txt").stat().st_size > 100
