from dataclasses import replace

import pytest
import torch

from gpt2_scratch.config import ExperimentConfig, ModelConfig, TrainConfig
from gpt2_scratch.data import load_prepared
from gpt2_scratch.model import GPT
from gpt2_scratch.runtime import load_checkpoint
from gpt2_scratch.trainer import build_optimizer, estimate_losses, learning_rate, train


def config():
    return ExperimentConfig(
        ModelConfig(block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.2),
        TrainConfig(
            max_steps=8,
            batch_size=2,
            warmup_steps=2,
            eval_interval=4,
            eval_batches=2,
            log_interval=2,
            device="cpu",
            num_threads=1,
        ),
    )


def test_lr_boundaries():
    cfg = config().train
    assert learning_rate(1, cfg) > 0
    assert learning_rate(cfg.warmup_steps, cfg) == cfg.learning_rate
    assert learning_rate(cfg.max_steps, cfg) == cfg.min_lr
    assert learning_rate(cfg.max_steps + 5, cfg) == cfg.min_lr
    assert learning_rate(3, cfg) >= learning_rate(4, cfg)


def test_optimizer_has_each_parameter_exactly_once():
    model = GPT(config().model)
    optimizer = build_optimizer(model, config().train)
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    assert all(p.ndim < 2 for p in optimizer.param_groups[1]["params"])
    assert optimizer.param_groups[1]["weight_decay"] == 0


def test_evaluation_preserves_rng_and_mode(prepared_data):
    tokenizer, arrays, _ = load_prepared(prepared_data)
    model = GPT(replace(config().model, vocab_size=tokenizer.vocab_size)).train()
    state = torch.get_rng_state().clone()
    first = estimate_losses(model, arrays, config().train, torch.device("cpu"))
    second = estimate_losses(model, arrays, config().train, torch.device("cpu"))
    assert first == second
    assert torch.equal(state, torch.get_rng_state())
    assert model.training


def test_interrupted_resume_matches_uninterrupted_training(prepared_data, tmp_path):
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    train(config(), prepared_data, full)
    paused = train(config(), prepared_data, resumed, stop_after=3)
    assert paused["status"] == "paused"
    summary = train(config(), prepared_data, resumed, resume=resumed / "last.pt")
    assert summary["status"] == "complete"
    a, b = load_checkpoint(full / "last.pt"), load_checkpoint(resumed / "last.pt")
    assert a["step"] == b["step"] == 8
    assert a["best_step"] == b["best_step"]
    assert a["best_val_loss"] == b["best_val_loss"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    assert torch.equal(a["train_rng"], b["train_rng"])
    assert torch.equal(a["torch_rng"], b["torch_rng"])


def test_final_checkpoint_saved_between_evaluation_intervals(prepared_data, tmp_path):
    cfg = replace(config(), train=replace(config().train, max_steps=7, eval_interval=4))
    run = tmp_path / "final"
    train(cfg, prepared_data, run)
    assert load_checkpoint(run / "last.pt")["step"] == 7
    assert (run / "best.pt").is_file()
    with pytest.raises(FileExistsError):
        train(cfg, prepared_data, run)


def test_resume_rejects_changed_configuration(prepared_data, tmp_path):
    run = tmp_path / "resume"
    train(config(), prepared_data, run, stop_after=3)
    changed = replace(config(), train=replace(config().train, seed=99))
    with pytest.raises(ValueError, match="identical"):
        train(changed, prepared_data, run, resume=run / "last.pt")
