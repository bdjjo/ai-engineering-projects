from dataclasses import replace

import numpy as np
import pytest
import torch

from distributed_gpt.config import ModelConfig, TrainConfig, learning_rate
from distributed_gpt.data import RankBatchStream, TokenDataset, prepare_synthetic
from distributed_gpt.distributed import resolve_precision
from distributed_gpt.performance import flops_per_token, peak_tflops, step_metrics, summarize


def test_shift_and_disjoint_rank_batches_across_epochs(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(257, dtype="<u2").tofile(path)
    dataset = TokenDataset(path, 8, 300)
    x, y = dataset[0]
    assert x.tolist() == list(range(8))
    assert y.tolist() == list(range(1, 9))
    ranks = [RankBatchStream(dataset, 2, rank=r, world_size=2, seed=5) for r in range(2)]
    single = RankBatchStream(dataset, 4, seed=5)
    epoch_orders = []
    for epoch in range(2):
        order = []
        for micro in range(epoch * 8, (epoch + 1) * 8):
            a, b = [rank.indices(micro) for rank in ranks]
            assert not set(a) & set(b)
            assert a + b == single.indices(micro)
            order.extend(a + b)
        assert sorted(order) == list(range(len(dataset)))
        epoch_orders.append(order)
    assert epoch_orders[0] != epoch_orders[1]
    resumed = RankBatchStream(dataset, 2, rank=1, world_size=2, seed=5)
    assert resumed.indices(13) == ranks[1].indices(13)
    with pytest.raises(IndexError):
        dataset[-1]


@pytest.mark.parametrize("values,block,vocab", [([0, 1], 4, 8), ([0, 8, 1], 2, 8)])
def test_invalid_token_data(tmp_path, values, block, vocab):
    path = tmp_path / "bad.bin"
    np.array(values, dtype="<u2").tofile(path)
    with pytest.raises(ValueError):
        TokenDataset(path, block, vocab)


def test_preparation_reproducible_and_refuses_overwrite(tmp_path):
    a = prepare_synthetic(tmp_path / "a")
    b = prepare_synthetic(tmp_path / "b")
    assert a == b
    with pytest.raises(FileExistsError):
        prepare_synthetic(tmp_path / "a")


def test_schedule_has_nonzero_start_and_cosine_floor():
    config = TrainConfig()
    assert learning_rate(0, config) == config.learning_rate / config.warmup_steps
    assert learning_rate(config.warmup_steps, config) == config.learning_rate
    assert learning_rate(config.steps - 1, config) == config.learning_rate * config.min_lr_ratio
    with pytest.raises(ValueError):
        learning_rate(config.steps, config)
    with pytest.raises(ValueError):
        replace(config, warmup_steps=config.steps)


def test_mfu_counts_global_tokens_and_devices_once():
    row = step_metrics(2, 4, 8, 1024, 2, 6e9, 312)
    assert row["global_tokens"] == 65536
    assert row["tokens_per_second"] == 32768
    assert row["mfu_percent"] == pytest.approx(31.50769230769231)
    one = step_metrics(2, 4, 8, 1024, 1, 6e9, 312)
    assert one["mfu_percent"] == row["mfu_percent"]
    assert step_metrics(1, 1, 1, 8, 1, 100)["mfu_percent"] is None


def test_pooled_throughput_uses_total_time():
    rows = [
        dict(step=i + 1, **step_metrics(s, 1, 1, 100, 1, 1e9, 10)) for i, s in enumerate([10, 1, 3])
    ]
    summary = summarize(rows, 1)
    assert summary["tokens_per_second"] == 50
    assert summary["mfu_percent"] == 0.5
    assert summarize(rows, 3)["tokens_per_second"] is None


@pytest.mark.parametrize(
    "name,precision,expected",
    [
        ("Tesla T4", "fp16", 65),
        ("Tesla T4", "bf16", None),
        ("NVIDIA A100-SXM4-40GB", "bf16", 312),
        ("A100 MIG 1g.5gb", "bf16", None),
        ("Unspecified GPU", "fp16", None),
        ("A100", "fp32", None),
    ],
)
def test_peak_table_is_precision_specific(name, precision, expected):
    assert peak_tflops(name, precision)[0] == expected


def test_cpu_precision_and_flops():
    assert resolve_precision("auto", torch.device("cpu")) == "fp32"
    with pytest.raises(ValueError):
        resolve_precision("fp16", torch.device("cpu"))
    assert flops_per_token(ModelConfig(block_size=64)) > flops_per_token(ModelConfig())
