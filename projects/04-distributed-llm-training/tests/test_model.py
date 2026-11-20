import copy
from dataclasses import replace

import pytest
import torch

from distributed_gpt.config import ModelConfig
from distributed_gpt.distributed import enable_activation_checkpointing
from distributed_gpt.model import GPT, optimizer_groups, parameter_count_formula


@pytest.mark.parametrize("bias", [True, False])
def test_tying_count_and_optimizer_coverage(bias):
    config = ModelConfig(bias=bias)
    model = GPT(config)
    assert model.lm_head.weight is model.token_embedding.weight
    assert model.parameter_count() == parameter_count_formula(config)
    groups = optimizer_groups(model, 0.1)
    params = [p for group in groups for p in group["params"]]
    assert len({id(p) for p in params}) == len(list(model.parameters())) == len(params)
    assert all(p.ndim == 1 for p in groups[1]["params"])
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            assert torch.count_nonzero(module.bias) == 0


def test_future_tokens_do_not_change_past_logits():
    model = GPT(ModelConfig()).eval()
    x = torch.randint(0, 128, (2, 16))
    changed = x.clone()
    changed[:, 8:] = (changed[:, 8:] + 1) % 128
    torch.testing.assert_close(model(x)[0][:, :8], model(changed)[0][:, :8])
    with pytest.raises(ValueError, match="context"):
        model(torch.zeros(1, 33, dtype=torch.long))


def test_accumulation_matches_full_batch_gradient():
    model = GPT(ModelConfig())
    accumulated = copy.deepcopy(model)
    x, y = torch.randint(0, 128, (2, 4, 16))
    model(x, y)[1].backward()
    for a, b in zip(x.chunk(2), y.chunk(2)):
        (accumulated(a, b)[1] / 2).backward()
    for full, micro in zip(model.parameters(), accumulated.parameters()):
        torch.testing.assert_close(full.grad, micro.grad, atol=2e-7, rtol=2e-5)


def test_activation_recomputation_preserves_dropout_and_gradients():
    model = GPT(replace(ModelConfig(), dropout=0.2))
    checked = copy.deepcopy(model)
    enable_activation_checkpointing(checked)
    x, y = torch.randint(0, 128, (2, 2, 16))
    torch.manual_seed(61)
    expected = model(x, y)[1]
    expected.backward()
    torch.manual_seed(61)
    actual = checked(x, y)[1]
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for a, b in zip(model.parameters(), checked.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_billion_model_count_without_allocating_weights():
    config = ModelConfig(vocab_size=50257, block_size=1024, n_layer=24, n_head=16, n_embd=2048)
    with torch.device("meta"):
        model = GPT(config)
    assert model.parameter_count() == parameter_count_formula(config)
    assert 1.3e9 < model.parameter_count() < 1.4e9
