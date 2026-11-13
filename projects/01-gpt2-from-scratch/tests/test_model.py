from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from gpt2_scratch.config import ModelConfig
from gpt2_scratch.model import GPT


def tiny_config(**kwargs):
    return replace(
        ModelConfig(vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=16), **kwargs
    )


def test_shapes_loss_and_weight_tying():
    model = GPT(tiny_config())
    x = torch.randint(11, (2, 8))
    y = torch.randint(11, (2, 8))
    logits, loss = model(x, y)
    assert logits.shape == (2, 8, 11)
    torch.testing.assert_close(loss, F.cross_entropy(logits.reshape(-1, 11), y.reshape(-1)))
    assert model.lm_head.weight is model.token_embedding.weight
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
def test_future_tokens_cannot_change_prefix_predictions(backend):
    model = GPT(tiny_config(attention_backend=backend)).eval()
    a = torch.tensor([[1, 2, 3, 4, 5, 6]])
    b = torch.tensor([[1, 2, 3, 9, 8, 7]])
    logits_a, _ = model(a)
    logits_b, _ = model(b)
    torch.testing.assert_close(logits_a[:, :3], logits_b[:, :3], atol=1e-7, rtol=1e-6)


def test_manual_and_sdpa_match_forward_and_backward():
    manual = GPT(tiny_config(dropout=0.0))
    sdpa = GPT(tiny_config(dropout=0.0, attention_backend="sdpa"))
    sdpa.load_state_dict(manual.state_dict())
    x, y = torch.randint(11, (2, 6)), torch.randint(11, (2, 6))
    logits_a, loss_a = manual(x, y)
    logits_b, loss_b = sdpa(x, y)
    torch.testing.assert_close(logits_a, logits_b, atol=1e-6, rtol=1e-5)
    loss_a.backward()
    loss_b.backward()
    for param_a, param_b in zip(manual.parameters(), sdpa.parameters(), strict=True):
        torch.testing.assert_close(param_a.grad, param_b.grad, atol=1e-6, rtol=1e-4)


def test_tiny_batch_can_be_overfit():
    model = GPT(tiny_config(dropout=0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0)
    x = torch.tensor([[1, 2, 3, 1, 2, 3]])
    y = torch.tensor([[2, 3, 1, 2, 3, 1]])
    initial = float(model(x, y)[1].detach())
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
    final = float(model(x, y)[1].detach())
    assert final < 0.1 and final < initial / 10


def test_generation_is_seeded_preserves_mode_and_crops_context():
    model = GPT(tiny_config(dropout=0.5)).train()
    prompt = torch.tensor([[1, 2, 3]])
    a = model.generate(prompt, 12, generator=torch.Generator().manual_seed(42))
    b = model.generate(prompt, 12, generator=torch.Generator().manual_seed(42))
    assert torch.equal(a, b)
    assert a.shape == (1, 15)
    assert torch.equal(a[:, :3], prompt)
    assert model.training


@pytest.mark.parametrize(
    "kwargs", [{"temperature": 0}, {"top_k": 0}, {"temperature": float("nan")}]
)
def test_invalid_sampling_arguments(kwargs):
    with pytest.raises(ValueError):
        GPT(tiny_config()).generate(torch.tensor([[1]]), 2, **kwargs)


def test_context_limit_and_invalid_head_dimension():
    with pytest.raises(ValueError, match="block_size"):
        GPT(tiny_config())(torch.ones((1, 9), dtype=torch.long))
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(n_head=3, n_embd=16)
