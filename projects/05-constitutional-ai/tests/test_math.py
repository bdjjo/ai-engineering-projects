import math

import pytest
import torch

from cai.modeling import encode_response, generate, pad_batch, sequence_logps
from cai.policy import reinforce_loss
from cai.reward import RewardModel, bradley_terry_loss, last_nonpad


def test_pairwise_loss_value_and_gradient_direction():
    chosen = torch.tensor([0.0], requires_grad=True)
    rejected = torch.tensor([0.0], requires_grad=True)
    loss = bradley_terry_loss(chosen, rejected)
    assert loss.item() == pytest.approx(math.log(2))
    loss.backward()
    assert chosen.grad.item() < 0 < rejected.grad.item()
    assert torch.isfinite(bradley_terry_loss(torch.tensor([-1000.0]), torch.tensor([1000.0])))


def test_pooling_handles_left_and_right_padding():
    hidden = torch.arange(8).reshape(2, 4, 1).float()
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])
    assert last_nonpad(hidden, mask).flatten().tolist() == [1, 7]
    with pytest.raises(ValueError):
        last_nonpad(hidden, torch.zeros_like(mask))


def test_reward_scores_are_padding_invariant(tiny):
    lm, tokenizer = tiny
    reward = RewardModel(lm.model).eval()
    row = encode_response(tokenizer, "question item0", "helpful answer", 64)
    other = encode_response(tokenizer, "question question item1", "helpful helpful answer", 64)
    one = reward(pad_batch([row], tokenizer.pad_token_id, "cpu"))[0]
    batch = reward(pad_batch([row, other], tokenizer.pad_token_id, "cpu"))[0]
    assert one.item() == pytest.approx(batch.item(), abs=1e-6)
    assert all(not p.requires_grad for p in reward.backbone.parameters())


def test_response_only_masking_and_causal_shift(tiny):
    model, tokenizer = tiny
    row = encode_response(tokenizer, "question item0", "helpful answer", 64)
    assert row["labels"][:4] == [-100] * 4
    batch = pad_batch([row], 0, "cpu")
    logp, count = sequence_logps(model.eval(), batch)
    assert count.item() == 3  # helpful, answer, EOS
    loss = model(**batch).loss
    assert (-logp / count).item() == pytest.approx(loss.item(), abs=1e-6)
    with pytest.raises(ValueError, match="exceeds"):
        encode_response(tokenizer, "question item0", "helpful answer", 3)
    with pytest.raises(ValueError, match="response target"):
        sequence_logps(model, pad_batch([{"input_ids": [3, 4]}], 0, "cpu"))


def test_generation_respects_context_and_keeps_rollout_targets(tiny):
    model, tokenizer = tiny
    with pytest.raises(ValueError, match="context"):
        generate(model, tokenizer, "question", max_new_tokens=130)
    result = generate(model, tokenizer, "question", max_new_tokens=3, sample=True)
    assert 1 <= result["new_tokens"] <= 3
    assert sum(t != -100 for t in result["labels"]) == result["new_tokens"]


def test_rl_sampling_ignores_inherited_repetition_penalty(tiny):
    model, tokenizer = tiny
    first = generate(model, tokenizer, "question", max_new_tokens=8, sample=True, seed=42)
    model.generation_config.repetition_penalty = 100.0
    model.generation_config.no_repeat_ngram_size = 1
    second = generate(model, tokenizer, "question", max_new_tokens=8, sample=True, seed=42)
    assert first["input_ids"] == second["input_ids"]


def test_reinforce_single_item_has_gradient_and_frozen_targets():
    logp = torch.tensor([-2.0], requires_grad=True)
    reference = torch.tensor([-3.0], requires_grad=True)
    reward = torch.tensor([2.0], requires_grad=True)
    loss, utility, kl = reinforce_loss(logp, reference, reward, baseline=0.5, kl_coef=0.1)
    loss.backward()
    assert kl.item() == 1 and utility.item() == pytest.approx(1.9)
    assert logp.grad.item() == pytest.approx(-1.4)
    assert reference.grad is None and reward.grad is None


def test_reinforce_expected_gradient_matches_exact_regularized_objective():
    logits = torch.tensor([0.2, -0.1], requires_grad=True)
    logp = logits.log_softmax(-1)
    ref = torch.tensor([0.5, 0.5]).log()
    rewards = torch.tensor([1.0, -0.5])
    exact = -(logp.exp() * (rewards - 0.2 * (logp - ref))).sum()
    expected_grad = torch.autograd.grad(exact, logits, retain_graph=True)[0]
    utility = rewards - 0.2 * (logp.detach() - ref)
    estimator_expectation = -(logp.exp().detach() * utility * logp).sum()
    actual_grad = torch.autograd.grad(estimator_expectation, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
