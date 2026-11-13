import math

import torch

from gpt2_scratch.config import ModelConfig
from gpt2_scratch.evaluation import baseline_nll, full_split_nll, metrics_from_nll
from gpt2_scratch.model import GPT


def test_full_evaluation_scores_partial_last_block_and_is_batch_invariant():
    model = GPT(ModelConfig(vocab_size=7, block_size=8, n_layer=1, n_head=2, n_embd=16)).eval()
    tokens = torch.arange(28) % 7
    nll_a, count_a = full_split_nll(model, tokens, batch_size=1)
    nll_b, count_b = full_split_nll(model, tokens, batch_size=3)
    assert count_a == count_b == 27
    assert math.isclose(nll_a, nll_b, abs_tol=1e-5)


def test_baseline_learns_predictable_transitions():
    train = torch.tensor([0, 1] * 100)
    held_out = torch.tensor([0, 1] * 10)
    values = baseline_nll(train, held_out, 2)
    assert values["bigram"] < values["unigram"]
    assert math.isclose(values["uniform"], 19 * math.log(2))
    metrics = metrics_from_nll(values["uniform"], 19, 19)
    assert metrics["bits_per_byte"] == 1.0
    assert metrics["perplexity"] == 2.0
