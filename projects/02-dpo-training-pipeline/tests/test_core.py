import math

import pytest
import torch

from dpo_pipeline.data import batch, encode, parse_hh
from dpo_pipeline.objective import dpo_loss, sequence_logps


def test_loss_at_reference_and_gradients():
    c = torch.tensor([-3.0], requires_grad=True)
    r = torch.tensor([-4.0], requires_grad=True)
    ref = torch.tensor([-3.0], requires_grad=True)
    loss, margin = dpo_loss(c, r, ref, torch.tensor([-4.0]))
    assert loss.item() == pytest.approx(math.log(2))
    assert margin.item() == 0
    loss.backward()
    assert c.grad < 0 and r.grad > 0 and ref.grad is None


@pytest.mark.parametrize("beta", [0, -1, float("nan"), float("inf")])
def test_bad_beta(beta):
    with pytest.raises(ValueError):
        dpo_loss(*[torch.zeros(1)] * 4, beta=beta)


def test_shift_and_prompt_mask():
    logits = torch.zeros(1, 4, 8, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, -100]])
    score, count = sequence_logps(logits, labels)
    assert score.item() == pytest.approx(-math.log(8)) and count.item() == 1
    score.sum().backward()
    assert logits.grad[0, 0].abs().sum() == 0
    assert logits.grad[0, 1].abs().sum() > 0
    assert logits.grad[0, 2:].abs().sum() == 0


def test_empty_response_rejected():
    with pytest.raises(ValueError):
        sequence_logps(torch.zeros(1, 3, 4), torch.full((1, 3), -100))


def test_history_preserved_and_mismatch_rejected():
    prefix = "\n\nHuman: first\n\nAssistant: earlier\n\nHuman: next\n\nAssistant: "
    result = parse_hh({"chosen": prefix + "yes", "rejected": prefix + "no"})
    assert len(result["messages"]) == 3
    with pytest.raises(ValueError):
        parse_hh({"chosen": prefix + "yes", "rejected": prefix.replace("first", "other") + "no"})
    with pytest.raises(ValueError):
        parse_hh({"chosen": prefix + "yes", "rejected": prefix + "yes"})


def test_pad_masks_do_not_remove_real_eos():
    result = batch(
        [
            {"input_ids": [1, 2], "labels": [-100, 2]},
            {"input_ids": [1, 3, 2], "labels": [-100, 3, 2]},
        ],
        2,
        "cpu",
    )
    assert result["labels"][0].tolist() == [-100, 2, -100]
    assert result["attention_mask"][0].tolist() == [1, 1, 0]


class Template:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        return [1, 3] if add_generation_prompt else [1, 3, 4, 2]


def test_template_budgets_and_mask():
    row = {"messages": [{"role": "user", "content": "x"}], "chosen": "y"}
    assert encode(row, Template())["chosen"]["labels"] == [-100, -100, 4, 2]
    with pytest.raises(ValueError):
        encode(row, Template(), max_prompt=1)
    with pytest.raises(ValueError):
        encode(row, Template(), max_length=3)
