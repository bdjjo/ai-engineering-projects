import re

import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from mech_interp.ioi_circuit import (
    logit_difference,
    make_prompts,
    patch_head,
    prepare_batches,
    run_activation_patching,
)


def tiny_model():
    prompts = make_prompts()
    words = sorted(set(re.findall(r"\w+|[^\w\s]", " ".join(p.clean for p in prompts))))
    vocab = {word: index + 2 for index, word in enumerate(words)}
    cfg = HookedTransformerConfig(
        n_layers=2,
        n_heads=2,
        d_model=16,
        d_head=8,
        d_mlp=32,
        n_ctx=64,
        d_vocab=128,
        act_fn="relu",
        normalization_type="LN",
        seed=1337,
        device="cpu",
    )
    model = HookedTransformer(cfg).eval()
    model.to_tokens = lambda text, **kwargs: torch.tensor(
        [[1] + [vocab[t] for t in re.findall(r"\w+|[^\w\s]", text)]]
    )
    model.to_single_token = lambda name: vocab[name.strip()]
    return model


def test_logit_difference_uses_distinct_subject_and_recipient():
    logits = torch.zeros(2, 3, 5)
    logits[0, -1, 1], logits[0, -1, 2] = 7, 3
    logits[1, -1, 3], logits[1, -1, 4] = 1, 6
    assert logit_difference(logits, torch.tensor([[1, 2], [3, 4]])).tolist() == [4, -5]
    with pytest.raises(ValueError, match="different"):
        logit_difference(logits, torch.tensor([[1, 1], [3, 4]]))


def test_prompt_labels_and_alignment():
    model = tiny_model()
    prompts = make_prompts()
    assert len(prompts) == 16
    for pair in prompts:
        assert f", {pair.subject} gave" in pair.clean
        assert f", {pair.recipient} gave" in pair.corrupt
    batches = list(prepare_batches(model, prompts, 3))
    assert sum(len(batch[0]) for batch in batches) == 16
    for _, clean, corrupt, answers in batches:
        assert clean.shape == corrupt.shape
        assert (answers[:, 0] != answers[:, 1]).all()


def test_patch_only_changes_target_head_without_mutating_input():
    value = torch.zeros(2, 3, 4, 5)
    source = {"test": torch.ones_like(value)}
    hook = type("Hook", (), {"name": "test"})()
    changed = patch_head(value, hook, source=source, head=2)
    assert changed[:, :, 2].eq(1).all()
    assert changed[:, :, [0, 1, 3]].eq(0).all()
    assert value.eq(0).all()


def test_actual_transformerlens_hooks_and_batch_invariance():
    model = tiny_model()
    prompts = make_prompts()[:2]
    a = run_activation_patching(model, prompts, batch_size=2)
    b = run_activation_patching(model, prompts, batch_size=1)
    assert a["self_patch_max_error"] < 1e-5
    assert torch.tensor(a["damage"]).abs().max() > 1e-6
    torch.testing.assert_close(
        torch.tensor(a["damage"]), torch.tensor(b["damage"]), atol=1e-5, rtol=1e-4
    )
    for block in model.blocks:
        assert len(block.attn.hook_z.fwd_hooks) == 0
