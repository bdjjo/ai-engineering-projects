from types import SimpleNamespace

import pytest
import torch

from mech_interp.activation_collector import collect_split, text_hash
from mech_interp.feature_browser import render_feature


class FakeModel:
    cfg = SimpleNamespace(n_ctx=128)

    def to_tokens(self, text, **kwargs):
        return torch.tensor([[999, 1, 2, 3, 4, 5]])

    def run_with_cache(self, tokens, names_filter, **kwargs):
        return None, {names_filter[0]: tokens.float().unsqueeze(-1).expand(-1, -1, 2)}


def test_collection_excludes_bos_duplicates_and_validation_overlap():
    first = (
        "A sufficiently long paragraph about a test dataset used to verify activation collection."
    )
    second = (
        "A different paragraph with enough characters to pass filtering and test the token budget."
    )
    records = [{"text": text} for text in [first, first, second]]
    acts, hashes = collect_split(FakeModel(), records, 7, "hook")
    assert acts.shape == (7, 2)
    assert acts.max() == 5
    assert len(hashes) == 2
    val, keys = collect_split(FakeModel(), records, 4, "hook", excluded=[text_hash(first)])
    assert keys == [text_hash(second)]
    assert val.shape == (4, 2)


def test_exhausted_collection_does_not_silently_change_budget():
    with pytest.raises(ValueError, match="exhausted"):
        collect_split(FakeModel(), [], 100, "hook")


def example_index():
    return {
        "hidden_dim": 1,
        "features": [
            {
                "id": 0,
                "active_fraction": 0.5,
                "max_activation": 2.0,
                "examples": [
                    {
                        "before": "<script>",
                        "token": "&test",
                        "after": "</script>",
                        "activation": 2.0,
                    }
                ],
            }
        ],
    }


def test_browser_escapes_text_and_validates_feature_id():
    title, rendered = render_feature(example_index(), 0)
    assert "Feature 0" in title
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "<mark>&amp;test</mark>" in rendered
    for feature in [-1, 1, 0.5, float("nan")]:
        with pytest.raises(ValueError):
            render_feature(example_index(), feature)
