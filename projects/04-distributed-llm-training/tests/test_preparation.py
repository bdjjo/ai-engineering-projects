import json
from types import SimpleNamespace

import datasets
import numpy as np
import pytest
import transformers

from distributed_gpt.preparation import normalized_hash, prepare_text


def test_text_preparation_budgets_eos_dedup_and_provenance(tmp_path, monkeypatch):
    shared = "Shared text " * 10
    records = {
        "train": [
            {"text": "short"},
            {"text": shared},
            {"text": shared},
            {"text": "Training data " * 10},
        ],
        "validation": [
            {"text": shared},
            {"text": "Held out example " * 10},
            {"text": "Another held out " * 10},
        ],
    }

    class Tokenizer:
        eos_token_id = 99

        def __len__(self):
            return 100

        def encode(self, text, add_special_tokens=False):
            return [1, 2, 3, 4]

    def load(name, subset, revision, split, streaming):
        assert name == "Salesforce/wikitext" and subset == "wikitext-2-raw-v1"
        assert revision == "dataset-sha" and streaming
        return records[split]

    monkeypatch.setattr(datasets, "load_dataset", load)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: SimpleNamespace(
            model_info=lambda _: SimpleNamespace(sha="tokenizer-sha"),
            dataset_info=lambda _: SimpleNamespace(sha="dataset-sha"),
        ),
    )
    manifest = prepare_text(tmp_path / "data", train_tokens=8, val_tokens=9)
    assert manifest["dataset_revision"] == "dataset-sha"
    assert manifest["tokenizer_revision"] == "tokenizer-sha"
    splits = manifest["splits"]
    assert not set(splits["train"]["text_sha256"]) & set(splits["validation"]["text_sha256"])
    assert splits["train"]["tokens"] == 8 and splits["validation"]["tokens"] == 9
    assert np.fromfile(tmp_path / "data/train.bin", dtype="<u2").tolist() == [
        1,
        2,
        3,
        4,
        99,
        1,
        2,
        3,
    ]
    assert json.loads((tmp_path / "data/manifest.json").read_text()) == manifest
    assert normalized_hash(" a  b\n") == normalized_hash("a b")
    with pytest.raises(FileExistsError):
        prepare_text(tmp_path / "data")
