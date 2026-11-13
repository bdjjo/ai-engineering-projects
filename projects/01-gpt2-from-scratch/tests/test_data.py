import json

import pytest
import torch

from gpt2_scratch.data import load_prepared, prepare_data, sample_batch, split_text
from gpt2_scratch.tokenizer import load_tokenizer


def test_splits_are_contiguous_and_reconstruct_source():
    text = "0123456789" * 100
    splits = split_text(text)
    assert [len(part) for part in splits.values()] == [900, 50, 50]
    assert "".join(splits.values()) == text


def test_batch_targets_are_shifted_and_last_window_is_valid():
    tokens = torch.arange(9)
    x, y = sample_batch(tokens, 3, 8, torch.Generator().manual_seed(2), torch.device("cpu"))
    assert x.shape == (3, 8)
    assert torch.equal(x + 1, y)
    assert int(y[0, -1]) == 8


def test_char_vocabulary_does_not_learn_from_validation(tmp_path):
    source = tmp_path / "text.txt"
    source.write_text("a" * 180 + "z" * 20, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown character"):
        prepare_data(source, tmp_path / "char")


def test_bpe_merges_fit_only_training_text(tmp_path):
    source = tmp_path / "text.txt"
    source.write_text("ab " * 60 + "zz " * 4 + "q" * 8, encoding="utf-8")
    directory = tmp_path / "bpe"
    manifest = prepare_data(source, directory, "bpe", 257)
    tokenizer = load_tokenizer(directory / "tokenizer.json")
    assert tokenizer.merges == [(ord("a"), ord("b"))]
    assert manifest["splits"]["train"]["end_character"] == 180
    assert tokenizer.decode(tokenizer.encode("zz q")) == "zz q"


def test_tampered_arrays_are_rejected(prepared_data):
    with (prepared_data / "train.npy").open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="hash"):
        load_prepared(prepared_data)


def test_loading_training_does_not_load_test_partition(prepared_data):
    # A test artifact can be absent throughout training: it is loaded only on explicit evaluation.
    (prepared_data / "test.npy").unlink()
    _, arrays, manifest = load_prepared(prepared_data)
    assert set(arrays) == {"train", "val"}
    assert manifest == json.loads((prepared_data / "manifest.json").read_text())
