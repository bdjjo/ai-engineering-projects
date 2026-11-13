import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_small_tests():
    torch.set_num_threads(1)
    torch.manual_seed(123)


@pytest.fixture
def prepared_data(tmp_path):
    from gpt2_scratch.data import prepare_data

    source = tmp_path / "text.txt"
    source.write_text("abcde\n" * 200, encoding="utf-8")
    directory = tmp_path / "prepared"
    prepare_data(source, directory)
    return directory
