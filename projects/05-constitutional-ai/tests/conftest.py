import pytest
import torch
from smoke import tiny_model


@pytest.fixture
def tiny():
    torch.manual_seed(1337)
    torch.set_num_threads(2)
    return tiny_model()


@pytest.fixture
def prompts():
    return [
        {
            "id": f"p{i}",
            "prompt": f"question item{i}",
            "split": split,
            "kind": "benign",
            "principle_id": "C13",
        }
        for i, split in enumerate(("train", "validation", "test"))
    ]
