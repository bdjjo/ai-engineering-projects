"""Shared model convention and portable experiment provenance."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import torch

MODEL_ID = "openai-community/gpt2"
MODEL_OPTIONS = {
    "center_writing_weights": True,
    "center_unembed": True,
    "fold_ln": True,
    "refactor_factored_attn_matrices": False,
}


def configure(seed=1337, threads=4):
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


def environment():
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformer-lens", "transformers", "datasets", "gradio")
        },
        "threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
    }


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def fresh_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}. Choose a new --out.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_model(device="cpu", revision=None):
    from huggingface_hub import HfApi
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer

    revision = revision or HfApi().model_info(MODEL_ID).sha
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    model = HookedTransformer.from_pretrained(
        "gpt2",
        revision=revision,
        tokenizer=tokenizer,
        device=device,
        dtype=torch.float32,
        **MODEL_OPTIONS,
    ).eval()
    model.requires_grad_(False)
    provenance = {"model_id": MODEL_ID, "revision": revision, "options": MODEL_OPTIONS}
    return model, provenance
