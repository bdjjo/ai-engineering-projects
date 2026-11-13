"""Small runtime utilities; no online experiment-tracking account is required."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import torch


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA is unavailable; install a matching PyTorch build or use --device cpu"
        )
    return torch.device(requested)


def configure_runtime(seed: int, threads: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False


def environment_info(device: torch.device) -> dict:
    return {
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "os": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "accelerator": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": "float32",
        "deterministic_algorithms": True,
    }


def source_hashes() -> dict[str, str]:
    """Identify training source without exposing local user paths."""
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "src" / "gpt2_scratch").glob("*.py"))
    paths += [
        root / name for name in ("train.py", "evaluate.py", "generate.py", "requirements.txt")
    ]
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
        if path.is_file()
    }


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def atomic_torch_save(payload: dict, path: str | Path) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path) -> dict:
    # Store only tensors and primitive containers, compatible with weights_only=True.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    return checkpoint


def cli_error(error: Exception) -> None:
    print(f"Error: {error}", file=sys.stderr)
    raise SystemExit(2)
