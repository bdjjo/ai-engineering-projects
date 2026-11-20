"""Bounded asynchronous DCP saves, optimizer restore, and rank-local RNG snapshots."""

import json
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)

from .data import write_json

OPTIONS = StateDictOptions(full_state_dict=False, cpu_offload=False)


class CheckpointManager:
    def __init__(self, context, root, signature):
        self.context, self.root, self.signature = context, Path(root), signature
        self.future = None
        self.path = None
        self.records = []

    def wait(self):
        if self.future is None:
            return
        start = time.perf_counter()
        self.future.result()  # Propagate background failures before claiming completion.
        if self.context.distributed:
            dist.barrier(group=self.context.checkpoint_group)
        if self.context.rank == 0:
            write_json(self.path / "COMPLETED.json", {"complete": True})
        self.records[-1]["join_seconds"] = time.perf_counter() - start
        self.future = None

    def save(self, model, optimizer, scaler, step, accumulation):
        self.wait()  # At most one staged checkpoint occupies host RAM per rank.
        start = time.perf_counter()
        path = self.root / f"step-{step:06d}"
        if self.context.rank == 0:
            path.mkdir(parents=True, exist_ok=False)
            write_json(path / "manifest.json", {"step": step, "signature": self.signature})
        self.context.barrier()
        runtime = {
            "torch_rng": torch.get_rng_state(),
            "scaler": scaler.state_dict(),
            "microstep": step * accumulation,
            "cuda_rng": torch.cuda.get_rng_state(self.context.device)
            if self.context.device.type == "cuda"
            else None,
        }
        torch.save(runtime, path / f"rank-{self.context.rank}.pt")
        model_state, optim_state = get_state_dict(model, optimizer, options=OPTIONS)
        self.future = dcp.async_save(
            {"model": model_state, "optimizer": optim_state},
            checkpoint_id=path,
            process_group=self.context.checkpoint_group,
        )
        self.path = path
        self.records.append({"step": step, "stage_seconds": time.perf_counter() - start})


def load_checkpoint(path, model, optimizer, scaler, context, signature, accumulation):
    path = Path(path)
    if not (path / "COMPLETED.json").is_file():
        raise ValueError("Checkpoint has no completion marker; do not resume a partial save")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["signature"] != signature:
        raise ValueError(
            "Resume requires matching config, data, source, precision, strategy and world size"
        )
    model_state, optim_state = get_state_dict(model, optimizer, options=OPTIONS)
    state = {"model": model_state, "optimizer": optim_state}
    dcp.load(state, checkpoint_id=path, process_group=context.checkpoint_group)
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
        options=OPTIONS,
    )
    runtime = torch.load(path / f"rank-{context.rank}.pt", map_location="cpu", weights_only=True)
    if runtime["microstep"] != manifest["step"] * accumulation:
        raise ValueError("Checkpoint/data cursor mismatch")
    scaler.load_state_dict(runtime["scaler"])
    torch.set_rng_state(runtime["torch_rng"])
    if runtime["cuda_rng"] is not None:
        torch.cuda.set_rng_state(runtime["cuda_rng"], context.device)
    return manifest["step"]
