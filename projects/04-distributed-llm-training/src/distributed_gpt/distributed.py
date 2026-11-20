"""Explicit device, precision, DDP, and FSDP setup."""

import os
from dataclasses import dataclass
from datetime import timedelta
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP

from .model import Block


@dataclass
class Context:
    rank: int
    world_size: int
    device: torch.device
    checkpoint_group: object = None

    @property
    def distributed(self):
        return dist.is_initialized()

    def barrier(self):
        if self.distributed:
            dist.barrier()

    def close(self):
        if self.distributed:
            if self.checkpoint_group is not None:
                dist.destroy_process_group(self.checkpoint_group)
            dist.destroy_process_group()


def initialize(strategy, device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if device == "cuda":
        if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
            raise RuntimeError("Requested CUDA device is unavailable")
        torch.cuda.set_device(local_rank)
        selected = torch.device("cuda", local_rank)
    elif device == "cpu":
        selected = torch.device("cpu")
    else:
        raise ValueError("Device must be auto, cpu, or cuda")
    if strategy == "fsdp" and selected.type != "cuda":
        raise RuntimeError("FSDP requires CUDA in this project; use CPU DDP for software checks")
    if strategy == "single" and world != 1:
        raise ValueError("A multi-process launch must use ddp or fsdp")
    if strategy != "single":
        if "RANK" not in os.environ:
            raise ValueError("Launch ddp/fsdp with torchrun, including for a one-rank FSDP check")
        dist.init_process_group(
            "nccl" if selected.type == "cuda" else "gloo", timeout=timedelta(seconds=180)
        )
        # Async checkpoint collectives must not interleave with training collectives.
        checkpoint_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=180))
    else:
        checkpoint_group = None
    return Context(rank, world, selected, checkpoint_group)


def resolve_precision(requested, device):
    if requested == "auto":
        if device.type == "cpu":
            return "fp32"
        return "bf16" if torch.cuda.get_device_capability(device)[0] >= 8 else "fp16"
    if device.type == "cpu" and requested != "fp32":
        raise ValueError("The CPU validation path uses fp32")
    if requested == "bf16" and (
        torch.cuda.get_device_capability(device)[0] < 8 or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("Native BF16 is unavailable; use fp16 or auto on a T4")
    return requested


def enable_activation_checkpointing(model):
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=lambda module: isinstance(module, Block),
    )


def wrap_model(model, strategy, precision, context, checkpointing):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    if strategy == "fsdp":
        mixed = (
            None
            if precision == "fp32"
            else MixedPrecision(param_dtype=dtype, reduce_dtype=torch.float32, buffer_dtype=dtype)
        )
        model = FSDP(
            model,
            device_id=context.device,
            use_orig_params=True,
            auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={Block}),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed,
            limit_all_gathers=True,
            sync_module_states=True,
        )
    elif strategy == "ddp":
        model = DDP(
            model,
            device_ids=[context.device.index] if context.device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    if checkpointing:
        enable_activation_checkpointing(model)
    return model


def make_scaler(strategy, precision, device):
    enabled = precision == "fp16"
    if strategy == "fsdp":
        return ShardedGradScaler(device="cuda", enabled=enabled)
    return torch.amp.GradScaler(device.type, enabled=enabled)
