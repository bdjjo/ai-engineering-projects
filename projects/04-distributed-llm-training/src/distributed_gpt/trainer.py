"""One training loop shared by single-device, DDP, and FSDP execution."""

import hashlib
import importlib.metadata
import platform
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist

from .checkpoint import CheckpointManager, load_checkpoint
from .config import learning_rate
from .data import RankBatchStream, TokenDataset, file_hash, write_json
from .distributed import initialize, make_scaler, resolve_precision, wrap_model
from .model import GPT, optimizer_groups
from .performance import flops_per_token, peak_tflops, step_metrics, summarize


def source_hashes():
    directory = Path(__file__).parent
    return {
        p.name: hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest()
        for p in sorted(directory.glob("*.py"))
    }


def autocast(device, precision):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


@torch.inference_mode()
def validate(model, stream, count, context, precision):
    model.eval()
    total = torch.zeros((), device=context.device)
    for i in range(count):
        x, y = stream.batch(i, context.device)
        with autocast(context.device, precision):
            _, loss = model(x, y)
        total += loss.detach().float()
    if context.distributed:
        dist.all_reduce(total)
    model.train()
    return total.item() / (count * context.world_size)


def run_training(
    config,
    data_dir,
    out,
    strategy="single",
    device="auto",
    threads=2,
    resume=None,
    stop_after=None,
    profile=False,
    peak_override=None,
):
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    context = initialize(strategy, device)
    manager = None
    try:
        precision = resolve_precision(config.train.precision, context.device)
        out, data_dir = Path(out), Path(data_dir)
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(
                "Output directory is not empty; choose a fresh --out, including on resume"
            )
        if context.rank == 0:
            out.mkdir(parents=True, exist_ok=True)
        context.barrier()
        train_data = TokenDataset(
            data_dir / "train.bin", config.model.block_size, config.model.vocab_size
        )
        val_data = TokenDataset(
            data_dir / "validation.bin", config.model.block_size, config.model.vocab_size
        )
        kwargs = dict(
            batch_size=config.train.batch_size, rank=context.rank, world_size=context.world_size
        )
        stream = RankBatchStream(train_data, seed=config.train.seed, **kwargs)
        val_stream = RankBatchStream(val_data, seed=config.train.seed + 10000, **kwargs)
        signature = {
            "config": config.to_dict(),
            "world_size": context.world_size,
            "strategy": strategy,
            "precision": precision,
            "source_sha256": source_hashes(),
            "data_sha256": {
                split: file_hash(data_dir / f"{split}.bin") for split in ("train", "validation")
            },
            "torch": str(torch.__version__),
        }
        torch.manual_seed(config.train.seed)
        with torch.device(context.device):
            raw = GPT(config.model)
        n_params = raw.parameter_count()
        groups = optimizer_groups(raw, config.train.weight_decay)
        model = wrap_model(raw, strategy, precision, context, config.train.activation_checkpointing)
        optimizer = torch.optim.AdamW(
            groups, lr=config.train.learning_rate, betas=(0.9, 0.95), foreach=False
        )
        scaler = make_scaler(strategy, precision, context.device)
        torch.manual_seed(config.train.seed + context.rank)
        start_step = 0
        if resume:
            start_step = load_checkpoint(
                resume, model, optimizer, scaler, context, signature, config.train.accumulation
            )
        end_step = config.train.steps if stop_after is None else min(stop_after, config.train.steps)
        if not start_step < end_step:
            raise ValueError("The requested run contains no new updates")
        initial_val = validate(
            model, val_stream, config.train.validation_batches, context, precision
        )
        gpu_name = (
            torch.cuda.get_device_name(context.device) if context.device.type == "cuda" else None
        )
        if context.device.type == "cpu":
            if peak_override is not None:
                raise ValueError("GPU MFU is unavailable for CPU runs")
            peak, peak_source = None, "CPU run: GPU MFU not measured"
        else:
            peak, peak_source = peak_tflops(gpu_name, precision, peak_override)
        if context.distributed:
            hardware = [None] * context.world_size
            dist.all_gather_object(hardware, (gpu_name, peak))
            if len(set(hardware)) != 1:
                raise ValueError("MFU aggregation currently requires homogeneous devices and peaks")
        metadata = {
            **signature,
            "parameter_count": n_params,
            "device": str(context.device),
            "gpu_name": gpu_name,
            "peak_tflops_per_device": peak,
            "peak_source": peak_source,
            "model_flops_per_token_estimate": flops_per_token(config.model),
            "effective_sharding": "FULL_SHARD"
            if strategy == "fsdp" and context.world_size > 1
            else "No cross-rank parameter sharding",
            "python": platform.python_version(),
            "platform": platform.system(),
            "threads_per_rank": threads,
            "numpy": importlib.metadata.version("numpy"),
            "profile_enabled": profile,
            "resumed_from_step": start_step,
            "initial_validation_loss": initial_val,
            "timing_scope": "Data fetch, forward/backward and optimizer; slowest rank per step. "
            "Excludes metric collectives, validation and checkpoint staging/join.",
            "flops_scope": "6N excluding position embeddings + 12*layers*width*sequence; "
            "approximate dense attention; excludes activation recomputation",
        }
        if context.rank == 0:
            write_json(out / "metadata.json", metadata)
        manager = CheckpointManager(context, out / "checkpoints", signature)
        if profile:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if context.device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            profiler = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
                on_trace_ready=lambda p: p.export_chrome_trace(
                    str(out / f"trace-rank{context.rank}.json")
                ),
                record_shapes=True,
                profile_memory=True,
            )
        else:
            profiler = None
        if context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(context.device)
        history = []
        model.train()
        context.barrier()
        wall_start = time.perf_counter()
        with profiler if profiler is not None else nullcontext():
            for update in range(start_step, end_step):
                lr = learning_rate(update, config.train)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                context.barrier()
                if context.device.type == "cuda":
                    torch.cuda.synchronize(context.device)
                timer = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss_sum = torch.zeros((), device=context.device)
                for micro in range(config.train.accumulation):
                    x, y = stream.batch(update * config.train.accumulation + micro, context.device)
                    # FSDP synchronizes each microstep to keep gradients sharded.
                    sync = (
                        model.no_sync()
                        if strategy == "ddp" and micro < config.train.accumulation - 1
                        else nullcontext()
                    )
                    with sync:
                        with autocast(context.device, precision):
                            _, loss = model(x, y)
                            scaled_loss = loss / config.train.accumulation
                        scaler.scale(scaled_loss).backward()
                    loss_sum += scaled_loss.detach().float()
                scaler.unscale_(optimizer)
                if strategy == "fsdp":
                    grad_norm = model.clip_grad_norm_(config.train.grad_clip)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.train.grad_clip
                    )
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < previous_scale
                if context.device.type == "cuda":
                    torch.cuda.synchronize(context.device)
                elapsed = torch.tensor(
                    time.perf_counter() - timer, dtype=torch.float64, device=context.device
                )
                if context.distributed:
                    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                    dist.all_reduce(loss_sum)
                loss_value = loss_sum.item() / context.world_size
                if not torch.isfinite(loss_sum):
                    raise FloatingPointError("Non-finite globally averaged loss")
                row = {
                    "step": update + 1,
                    "loss": loss_value,
                    "learning_rate": lr,
                    "grad_norm": float(grad_norm) if torch.isfinite(grad_norm) else None,
                    "optimizer_step_skipped": skipped,
                    **step_metrics(
                        elapsed.item(),
                        config.train.batch_size,
                        config.train.accumulation,
                        config.model.block_size,
                        context.world_size,
                        flops_per_token(config.model),
                        peak,
                    ),
                }
                history.append(row)
                if profiler is not None:
                    profiler.step()
                if context.rank == 0 and (
                    (update + 1) % config.train.log_every == 0 or update + 1 == end_step
                ):
                    print(
                        f"Step {update + 1:5d} | loss {loss_value:.4f} | "
                        f"{row['tokens_per_second']:,.0f} global tok/s | MFU {row['mfu_percent']}",
                        flush=True,
                    )
                    write_json(out / "history.json", history)
                if (update + 1) % config.train.save_every == 0 or update + 1 == end_step:
                    manager.save(model, optimizer, scaler, update + 1, config.train.accumulation)
        manager.wait()
        context.barrier()
        wall_seconds = time.perf_counter() - wall_start
        final_val = validate(model, val_stream, config.train.validation_batches, context, precision)
        peak_memory = None
        if context.device.type == "cuda":
            peak_memory = torch.tensor(
                [
                    torch.cuda.max_memory_allocated(context.device),
                    torch.cuda.max_memory_reserved(context.device),
                ],
                device=context.device,
            )
            if context.distributed:
                dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
            peak_memory = {
                "max_rank_allocated_bytes": int(peak_memory[0]),
                "max_rank_reserved_bytes": int(peak_memory[1]),
            }
        summary = {
            "start_step": start_step,
            "end_step": end_step,
            "initial_validation_loss": initial_val,
            "final_validation_loss": final_val,
            "performance": summarize(history, max(config.train.measure_after, start_step)),
            "loop_wall_seconds_including_checkpoints": wall_seconds,
            "loop_tokens_per_second": sum(r["global_tokens"] for r in history) / wall_seconds,
            "skipped_optimizer_steps": sum(r["optimizer_step_skipped"] for r in history),
            "cuda_peak_memory": peak_memory,
            "checkpoints": manager.records,
            "profile_enabled": profile,
            "gpu_mfu_measured": peak is not None,
            "completed": True,
        }
        if context.rank == 0:
            write_json(out / "summary.json", summary)
            write_json(out / "history.json", history)
            if profiler is not None:
                (out / "profiler.txt").write_text(
                    profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=15),
                    encoding="utf-8",
                )
        return summary
    finally:
        try:
            if manager is not None:
                manager.wait()
        finally:
            context.close()
