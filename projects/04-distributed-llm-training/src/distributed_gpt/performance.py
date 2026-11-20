"""Estimated model FLOPs and explicitly bounded timing measurements."""

import math

from .model import parameter_count_formula


def flops_per_token(config):
    # 6N approximation plus the sequence-dependent attention term. Recompute is excluded.
    non_position = parameter_count_formula(config) - config.block_size * config.n_embd
    return 6 * non_position + 12 * config.n_layer * config.n_embd * config.block_size


def peak_tflops(device_name, precision, override=None):
    if override is not None:
        if not math.isfinite(override) or override <= 0:
            raise ValueError("Peak TFLOP/s must be finite and positive")
        return override, "User-supplied dense peak for the selected arithmetic precision"
    if "MIG" in device_name.upper():
        return None, "MIG partition: supply its own dense peak explicitly"
    if "A100" in device_name and precision in {"bf16", "fp16"}:
        return 312.0, "NVIDIA A100 datasheet; dense FP16/BF16 Tensor Core peak"
    if "T4" in device_name and precision == "fp16":
        return 65.0, "NVIDIA T4 datasheet; dense mixed FP16/FP32 Tensor Core peak"
    return None, "Unknown device/precision: MFU omitted; no guessed peak"


def step_metrics(
    seconds,
    batch_size,
    accumulation,
    sequence_length,
    world_size,
    model_flops_per_token,
    peak_per_device=None,
):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Step duration must be finite and positive")
    if min(batch_size, accumulation, sequence_length, world_size, model_flops_per_token) <= 0:
        raise ValueError("Token and FLOP counts must be positive")
    tokens = batch_size * accumulation * sequence_length * world_size
    throughput = tokens / seconds
    mfu = None
    if peak_per_device is not None:
        if not math.isfinite(peak_per_device) or peak_per_device <= 0:
            raise ValueError("Invalid device peak")
        mfu = 100 * throughput * model_flops_per_token / (world_size * peak_per_device * 1e12)
    return {
        "seconds": seconds,
        "global_tokens": tokens,
        "tokens_per_second": throughput,
        "tokens_per_second_per_rank": throughput / world_size,
        "mfu_percent": mfu,
    }


def summarize(records, warmup_steps):
    eligible = [r for r in records if r["step"] > warmup_steps]
    if not eligible:
        return {"measured_steps": 0, "tokens_per_second": None, "mfu_percent": None}
    duration = sum(r["seconds"] for r in eligible)
    tokens = sum(r["global_tokens"] for r in eligible)
    mfu = (
        None
        if eligible[0]["mfu_percent"] is None
        else sum(r["mfu_percent"] * r["seconds"] for r in eligible) / duration
    )
    return {
        "measured_steps": len(eligible),
        "timed_seconds": duration,
        "global_tokens": tokens,
        "tokens_per_second": tokens / duration,
        "mfu_percent": mfu,
    }
