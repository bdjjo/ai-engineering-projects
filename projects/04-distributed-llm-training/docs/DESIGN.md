# Training and measurement notes

## Model and distribution

The model uses pre-LayerNorm blocks, GELU, causal scaled dot-product attention,
learned positions, and tied input/output embeddings. The `bias=false` setting
removes linear biases; LayerNorm retains its affine weight and bias. With width
`d`, layers `L`, vocabulary `V`, and context `S`, its parameter count is
`(V + S)d + 12Ld² + (4L + 2)d`. Enabling linear biases adds `9Ld` parameters.

Every optimizer attempt processes `B × A × S × W` tokens, where B is microbatch
per rank, A is accumulation, S is context, and W is world size. Every microbatch
loss is divided by A. DDP skips gradient synchronization on earlier microbatches;
its final backward averages the accumulated gradients across ranks. FSDP performs
reduce-scatter on each microbatch to retain sharded gradients. This uses more
communication than FSDP `no_sync`, which would retain full gradients.

FSDP wraps each transformer block with FULL_SHARD and keeps shared embeddings at
the root. Original parameter references are retained with `use_orig_params=True`.
Weight-decay groups are formed **before** FSDP changes local parameter shapes.
Reductions use FP32; mixed-precision parameters use the requested FP16/BF16.
FP16 uses ShardedGradScaler under FSDP and GradScaler for other strategies.
An overflow consumes a data step but skips an optimizer update and is reported.
The schedule and checkpoint cursor count optimizer **attempts**.

Activation checkpointing wraps blocks with non-reentrant recomputation and preserves
RNG for dropout. Recomputed work costs time and is not counted as useful model FLOPs.

## Data and repeatability

Token files are little-endian uint16, validated against the model vocabulary.
Inputs are non-overlapping context-length blocks with targets shifted by one.
The last target token is the next block's first input; train/validation files are
separate. Each epoch uses a deterministic permutation seeded by `seed + epoch`.
Ranks take disjoint slices of a global batch. Incomplete global batches are dropped.
The epoch and batch are computed from the saved absolute microstep, so a resumed
run does not restart its data stream or repeat its first epoch by accident.

WikiText uses its official training/validation splits. OpenWebText-10k uses a
deterministic normalized-text hash bucket split, not a new independent corpus.
Both paths skip short records and exact normalized duplicates, append EOS, truncate
each record to 50,000 characters, and fill bounded token budgets. This does not
remove near duplicates or constitute corpus-wide contamination analysis.

## Asynchronous recovery

PyTorch Distributed Checkpoint saves canonical model and optimizer state. Its
default 2.7.1 blocking stager takes a training-safe CPU copy before returning;
serialization/storage then runs in a background thread. Staging is synchronous
and consumes host RAM. At most one background checkpoint is active per rank.
A dedicated Gloo group separates checkpoint collectives from training collectives.

Each rank also saves RNG state, the AMP scaler and absolute data position. A
manifest records configuration and hashes. The completion marker is written only
after every rank's future and checkpoint-group barrier finish. Resume rejects an
incomplete save or mismatching signature. Background errors propagate to the caller.
The CPU tests mutate model/Adam tensors immediately after save and confirm the
snapshot stays unchanged, then verify exact interrupted-training recovery.

This is a research implementation, not a fault-tolerant training service. It
does not provide automatic preemption handling, elastic restart, checkpoint
retention/garbage collection, or remote object-store durability. Local filesystem
completion does not establish the durability guarantees of a mounted cloud drive.

## MFU and timing

Approximate useful training FLOPs/token:

```text
F = 6 × (N − S × d) + 12 × L × d × S
tokens/update = B × A × S × W
global tokens/s = tokens/update ÷ slowest-rank update duration
MFU (%) = 100 × global tokens/s × F ÷ (W × peak_TFLOPS_per_GPU × 10^12)
```

N is the unique parameter count. Position embeddings are excluded from the 6N
matmul estimate; the second term approximates sequence-dependent attention work.
This is an estimate, not an operation-by-operation FLOP counter. It excludes
activation recomputation, optimizer work, and communication from the FLOP numerator.
There is no claim that SDPA always selects the same kernel on different hardware.

Updates synchronize CUDA before/after timing and use the maximum duration across
ranks. The timed region includes data fetching, forward/backward, gradient
communication, clipping, and optimizer work. Metric collectives, logging,
validation, staging/join, and the pre-step barrier are excluded. Separate loop-wall
throughput includes barriers, metrics, logging, profiling hooks and checkpoint work,
but excludes model/data initialization and validation. Aggregate throughput uses
total tokens divided by total measured time, after the configured warmup exclusion.
Resumed runs report their own segment; they do not fabricate a combined benchmark.

The built-in peaks are **dense** Tensor Core values: A100 FP16/BF16 312 TFLOP/s,
T4 mixed FP16/FP32 65 TFLOP/s. Sparse peaks are inappropriate here. Unknown GPU
models, MIG partitions, and unsupported precision combinations report null MFU
unless an explicit, documented `--peak-tflops` is supplied. CPU MFU is always null.
The code does not clamp MFU at 100%; an impossible value should expose an incorrect
peak/FLOP assumption or measurement, not hide it. GPU peak memory is the maximum
allocated/reserved across ranks after training begins, not a whole-process startup peak.

## References

- [PyTorch 2.7 FSDP API](https://docs.pytorch.org/docs/2.7/fsdp.html)
- [PyTorch 2.7 Distributed Checkpoint](https://docs.pytorch.org/docs/2.7/distributed.checkpoint.html)
- [Asynchronous checkpoint recipe](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html)
  (newer optional staging APIs in the live tutorial are not required here).
- [PyTorch FSDP paper](https://arxiv.org/abs/2304.11277)
- [NVIDIA A100 datasheet](https://images.nvidia.com/data-center/a100/a100-datasheet.pdf)
- [NVIDIA T4 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf)
