"""Seeded SAE training, train-only scaling, held-out diagnostics, safe checkpoints."""

import time
from dataclasses import asdict

import torch

from .runtime import configure, environment, fingerprint, fresh_directory, write_json
from .sparse_autoencoder import SparseAutoencoder, evaluate


def train_sae(activation_path, config, out, device="cpu", threads=4):
    configure(config.seed, threads)
    bundle = torch.load(activation_path, map_location="cpu", weights_only=True)
    train, val = bundle["train"], bundle["validation"]
    if train.ndim != 2 or val.ndim != 2 or train.shape[1] != val.shape[1]:
        raise ValueError("Expected aligned train/validation matrices")
    if (
        len(train) < 2
        or len(val) < 2
        or not torch.isfinite(train).all()
        or not torch.isfinite(val).all()
    ):
        raise ValueError("Need finite, nonempty training and validation activations")
    if config.d_model != train.shape[1]:
        raise ValueError("SAE width does not match the activations")
    out = fresh_directory(out)
    # Accumulate in chunks, avoiding a second float32 copy of a large activation store.
    total_abs = 0.0
    total = torch.zeros(config.d_model)
    for chunk in train.split(4096):
        chunk = chunk.float()
        total_abs += chunk.abs().sum().item()
        total += chunk.sum(0)
    scale = total_abs / train.numel()
    if scale <= 1e-8:
        raise ValueError("Activation scale is zero")
    mean = total / (len(train) * scale)
    sae = SparseAutoencoder(config).to(device)
    with torch.no_grad():
        sae.b_dec.copy_(mean.to(device))
    optimizer = torch.optim.Adam(sae.parameters(), lr=config.lr)
    generator = torch.Generator().manual_seed(config.seed)
    history = []
    start = time.perf_counter()
    initial = evaluate(sae, val, scale, mean)
    history.append({"step": 0, "validation": initial})
    seen = torch.zeros(config.hidden_dim, dtype=torch.bool, device=device)
    for step in range(1, config.steps + 1):
        indices = torch.randint(len(train), (config.batch_size,), generator=generator)
        x = train[indices].to(device).float() / scale
        loss, _, features = sae(x)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        seen |= (features.detach() > 0).any(0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        sae.project_decoder_gradient()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
        optimizer.step()
        sae.normalize_decoder()
        if step % config.eval_every == 0 or step == config.steps:
            metrics = evaluate(sae, val, scale, mean)
            row = {
                "step": step,
                "train_batch_loss": loss.item(),
                "validation": metrics,
                "inactive_in_train_window": (~seen).float().mean().item(),
                "elapsed_seconds": time.perf_counter() - start,
            }
            history.append(row)
            seen.zero_()
            write_json(out / "history.json", history)
            print(
                f"Step {step:5d} | val MSE {metrics['mse_per_dimension']:.4f} | "
                f"L0 {metrics['l0']:.1f} | variance explained "
                f"{metrics['variance_explained']:.1%}",
                flush=True,
            )
    checkpoint = {
        "format_version": 1,
        "state": {key: value.detach().cpu() for key, value in sae.state_dict().items()},
        "config": asdict(config),
        "scale": scale,
        "train_mean": mean,
        "step": config.steps,
        "activation_metadata": bundle["metadata"],
    }
    torch.save(checkpoint, out / "sae_final.pt")
    summary = {
        "config": asdict(config),
        "scale": scale,
        "device": device,
        "environment": environment(),
        "activation_metadata": bundle["metadata"],
        "activation_sha256": fingerprint(activation_path),
        "checkpoint_sha256": fingerprint(out / "sae_final.pt"),
        "initial_validation": initial,
        "final_validation": history[-1]["validation"],
        "wall_seconds": time.perf_counter() - start,
        "selection": "Final step of a fixed budget; no validation checkpoint selection",
        "objective": "Mean token squared L2 reconstruction error + lambda * mean token feature L1",
        "sampling": "Uniform token sampling with replacement; constant learning rate",
    }
    write_json(out / "summary.json", summary)
    return summary
