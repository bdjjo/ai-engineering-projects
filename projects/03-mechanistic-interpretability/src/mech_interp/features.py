"""Precompute a bounded feature index from an explicit, separate probe corpus."""

import torch

from .runtime import fingerprint
from .sparse_autoencoder import SAEConfig, SparseAutoencoder


@torch.inference_mode()
def collect_probes(model, texts, hook_name):
    activations, rows = [], []
    for document, text in enumerate(texts):
        tokens = model.to_tokens(text, prepend_bos=True)
        if tokens.shape[1] > model.cfg.n_ctx:
            raise ValueError("Probe text exceeds the model context")
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name], return_type=None)
        activations.append(cache[hook_name][0, 1:].cpu().float())
        pieces = model.to_str_tokens(tokens[0])
        for position in range(1, tokens.shape[1]):
            rows.append(
                {
                    "document": document,
                    "position": position,
                    "before": "".join(pieces[max(1, position - 8) : position]),
                    "token": pieces[position],
                    "after": "".join(pieces[position + 1 : position + 9]),
                }
            )
    return torch.cat(activations), rows


@torch.inference_mode()
def build_feature_index(checkpoint_path, activations, rows, corpus_sha256, top_k=4):
    if len(rows) != len(activations) or top_k < 1 or len(rows) == 0:
        raise ValueError("Invalid feature-index inputs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    scale = checkpoint["scale"]
    if scale <= 0:
        raise ValueError("Invalid checkpoint scale")
    sae = SparseAutoencoder(SAEConfig(**checkpoint["config"]))
    sae.load_state_dict(checkpoint["state"])
    sae.eval()
    # Probe corpus is explicitly bounded; encode in batches, retain only top candidates.
    width = sae.config.hidden_dim
    candidates = min(max(top_k * 12, top_k), len(rows))
    values = torch.full((candidates, width), -torch.inf)
    positions = torch.zeros((candidates, width), dtype=torch.long)
    counts = torch.zeros(width, dtype=torch.long)
    for offset in range(0, len(rows), 128):
        features = sae.encode(activations[offset : offset + 128].float() / scale)
        counts += (features > 0).sum(0)
        new_positions = torch.arange(offset, offset + len(features))[:, None].expand(-1, width)
        combined = torch.cat([values, features])
        all_positions = torch.cat([positions, new_positions])
        values, selection = combined.topk(candidates, dim=0)
        positions = all_positions.gather(0, selection)
    features_out = []
    for feature in range(width):
        examples, documents = [], set()
        for value, position in zip(values[:, feature].tolist(), positions[:, feature].tolist()):
            row = rows[position]
            if value <= 0 or row["document"] in documents:
                continue
            examples.append({**row, "activation": value})
            documents.add(row["document"])
            if len(examples) == top_k:
                break
        features_out.append(
            {
                "id": feature,
                "active_fraction": counts[feature].item() / len(rows),
                "max_activation": max(0.0, values[0, feature].item()),
                "examples": examples,
            }
        )
    return {
        "format_version": 1,
        "hidden_dim": width,
        "probe_tokens": len(rows),
        "probe_documents": len({row["document"] for row in rows}),
        "corpus_sha256": corpus_sha256,
        "checkpoint_sha256": fingerprint(checkpoint_path),
        "model": checkpoint["activation_metadata"]["model"],
        "hook_name": checkpoint["activation_metadata"]["hook_name"],
        "scale": scale,
        "training_config": checkpoint["config"],
        "scope": "Top examples within the supplied probe corpus; no semantic labels validated",
        "features": features_out,
    }
