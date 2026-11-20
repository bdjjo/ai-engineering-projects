"""Build browser examples using the checkpoint's exact model, hook, and scale."""

import argparse
import json
from pathlib import Path

import torch

from mech_interp.features import build_feature_index, collect_probes
from mech_interp.runtime import MODEL_OPTIONS, configure, fingerprint, load_model, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/sae/sae_final.pt")
    parser.add_argument("--corpus", default="examples/browser_corpus.json")
    parser.add_argument("--out", default="outputs/feature_index.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError("Feature index already exists; choose a new --out")
    configure(threads=args.threads)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    provenance = ckpt["activation_metadata"]["model"]
    if provenance["options"] != MODEL_OPTIONS:
        raise ValueError("Checkpoint/model processing conventions do not match")
    model, loaded = load_model(args.device, provenance["revision"])
    if loaded != provenance:
        raise ValueError("Checkpoint/model provenance does not match")
    texts = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    acts, rows = collect_probes(model, texts, ckpt["activation_metadata"]["hook_name"])
    del model
    result = build_feature_index(args.checkpoint, acts, rows, fingerprint(args.corpus))
    write_json(args.out, result)
    print(f"Indexed {result['hidden_dim']} features on {result['probe_tokens']} probe tokens")


if __name__ == "__main__":
    main()
