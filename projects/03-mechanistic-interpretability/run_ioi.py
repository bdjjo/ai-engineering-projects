"""Run a GPT-2 Small head sweep and save the exact measurements."""

import argparse
import time

from mech_interp.ioi_circuit import plot_effects, run_activation_patching
from mech_interp.runtime import configure, environment, fresh_directory, load_model, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/ioi")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--revision")
    args = parser.parse_args()
    configure(threads=args.threads)
    out = fresh_directory(args.out)
    model, provenance = load_model(args.device, args.revision)
    start = time.perf_counter()
    result = run_activation_patching(model, batch_size=args.batch_size)
    result.update(
        model=provenance,
        environment=environment(),
        device=args.device,
        wall_seconds=time.perf_counter() - start,
    )
    write_json(out / "results.json", result)
    plot_effects(result, out / "ioi_patching.png")
    print(f"Clean logit difference: {result['clean_logit_diff']:.4f}")
    for head in result["top_heads_by_damage"]:
        print(head)


if __name__ == "__main__":
    main()
