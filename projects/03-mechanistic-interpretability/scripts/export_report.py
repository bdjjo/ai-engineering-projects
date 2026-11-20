"""Export small measured artifacts, preserving every supplied SAE comparison."""

import argparse
import csv
import json
import shutil
from pathlib import Path

from mech_interp.runtime import fresh_directory, write_json


def export(ioi_path, sae_path, index_path, comparisons, out):
    ioi_path, sae_path = Path(ioi_path), Path(sae_path)
    ioi = json.loads((ioi_path / "results.json").read_text())
    chosen = json.loads((sae_path / "summary.json").read_text())
    index = json.loads(Path(index_path).read_text())
    if index["checkpoint_sha256"] != chosen["checkpoint_sha256"]:
        raise ValueError("Feature index does not belong to the selected SAE checkpoint")
    records = []
    for path in [*map(Path, comparisons), sae_path]:
        report = json.loads((path / "summary.json").read_text())
        if report["activation_sha256"] != chosen["activation_sha256"]:
            raise ValueError("SAE runs do not share the same activation cache")
        records.append((path, report))
    out = fresh_directory(out)
    for source, destination in [
        (ioi_path / "results.json", "ioi_results.json"),
        (ioi_path / "ioi_patching.png", "ioi_patching.png"),
        (sae_path / "summary.json", "sae_summary.json"),
        (sae_path / "history.json", "sae_history.json"),
        (Path(index_path), "feature_index.json"),
    ]:
        shutil.copyfile(source, out / destination)
    rows = []
    for number, (path, report) in enumerate(records):
        target = out / "sae_comparisons" / str(number + 1)
        target.mkdir(parents=True)
        shutil.copyfile(path / "summary.json", target / "summary.json")
        shutil.copyfile(path / "history.json", target / "history.json")
        config, metrics = report["config"], report["final_validation"]
        rows.append(
            {
                "run": number + 1,
                "l1_coefficient": config["l1_coefficient"],
                "steps": config["steps"],
                "hidden_dim": config["hidden_dim"],
                "variance_explained": metrics["variance_explained"],
                "mse_per_dimension": metrics["mse_per_dimension"],
                "l0": metrics["l0"],
                "inactive_fraction": metrics["inactive_fraction"],
                "wall_seconds": report["wall_seconds"],
                "used_for_browser": path == sae_path,
            }
        )
    with (out / "sae_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        out / "manifest.json",
        {
            "description": "Measured GPT-2 IOI sweep and CPU SAE comparison",
            "ioi_prompts": ioi["n_prompts"],
            "sae_runs": len(rows),
            "selection": "Validation-guided choice of moderate L1, then longer training",
            "independent_test_evaluation": False,
            "probe_corpus": "examples/browser_corpus.json; original AI-assisted paragraphs",
            "deployment_target": "bdjjo/gpt2-feature-explorer",
            "space_deployment_status": "Packaged; authenticated publication not yet verified",
        },
    )
    print(f"Exported {len(rows)} SAE runs and IOI measurements to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ioi", required=True)
    parser.add_argument("--sae", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--comparisons", nargs="*", default=[])
    parser.add_argument("--out", default="reports/cpu-pilot")
    args = parser.parse_args()
    export(args.ioi, args.sae, args.index, args.comparisons, args.out)


if __name__ == "__main__":
    main()
