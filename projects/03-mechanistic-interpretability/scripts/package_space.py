"""Package a small Gradio Space without model weights or training dependencies."""

import argparse
import json
import shutil
from pathlib import Path

from mech_interp.runtime import fresh_directory

ROOT = Path(__file__).resolve().parents[1]


def package(index_path, out):
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    if len(index["features"]) != index["hidden_dim"]:
        raise ValueError("Incomplete feature index")
    out = fresh_directory(out)
    shutil.copyfile(index_path, out / "feature_index.json")
    shutil.copyfile(ROOT / "src/mech_interp/feature_browser.py", out / "browser.py")
    (out / "app.py").write_text(
        "from browser import build_browser\n\n"
        'demo = build_browser("feature_index.json")\n'
        'if __name__ == "__main__":\n'
        "    demo.launch()\n",
        encoding="utf-8",
    )
    (out / "requirements.txt").write_text("gradio==5.49.1\n", encoding="utf-8")
    (out / "README.md").write_text(
        "---\ntitle: GPT-2 Feature Explorer\nemoji: 🔬\ncolorFrom: blue\ncolorTo: indigo\n"
        "sdk: gradio\nsdk_version: 5.49.1\npython_version: '3.12'\napp_file: app.py\n"
        "pinned: false\nlicense: mit\n---\n\n"
        "# GPT-2 Feature Explorer\n\n"
        "A precomputed browser for a small sparse-autoencoder experiment. "
        "Examples come from 32 original probe paragraphs. "
        "Feature meanings have not been validated.\n\n"
        "[Code, training settings, and limitations](https://github.com/bdjjo/ai-engineering-projects/"
        "tree/main/projects/03-mechanistic-interpretability).\n\n"
        "No model weights, training data, or credentials are included in this Space.\n",
        encoding="utf-8",
    )
    shutil.copyfile(ROOT.parents[1] / "LICENSE", out / "LICENSE")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="reports/cpu-pilot/feature_index.json")
    parser.add_argument("--out", default="space-build")
    args = parser.parse_args()
    print(package(args.index, args.out))


if __name__ == "__main__":
    main()
