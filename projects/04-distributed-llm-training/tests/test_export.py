import json
import os
import subprocess
import sys
from pathlib import Path


def test_export_ignores_unavailable_inherited_backend(tmp_path):
    project = Path(__file__).resolve().parents[1]
    reference = json.loads((project / "reports/cpu-single/results.json").read_text())
    source = tmp_path / "run"
    source.mkdir()
    for name in ("metadata", "summary", "history"):
        (source / f"{name}.json").write_text(json.dumps(reference[name]))
    destination = tmp_path / "report"
    # An intentionally absent backend reproduces Colab's separate-environment failure.
    env = dict(os.environ, MPLBACKEND="module://unavailable_colab_backend.backend_inline")
    subprocess.run(
        [
            sys.executable,
            str(project / "scripts/export_report.py"),
            "--run",
            str(source),
            "--out",
            str(destination),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert json.loads((destination / "results.json").read_text()) == reference
    assert (destination / "training.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
