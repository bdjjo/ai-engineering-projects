"""Build a source-only portfolio ZIP using an explicit allowlist."""

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = [
    "README.md",
    "LICENSE",
    "CONTRIBUTING.md",
    ".gitignore",
    ".gitattributes",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-lock-windows-py312.txt",
    "train.py",
    "generate.py",
    "evaluate.py",
]
DIRECTORIES = ["src/gpt2_scratch", "scripts", "tests", "configs", "docs", "reports", ".github"]
EXTENSIONS = {".py", ".md", ".txt", ".json", ".jsonl", ".csv", ".png", ".yml"}
paths = [ROOT / name for name in ROOT_FILES]
paths.append(ROOT / "data" / "README.md")
for directory in DIRECTORIES:
    paths.extend(
        path
        for path in (ROOT / directory).rglob("*")
        if path.is_file() and path.suffix in EXTENSIONS and "__pycache__" not in path.parts
    )
paths = sorted(set(paths))
if any(not path.is_file() for path in paths):
    raise FileNotFoundError("a required release file is missing")
destination = ROOT / "dist"
destination.mkdir(exist_ok=True)
output = destination / "gpt2-from-scratch-portfolio.zip"
hashes = {
    path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in paths
}
with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in paths:
        archive.write(path, "gpt2-from-scratch/" + path.relative_to(ROOT).as_posix())
    archive.writestr("gpt2-from-scratch/SHA256SUMS.json", json.dumps(hashes, indent=2) + "\n")
print(f"Packaged {len(paths)} files, {output.stat().st_size:,} bytes: {output}")
