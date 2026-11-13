# Troubleshooting

| Symptom | What to check |
|---|---|
| `No module named torch` or `gpt2_scratch` | Use the Python interpreter from the environment where you installed the dependencies. Run `python -m pip install -e . --no-deps` from the project folder. |
| PowerShell refuses `Activate.ps1` | Activation is optional. Run `.\.venv\Scripts\python.exe` directly in place of `python`. |
| Dependency installation fails on a new Python release | Use Python 3.11 or 3.12 for the tested dependency pins. |
| `CUDA is unavailable` | Run `python scripts/check_environment.py`. Use a CPU config or install a compatible CUDA build of PyTorch. |
| Prepared-data directory is not empty | Reuse the prepared data. For a different tokenizer or source, choose a new `--out` path. |
| Run directory is not empty | Use a new path for a fresh run, or resume `last.pt` with the original settings. |
| Data or tokenizer hash mismatch | Prepare a fresh directory from the verified source. Do not mix artifacts from different datasets. |
| Resume rejects changed source or settings | Restore the original source and settings, or start a new run. Resume restores the optimizer and random states as well as weights. |
| Training is slow | Run `python scripts/benchmark.py --config configs/char_cpu.json`. Start with the smoke preset and tune CPU thread count in a separate config. |
| Character tokenizer rejects a prompt | Use characters from the training alphabet or train with the BPE tokenizer. |
| Missing checkpoint after cloning | Train first. Checkpoints are not included in the repository. |

When reporting an issue, include the command, config, environment-script output,
and traceback.
