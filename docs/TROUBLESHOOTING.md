# Troubleshooting

| Symptom | Explanation and next action |
|---|---|
| `No module named torch` or `No module named gpt2_scratch` | Check which interpreter is running. Use `.\.venv\Scripts\python.exe`; install requirements and run `python -m pip install -e . --no-deps` with that interpreter. |
| Default Python is 3.15 | The pinned environment uses Python 3.12. Use the project's existing `.venv`, or install Python 3.12 before creating a new environment. |
| PowerShell refuses `Activate.ps1` | Activation is optional. Invoke `.\.venv\Scripts\python.exe` directly instead of changing machine-wide execution policy. |
| `CUDA is unavailable` | The measured environment uses a CPU build. Use the CPU config, or install a compatible CUDA build and verify it with `scripts/check_environment.py`. |
| Prepared-data directory is not empty | It was already prepared. Reuse it, or choose a new `--out` path for a new tokenizer experiment. |
| Run directory is not empty | Use a new `--out` path for a fresh experiment or resume the existing `last.pt` with the original settings. |
| Dataset/tokenizer hash mismatch | The artifacts differ from the manifest. Prepare a new dataset directory from the verified source instead of mixing files from different experiments. |
| Resume rejects changed source or settings | Exact resume is deliberately strict. Restore the original source/settings or start a new experiment. |
| Training is slow | Measure with `scripts/benchmark.py`; use the smoke config for learning. Compare thread counts in separate config files. No runtime is guaranteed across computers. |
| Character tokenizer rejects a prompt | Use characters from the training alphabet, or use the BPE model for arbitrary UTF-8 input. |
| Generated text is repetitive or incoherent | This is a small model on a narrow corpus. Compare temperature settings and inspect held-out loss; a different seed alone is not evidence of improvement. |
| Loss becomes nonfinite | The training loop stops instead of recording misleading metrics. Check the config, learning rate, gradient behavior, and data; reproduce on the smoke config first. |
| Pytest cannot use the system temp folder in the Codex Windows sandbox | Create `.local` and run `python -m pytest -q --basetemp=.local/pytest-temp`. |
| Missing checkpoint after cloning | Checkpoints are excluded from Git. Run training first, or load a separately provided release artifact if one exists. |

If you report an issue, include the exact command, configuration, environment-script
output, and the error. Avoid including private credentials or unrelated local files.
