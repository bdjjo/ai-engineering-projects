# Contributing

Use Python 3.11 or 3.12 and follow the installation commands in the README.
Keep the implementation readable: this project is intended to teach the math.

Before opening a pull request, run:

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

Tests use tiny synthetic text and do not download data or require a GPU.
For changes to attention, preserve the causal-mask test and agreement between
manual attention and SDPA, including gradients. For training changes, preserve
the exact CPU resume test. Save experimental runs into new directories.

Report failures with the command, Python/PyTorch versions, config, and traceback.
Do not commit virtual environments, raw corpora, private credentials, or weights.
Use small JSON/CSV logs and plots for experiment evidence.
