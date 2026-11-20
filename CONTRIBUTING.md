# Contributing

Run commands from the folder of the project you are changing. Dependencies belong
to that project so experiments can evolve independently.

## GPT-2 from scratch

Follow the [installation instructions](projects/01-gpt2-from-scratch/README.md).
From the repository root, with that environment active:

```bash
cd projects/01-gpt2-from-scratch
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

The tests use small synthetic datasets and run on a CPU without downloading training
data. Add behavioral tests when changing tokenization, attention, evaluation, or
checkpoint handling. For experiments, include the configuration, seed, data
provenance, evaluation protocol, and unedited outputs needed to assess the result.

## Adding a project

The collection is planned for 12 projects on `main`. Use the next numbered directory,
using `projects/NN-short-name/`; keep existing paths stable.

Each project should contain:

- A README with the purpose, setup, runnable commands, and known limitations.
- Its own dependency manifest and source code.
- Tests for the behavior that matters and a matching job in `.github/workflows/ci.yml`.
- Small result artifacts with enough context to reproduce any reported measurement.

Add a row to the root README when the project is ready to run. Keep datasets,
checkpoints, credentials, and machine-specific environments out of commits.

For bug reports, include the command, configuration, Python and package versions,
and traceback. For changes, explain the behavior before and after the patch and
which checks you ran.
