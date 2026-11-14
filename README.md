# AI Engineering Projects

[![Tests](https://github.com/bdjjo/ai-engineering-projects/actions/workflows/ci.yml/badge.svg)](https://github.com/bdjjo/ai-engineering-projects/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Implementations and experiments in machine learning, with runnable code, tests, and
measured results. Each project has its own setup instructions and dependencies.

| Project | What it covers | Results |
|---|---|---|
| [01 · GPT-2 from scratch](projects/01-gpt2-from-scratch) | A small GPT in PyTorch, character and byte BPE tokenizers, training, checkpoint recovery, and generation | [Two CPU experiments and held-out evaluation](projects/01-gpt2-from-scratch/docs/RESULTS.md) |
| [02 · SFT and DPO](projects/02-dpo-training-pipeline) | LoRA fine-tuning, a custom DPO loss, and blinded preference evaluation | Two DPO learning rates compared on a T4; 64 response pairs generated, evaluation incomplete |

## Get started

```bash
git clone https://github.com/bdjjo/ai-engineering-projects.git
cd ai-engineering-projects/projects/01-gpt2-from-scratch
```

Follow the project's [README](projects/01-gpt2-from-scratch/README.md) to install
dependencies, train a model, and generate text. Datasets and checkpoints are created
locally when you run the project.

## Layout

Projects live in numbered folders under `projects/` and share the `main` branch.
New projects will be added alongside the existing ones, each with its own README,
dependencies, tests, and results.

```text
projects/
  01-gpt2-from-scratch/
  02-dpo-training-pipeline/
.github/workflows/          Automated checks
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development checks and the convention for
adding a project. Code is available under the [MIT license](LICENSE); dataset
provenance is documented within each project.
