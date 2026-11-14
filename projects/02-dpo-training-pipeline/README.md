# SFT and direct preference optimization

A two-stage LoRA training pipeline for `meta-llama/Llama-3.2-1B-Instruct`.
Supervised fine-tuning learns from OpenHermes demonstrations; a custom PyTorch DPO
objective then learns from Anthropic HH-RLHF preference pairs.

**Status:** CPU checks pass, and the owner completed a small 1B Colab GPU pilot.
The pilot did not demonstrate better generated responses from DPO. A larger experiment
and independent human evaluation remain pending.

## Colab pilot

The owner reported 10 updates per stage on a Tesla T4. SFT validation loss fell from
0.9134 to 0.8672. DPO reached its best validation loss at step 1 (0.6713 versus the
initial 0.6931), then worsened to 0.7236 at step 10. The comparison code selects the
best adapter, rather than the final adapter.

Eight blinded comparisons yielded **0 DPO wins, 2 SFT wins, and 6 ties**: a tie-adjusted
DPO win rate of **37.5%**, with a prompt-bootstrap interval of **18.75–50.0%**.
Ratings were suggested by an AI assistant and saved by the owner; this is not an
independent human evaluation. Several responses were identical or truncated, and some
refusals were inappropriate to the prompt. These results do not establish a general
advantage for either model or a broad safety improvement.

[Pilot results](reports/colab-pilot.json) record the supplied measurements and their
limitations. Original GPU artifacts have not yet been independently inspected.

## Setup

Use Python 3.11 or 3.12, a CUDA-capable GPU, and access to the
[Llama model](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).
Accept the model terms with your own Hugging Face account, then authenticate locally.
Install a compatible CUDA PyTorch build before the project dependencies; existing
compatible PyTorch installations are preserved by these requirements.

```bash
cd projects/02-dpo-training-pipeline
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
huggingface-cli login
python -m pytest -q
python scripts/smoke.py
```

The offline smoke test creates a tiny randomly initialized Llama with LoRA, trains
SFT, reloads its adapter into the same base weights, and runs DPO. It needs no account,
network access, or GPU. [Raw CPU results](reports/cpu-smoke.json) are synthetic checks.

## Train

```bash
python run.py prepare
python run.py sft --steps 500
python run.py dpo --steps 500 --beta 0.1
python run.py compare
```

Preparation requests up to 10,000 SFT examples, 5,000 DPO pairs, 128 validation
examples per stage, and 128 test comparisons. Exact accepted counts, exclusions,
dataset revisions, model revision, and file hashes are saved in the data manifest.
The default run directory is `runs/llama`; use `--data` and `--out` to select paths.
Start with smaller subsets and 10 updates on a new GPU before a longer run.

Training uses one sequence per forward pass with four accumulated microsteps per
update, uniform example sampling with replacement, constant learning rates (2e-4 SFT,
5e-5 DPO), gradient clipping, and seed 1337. These are step budgets, not epochs.
LoRA uses rank 16 and alpha 32 on the attention projections. BF16 is selected when
supported, otherwise FP16 with gradient scaling. GPU memory and runtime have not
yet been measured; reduce sequence limits during data preparation if memory is tight.

Each stage saves `best/`, `last/`, metrics, and provenance. DPO starts from `sft/best`.
Stages refuse to overwrite nonempty output directories. Adapter saves do not include
optimizer or RNG state: exact interrupted-run resume is not implemented. Preserve
outputs on persistent storage and start a fresh run when repeating a stage.

## Data and objective

- Full conversation history must match between chosen and rejected responses.
- The tokenizer's own chat template defines training and generation formatting.
- Prompt and padding labels are masked; genuine response end tokens remain scored.
- Oversized examples are excluded, with counts, rather than silently truncating
  conversation history or the preferred response. This biases the sample toward short text.
- Training and validation are assigned by prompt hash within source training data.
  Exact duplicate prompts are excluded across selected partitions and datasets.
  Official HH test data is used only for final generated comparisons. Near-duplicate
  text and related conversations are not eliminated.
- DPO sums causal response-token log probabilities. Its reference is the frozen SFT
  checkpoint, cached before policy updates. This avoids loading a second 1B model.

`loss = -logsigmoid(beta * ((policy_chosen - reference_chosen) -
(policy_rejected - reference_rejected)))`

At initialization, matching policy and reference give loss `log(2)` and zero margin.
Strict reward accuracy is zero on those ties. It is a diagnostic distinct from a
human-rated win rate. DPO does not guarantee better responses or lower harmfulness.

## Measure the win rate

`compare` generates deterministic continuations for held-out prompts from both selected
adapters. It creates randomized A/B pairs and a separate mapping file. Before opening
the mapping, rate every pair in `blind-comparison.json` with `A`, `B`, or `tie`, and
record a reason. Judge correctness, relevance, completeness, and appropriate handling
of unsafe requests; reward neither verbosity nor blanket refusals alone. Prefer an
independent rater and record the rater count and rubric with the eventual results.

```bash
python scripts/score_judgments.py --ratings runs/llama/blind-comparison.json --key runs/llama/comparison-key.json --out runs/llama/win-rate.json
```

The script requires complete ratings and reports wins, losses, ties, a tie-adjusted
win rate, and a prompt-bootstrap interval. That interval does not measure rater bias
or disagreement. The test is one seed, one corpus, and short continuations; it does
not establish broad alignment or deployment safety.

## References and licensing

[DPO paper](https://arxiv.org/abs/2305.18290) ·
[OpenHermes](https://huggingface.co/datasets/teknium/OpenHermes-2.5) ·
[HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) ·
[PEFT](https://huggingface.co/docs/peft/index)

Source code uses the collection's MIT license. Model and dataset terms are separate;
no model weights or dataset records are included in the repository.
