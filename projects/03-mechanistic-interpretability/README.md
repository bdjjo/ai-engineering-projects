# Mechanistic interpretability: inside GPT-2

An activation-patching study of indirect object identification (IOI), a residual-stream
sparse autoencoder (SAE), and a Gradio feature explorer. The project includes real
GPT-2 experiments, offline behavioral tests, and explicit limits on what the results show.

Given “When Mary and John went to the store, John gave a drink to”, the IOI task is
to favor **Mary** over **John**. Head interventions measure which internal outputs
affect that preference. The SAE learns a sparse representation of a separate text
corpus; its features are an exploratory decomposition, not established IOI mechanisms.

## Measured results

| Experiment | CPU result |
|---|---|
| IOI head sweep | Correct recipient preferred over the subject on 16/16 prompts; mean logit difference 3.8115 |
| Largest decrease from head patching | Layer 9, head 9: 1.7135 logits |
| Reference SAE | 1,024 features, 3,000 updates; 64.6% held-out variance explained and 206.6 active features per token |
| Feature explorer | 781 probe tokens from 32 original paragraphs; feature meanings unvalidated |

All four SAE runs, including less sparse and lower-fidelity alternatives, appear
in the [results report](docs/RESULTS.md). These are small, single-seed experiments.
The reference SAE does not reach the original brief's suggested sparsity or
reconstruction targets. The full GPU experiment remains unrun.

## Install

Use Python **3.11 or 3.12**. From this directory, create a separate environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

Windows PowerShell: use `py -3.12 -m venv .venv` and
`.\.venv\Scripts\Activate.ps1` instead of the first two commands.
If Windows reports an overly long dataset cache filename, set
`$env:HF_DATASETS_CACHE="$env:TEMP\p3-datasets"` before collecting activations.
For GPU use, install a suitable CUDA PyTorch build before the remaining dependencies.
TransformerLens 2.15.4 and Gradio 5.49.1 are the tested versions; the original brief's
older pins are not required by this implementation. No gated model access is needed.

## Run the CPU pilot

All paths below are relative to this project directory. First runs download GPT-2
Small (roughly 0.5 GB of weights) and stream the requested dataset records.

```bash
python run_ioi.py --out outputs/ioi --batch-size 2
python collect_activations.py --out data/activations.pt
python train_sae.py --activations data/activations.pt --out outputs/sae
python export_features.py --checkpoint outputs/sae/sae_final.pt --out outputs/feature_index.json
python app.py --index outputs/feature_index.json
```

Open `http://127.0.0.1:7860`. To browse the checked-in reference index without model
downloads or training, run `python app.py` after installation.
Outputs refuse to overwrite existing experiments. Choose new paths to repeat a run.
Training checkpoints contain weights and configuration; interrupted optimizer/RNG
recovery is not implemented, so an incomplete training stage must be restarted in a new folder.

## What is measured

**IOI.** Sixteen prompts balance four name pairs, two settings, and both name orders.
The corrupt prompt swaps both names while answer IDs remain fixed to the clean
recipient and subject. Each prompt pair must have equal token length. Batches are
grouped by length, so final-position scores never read padding tokens. Names must
each tokenize to one token including the leading space.

For every one of the 144 heads, `hook_z` is patched at **all positions**:

- Damage = clean score − score after replacing one clean head with its corrupt output.
- Recovery = score after replacing one corrupt head with its clean output − corrupt score.
- Score = logit(clean recipient) − logit(clean subject); positive favors the recipient.

An identical-source patch checks that the intervention mechanism preserves the
unmodified score. Positive values are red in the plots. All raw per-prompt effects
are saved. Because both name orders are included, clean and corrupt examples are
paired symmetrically; mean damage and recovery are consequently the same up to
numerical error. They are not independent evidence. This is one narrow prompt family,
not a sample of general language understanding.

**SAE.** The hook is `blocks.8.hook_resid_post`: after the **ninth** transformer block,
using zero-based layer numbers. The reference CPU preset trains 1,024 features for 3,000 updates
on 8,192 WikiText-2 training tokens, with 2,048 official validation tokens. Exact
normalized-text duplicates are excluded. Source prefixes are capped at 128 tokens;
BOS is excluded and no padding activations are collected. Related passages and
near-duplicates are not eliminated.

`features = ReLU((x / scale − b_dec) W_enc + b_enc)`

`reconstruction = features W_dec + b_dec`

The loss is token-mean **summed** squared reconstruction error plus `1.0 ×` token-mean
feature L1. Its coefficient is not interchangeable with coefficients for an objective
that averages over feature dimensions. Decoder **rows** (one per feature) have unit
norm; decoder gradients are projected onto their tangent planes before updates.
Scale and per-dimension mean come only from training activations. The browser loads
that saved scale and the exact model revision and processing convention.

Held-out metrics include per-dimension MSE, mean active features per token (L0),
inactive fraction on the whole validation set, and variance explained relative to
the training mean. Inactivity on a finite sample does not establish a permanently
dead feature. No target variance-explained or sparsity score is guaranteed.

## Results and limitations

Three 500-update runs compare L1 coefficients 0.01, 1, and 5. A 3,000-update follow-up
uses coefficient 1. All runs are reported; the longer moderate-penalty run supplies
the feature browser. The same validation set informed this choice, so it is not
an untouched final test set.

The [experiment report](docs/RESULTS.md) contains the measured CPU results and links
to raw artifacts. This study does **not** reproduce the paper's complete 26-head
circuit, path-patching analysis, or faithfulness/completeness/minimality evaluation.
A large head effect alone does not identify that head's computational role.

The browser ranks features on 32 original, AI-assisted probe paragraphs written
for this project, separate from the SAE corpus. It displays up to four distinct
paragraphs per feature, selected from a bounded set of highest-activation candidates.
Absence of an example does not establish absence of a feature in general text.
These are exploratory contexts; no semantic labels or causal feature interpretations
have been independently validated. The SAE has not been shown to recover IOI heads
or to preserve GPT-2's language-model loss when substituted into the model.

## Larger GPU experiment

This is a runnable preset, not a completed reference result:

```bash
python collect_activations.py --device cuda --corpus openwebtext --train-tokens 500000 --val-tokens 20000 --out data/owt-layer8.pt
python train_sae.py --device cuda --config configs/gpu.json --activations data/owt-layer8.pt --out outputs/sae-gpu
python export_features.py --device cuda --checkpoint outputs/sae-gpu/sae_final.pt --out outputs/gpu-feature-index.json
```

The preset uses 4,096 features, a batch of 2,048, and 20,000 updates. OpenWebText-10k
records are split by normalized-text hash before collection; 10% of hash buckets
are reserved for validation. This is a different corpus and budget from the CPU
pilot, so do not compare the metrics as if only the device changed. Model and dataset
revisions, text fingerprints, activation/checkpoint hashes, and package versions are
recorded. Use `--revision` and `--dataset-revision` to repeat specific revisions. CUDA has not been
benchmarked here; reduce the batch size if necessary.

## Hugging Face Space

The deployment contains precomputed measurements and a small Gradio app. It needs
neither GPT-2 nor the SAE weights at runtime.

```bash
python scripts/package_space.py --index reports/cpu-pilot/feature_index.json
hf auth login
python scripts/deploy_space.py --repo YOUR_USERNAME/gpt2-feature-explorer
```

Authenticate locally with your own write-capable token. The publisher verifies the
account owner and uploads only the six packaged files. Publishing does not request
paid hardware. The deployment target for this repository is `bdjjo/gpt2-feature-explorer`;
see the results report for its actual deployment status.

## Development and references

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

Tests run offline, including real TransformerLens hooks on a tiny randomly initialized
model, training/reload, saved scaling, data separation, and browser escaping.
CI checks Linux and Windows. Tiny-model tests are software validation, not GPT-2 findings.

- [Wang et al., Interpretability in the Wild (2022)](https://arxiv.org/abs/2211.00593)
- [TransformerLens activation-patching tutorial](https://transformerlensorg.github.io/TransformerLens/generated/demos/Main_Demo.html)
- [Towards Monosemanticity (2023)](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
- [WikiText dataset](https://huggingface.co/datasets/Salesforce/wikitext)
- [OpenWebText-10k dataset](https://huggingface.co/datasets/stas/openwebtext-10k)
- [Hugging Face Gradio Spaces](https://huggingface.co/docs/hub/spaces-sdks-gradio)

Source code and original probe paragraphs use the collection's MIT license. Model
and dataset terms are separate. No weights, downloaded dataset text, or credentials
are committed. Training corpus artifacts stay local.
