# Model card

## Model and intended use

Small, randomly initialized GPT-style decoder-only language models for learning
transformers, testing a training pipeline, and demonstrating ML engineering practices.
The code was developed with AI assistance and is accompanied by explanatory material
and behavioral tests. This is an educational portfolio artifact.

The models are not instruction-tuned assistants and are not intended for factual
question answering, translation, consequential decisions, or general deployment.

## Architecture

| Property | Character CPU | Byte BPE CPU |
|---|---:|---:|
| Layers | 4 | 4 |
| Heads | 4 | 4 |
| Embedding width | 128 | 128 |
| Context | 128 tokens | 128 tokens |
| Vocabulary | 65 | 512 |
| Parameters | 818,048 | 875,264 |
| Dropout | 0.1 | 0.1 |
| Position encoding | Learned absolute | Learned absolute |
| Attention | Explicit scaled dot-product | Explicit scaled dot-product |
| Weight tying | Input/output embeddings | Input/output embeddings |

The models use pre-normalization, GELU feed-forward layers with a 4× width expansion,
residual connections, and a final layer norm. They are inspired by GPT-2's architecture
but do not replicate its tokenizer, model size, pretrained weights, or training corpus.
There is no KV cache or distributed training.

## Data

Tiny Shakespeare from the char-rnn repository: 1,115,394 ASCII bytes/characters.
The source is checked against SHA-256
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
Contiguous 90/5/5 splits are made before fitting the tokenizer. The learned BPE merges
and character vocabulary are fitted only on training text. See [data provenance](../data/README.md).

This corpus is a narrow literary domain. Repeated phrases may occur across partitions.
The split is not de-duplicated or grouped by play, so metrics should not be treated as
a broad generalization benchmark.

## Training and evaluation

The reference configurations use 2,500 AdamW updates, batch size 16, gradient clipping
at 1.0, 100 warmup updates, cosine learning-rate decay, and seed 1337. Each run processes
5,120,000 target tokens, sampled with replacement. This is a different amount of source
text for each tokenizer. Computation uses CPU float32 with four PyTorch threads.

Validation windows select the checkpoint; final evaluation scores all validation and
test targets once with context reset at each 128-token block. Uniform, add-one unigram,
and add-one bigram baselines use the same held-out targets. Perplexity, cross-entropy,
and byte-normalized NLL are reported in [RESULTS.md](RESULTS.md).

The training measurements come from the supplied local CPU runs. Automated checks
also run on Ubuntu/Python 3.11 and 3.12 and Windows/Python 3.12 through
[GitHub Actions](https://github.com/bdjjo/gpt2-from-scratch/actions/workflows/ci.yml).
The GPU preset has not been trained or benchmarked.

## Limitations

- The models can produce recognizable dialogue formatting while remaining incoherent,
  repetitive, or grammatically incorrect.
- One seed and one small corpus do not establish statistically robust performance.
- Training and testing use a literary domain with historical language and biases.
- Character prompts are limited to the training alphabet. BPE accepts arbitrary UTF-8,
  but acceptance does not imply the model understands unseen languages or symbols.
- BPE sampling can produce invalid UTF-8 byte sequences, decoded with replacement.
- The experiments do not hold parameter count, byte context, or total text exposure
  constant across tokenizers. They are not a controlled causal comparison.
- Exact reproduction across different versions, devices, and numerical backends is
  not guaranteed. The checkpoint-resume equality test covers one controlled CPU setting.
- Weights are local run artifacts, excluded from the source repository. Anyone cloning
  the source must train to regenerate them unless they are separately released.

## References

Architecture context: [GPT-2 technical report](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
Data context: [char-rnn](https://github.com/karpathy/char-rnn).
The implementation uses no pretrained checkpoints, Hugging Face model classes, or nanoGPT code.
