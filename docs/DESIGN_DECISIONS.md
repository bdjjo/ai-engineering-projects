# Engineering decisions beyond the starter brief

The supplied brief is a useful learning outline. This implementation makes several
choices explicit so the final project can be inspected and reproduced.

| Issue or decision | Implementation | Why it matters |
|---|---|---|
| The brief promises BPE but supplies a character tokenizer | Provide both, with a real byte BPE algorithm and deterministic merge order | The implementation matches the stated feature; characters remain a useful baseline |
| Original GPT-2 vocabulary size does not match a new tokenizer | Resolve model vocabulary from the prepared tokenizer | Avoid allocating unused outputs or indexing the wrong vocabulary |
| Vocabulary fitted on all text | Fit vocabulary/merges only on the training partition | Prevent validation/test statistics from determining learned preprocessing |
| Only a validation split | Reserve a separate 5% final test partition | Separate checkpoint selection from final measurement |
| Ambiguous “from scratch” attention | Default to explicit attention math; retain optional SDPA with output/gradient equivalence tests | Make both the implementation and optimization boundary clear |
| Runtime depends strongly on hardware | Supply a small smoke run, CPU presets, and a measured benchmark command | Verify correctness and estimate cost before committing to a long run |
| Evaluation can consume a shared training sampler | Separate RNG streams and recreate fixed evaluation windows | Logging frequency does not change training batches |
| Warmup begins with a zero-learning-rate update | Use one-based optimizer update numbering | The first update actually changes parameters |
| Only intermediate checkpoints are saved | Save best and last checkpoints, including the final update | Generation can use the selected model and training can resume |
| Config objects pickled into checkpoints | Store tensor state dictionaries and primitive configuration dictionaries | Load the checkpoint using `weights_only=True` |
| Resuming only model weights | Restore optimizer state, update count, sampling RNG, and dropout RNG | Preserve the training trajectory in the tested CPU environment |
| Final evaluation uses random samples | Score every held-out next-token target once, including the remainder | Define a complete and auditable denominator |
| Perplexity alone can mislead | Include unigram/bigram baselines, raw loss, perplexity, and bits per byte | Establish a meaningful baseline and make token units visible |
| Attractive samples can hide failures | Publish every sample from a fixed prompt/seed plan | Show qualitative strengths and weaknesses without selection after generation |
| GPU configuration can look like a measured result | Mark GPU and hosted CI execution as unverified until run | Separate implemented capability from experimental evidence |

## Deliberate scope limits

The project uses float32, a small vocabulary, straightforward sampling, local JSON logs,
and a single-process training loop. There is no external tracking account, web demo,
KV cache, mixed precision, or distributed training. Those would add engineering work
without being necessary to explain the current experiment.

The byte BPE implementation is intentionally readable. It is suitable for this corpus;
it is not a tokenizer optimized for training on billions of bytes. The original GPT-2
regex and pretrained token IDs are not reproduced.

The character and BPE experiments have equal update counts, batch sizes, and token
contexts. This does **not** make their effective text exposure, byte context, or parameter
count equal. A future tokenizer study should decide which budget to hold constant and
run multiple seeds before making a causal claim about tokenizer quality.

## Evidence to inspect

- `tests/test_model.py`: mask behavior, full logits, gradient agreement, tiny-batch overfit.
- `tests/test_training.py`: update schedule, optimizer grouping, RNG isolation, and exact resume.
- `tests/test_data.py`: train-only fitting and artifact tampering checks.
- `tests/test_evaluation.py`: partial-final-block coverage and batching invariance.
- `reports/`: raw configurations, measurements, and generated samples.
