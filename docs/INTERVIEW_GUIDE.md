# Explain the work with evidence

This project was developed with AI assistance. Understand, verify, and extend the
implementation before presenting it as evidence of your own expertise. Describe
your contribution accurately; understanding the design and defending the measurements
is more convincing than claiming every line was written unaided.

## A 60-second introduction

> I built and tested an educational GPT-style language-model pipeline in PyTorch.
> It includes an explicit implementation of causal multi-head attention and two
> tokenizers: characters and byte-pair encoding. I trained small models from random
> initialization on Tiny Shakespeare using a train/validation/test split. I added
> deterministic sampling, resumable checkpoints, and tests that verify causality,
> gradients, and exact CPU resume. I compare held-out results with unigram and
> bigram baselines and document where the generated text still fails.

Adapt this introduction to your actual involvement and use the measured numbers
from [RESULTS.md](RESULTS.md). Do not describe the project as training the original
GPT-2, achieving state of the art, or reproducing results across hardware you have
not tested.

## A five-minute demonstration

1. **Problem and scope, 30 seconds:** explain next-token prediction and the small corpus.
2. **Architecture, 60 seconds:** open `model.py`; trace embeddings, attention, MLP, and logits.
3. **Correctness, 60 seconds:** run the causal-mask and checkpoint-resume tests.
4. **Results, 60 seconds:** show the loss curve, held-out baselines, and evaluation protocol.
5. **Generation, 45 seconds:** generate from a saved checkpoint using a fixed prompt and seed.
6. **Limitations and next experiment, 45 seconds:** explain one failure and one testable improvement.

Useful commands, after activating the environment:

```powershell
python -m pytest -q tests/test_model.py -k future
python -m pytest -q tests/test_training.py -k resume
python generate.py --checkpoint runs/char-cpu/best.pt --prompt "ROMEO:" --tokens 200 --seed 42
```

The trained weights exist locally after training and are excluded from the source
repository. On a fresh clone, train first. The committed logs and samples let a
reviewer inspect the experiment without downloading weights.

## Questions to practice

| Question | What a strong answer should cover |
|---|---|
| What makes the model autoregressive? | Each distribution depends only on previous/current input tokens; generation appends one token at a time. |
| Why is the diagonal allowed in the attention mask? | Position t uses token t to predict token t+1. |
| What happens without the causal mask? | The model can read later tokens containing the answer; training no longer matches generation. |
| Why divide attention scores by sqrt(head dimension)? | It moderates score magnitude before softmax as the dot-product dimension increases. |
| Why multiple heads? | Separate learned projections can capture different relationships; roles are not explicitly assigned. |
| Why pre-normalization and residual connections? | Normalization controls feature scale; residual paths carry information and gradients through stacked blocks. |
| Why fit BPE only on training text? | Validation and test statistics must not determine learned token merges. |
| How is this BPE different from GPT-2's? | Smaller learned vocabulary, a simpler regex, and different token IDs; it is not pretrained-tokenizer compatible. |
| Why compare against a bigram? | It checks whether the transformer improves beyond local transition statistics. |
| Why not compare character and BPE perplexities directly? | Their units are different; report byte-normalized NLL and discuss context/exposure differences. |
| What does an increasing train/validation gap suggest? | Possible overfitting or distribution mismatch; the gap alone does not identify the cause. |
| Why save optimizer and RNG state? | AdamW moments, stochastic masks, and sampled batches affect subsequent updates. |
| What does the resume test prove? | Exact equality in a tiny controlled CPU experiment in one software environment, not universal determinism. |
| Why is GPU performance not claimed? | The GPU preset is supplied but the measured experiments ran on CPU. |
| How would you speed up inference? | KV caching could reuse prior keys/values, but position handling and context cropping need careful tests. |

## Honest portfolio wording

Use these as templates only after you can explain the work:

- Developed and validated a GPT-style language-model pipeline in PyTorch, including
  byte BPE, causal attention, AdamW training, checkpoint recovery, and seeded generation.
- Measured held-out language-model performance against frequency baselines and
  published configurations, raw metrics, reproducibility checks, and limitations.

Add measured parameter counts and evaluation values from the report. Avoid turning
perplexity reductions into “accuracy” improvements. Do not list an unrun GPU setting
as training experience.

## Your next independent contribution

Choose one experiment, state a hypothesis, and change only what you need to test it:

1. **Attention backend:** run manual and SDPA on identical settings; compare update
   throughput and held-out validation loss. Keep the test partition reserved.
2. **Context length:** compare 64 and 128 tokens while recording batch token count
   and the computational difference.
3. **Dropout:** compare 0.0 and 0.1 under a fixed training budget; inspect the train/
   validation gap.
4. **Tokenizer vocabulary:** compare 256, 512, and 768 BPE tokens; report bytes per
   token, embedding parameters, and byte-normalized validation NLL.
5. **Variance:** repeat one configuration using at least three seeds; report the
   individual values and mean/spread instead of selecting only the best run.

Document the question, config, command, evidence, interpretation, and what you would
do next. That reasoning will make the portfolio more distinctly yours.
