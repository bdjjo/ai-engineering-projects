# Design and measurement

## Where this differs from the research paper

[Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
describes a supervised critique/revision phase followed by reinforcement learning
from AI preference feedback. This project borrows that structure at teaching scale.
It uses a 15-principle constitution adapted from the project brief, not the paper's
exact constitution or a claim to reproduce Anthropic's results.

The preferences compare an original response with its critique-driven revision.
Accepted preferred answers become SFT targets. A frozen pretrained transformer
with a trained scalar head models the preferences. RL uses capped sequence-level
REINFORCE with an exponential moving baseline, rather than PPO or a learned value
function. The reward model supplies the actual RL reward; it is not a language
model checkpoint passed to a generation function.

This separation matters. Training a reward model alone does not update the assistant.
The SFT and RL stages each save a generative policy adapter, and evaluation loads
those adapters explicitly.

## Data and feedback

The 72 original prompts are split into 36 train, 18 validation, and 18 test prompts.
Every partition covers all 15 principles. Exact normalized duplicates are rejected.
Imported datasets can provide a `family` field to keep paraphrases together; the
authored scenarios are manually assigned distinct scenarios. There is no claim
that this tiny set rules out semantic overlap or benchmark contamination in the
pretrained models.

Critiques use the relevant principle together with helpfulness, autonomy, and
uncertainty rules. Quoted JSON separates candidate content from evaluator instructions,
but this is not a guarantee against prompt injection into an LLM judge.

The preference judge sees A/B candidates without their original/revised labels,
then sees the opposite order. A strict matching preference is required. Consistent
ties, order disagreements, malformed labels, empty completions, identical answers,
and generation length limits are counted separately. The judge can prefer the
original. No arbitrary parsing fallback assigns a neutral or preferred label.

## Reward objective

For a chosen/rejected pair, the objective is

`L_reward = -mean(logsigmoid(r_chosen - r_rejected))`.

The reward head uses the last non-padding hidden state, including the assistant
turn terminator where the template adds one. Pooling finds the last actual token
for either padding side. The frozen backbone receives explicit position IDs derived
from the attention mask. Features can be cached because the backbone is not trained.

The scalar head is optimized in FP32. The checkpoint with lowest validation
pairwise loss is retained, including step zero if training fails to improve it.
Validation strict preference accuracy, ties, margins, and a simple “longer answer
wins” baseline help reveal shortcuts. The normalization mean and population standard
deviation are fitted on training rewards only. RL rewards use this fixed scale and
are clipped to [-5, 5]. Reward magnitude is not a calibrated safety probability.

## Policy objectives

SFT scores only assistant response tokens, with a one-token causal shift. Prompt
tokens and batch padding have label -100. It minimizes token-weighted negative
log likelihood and selects the lowest validation NLL checkpoint.

RL begins with the selected SFT adapter. A frozen copy of that same policy provides
the reference. For sampled answer `y` to prompt `x`:

```text
sampled_KL = log pi(y|x) - log pi_ref(y|x)
utility = normalized_reward(x,y) - kl_coefficient * sampled_KL
loss = -mean(stop_gradient(utility - previous_baseline) * log pi(y|x))
```

The moving baseline is updated only after the optimizer step. Even batch size one
can therefore have a nonzero gradient. Sampling uses temperature 1, top-k 0, top-p 1;
the score-function gradient uses the same unmodified categorical policy. A fresh
generation configuration clears inherited repetition penalties and other logit
processors during RL sampling. Greedy feedback/evaluation retains the pretrained
checkpoint's generation settings apart from the explicit length and EOS settings.
Prompt and
padding tokens do not contribute. EOS is scored when generated. If the generation
budget is exhausted, the capped answer remains an RL action and is flagged.

The KL term is a sampled sequence log-ratio. An individual observation can be
negative even though the expected KL is nonnegative. This estimator can be noisy;
the project does not claim the stability of a production PPO implementation.
A unit test compares its expected gradient with an exactly enumerated two-action
regularized objective. Gradients through the reward, reference, and advantage are
disabled; policy gradients are clipped to norm 1.

RL checkpoint selection uses greedy validation learned reward. Selection may keep
step zero and is vulnerable to reward hacking. Test evaluation is therefore a
separate stage, and learned-reward improvement is never labeled alignment improvement.

## Evaluation and uncertainty

The evaluator compares base versus SFT and base versus RL using the same reserved
prompts and greedy response generation. A separate judge configuration returns
integer safety/helpfulness ratings from 1 to 5 plus a refusal flag. Pairwise judging
uses both answer orders. The default larger preset uses a bigger model from the same
family; this does not eliminate shared biases. The CPU feedback pilot uses one
checkpoint in all roles and is particularly weak as an evaluator.

Means and paired deltas use only prompts with valid ratings and complete responses
on both sides. Coverage and truncation counts are reported alongside them. Preference
win rate gives a tie half a win, includes consistent ties in the denominator, and
excludes invalid/disagreeing judgments. If nothing is valid, the metric is null.
Benign refusal is reported separately; a refusal can sometimes be justified even
for a prompt assigned benign by the dataset author.

The report provides percentile bootstrap intervals over prompts, using 2,000 draws
with a fixed seed. No interval is shown for fewer than two observations. These
intervals describe prompt variation under a fixed judge; they do not capture judge
error, training-seed variation, or population representativeness.

## Repeatability and operational limits

Runs record exact model revisions, package versions, prompt hashes, constitution,
configuration, device, and source hashes. Each completed feedback record is written
atomically. Collection can reuse completed rows only with the same input identity.
Training directories are created exclusively to avoid mixing incompatible runs.
Completed stage summaries mark reusable stages; partial training directories require
a fresh run. Optimizer/RNG recovery within a training stage is not implemented.

An explicit `--feedback-from` import can carry completed feedback into a new run
after an implementation change. It verifies prompt/config identity and reconstructs
the preference pairs from the raw records before importing. The new manifest embeds
the upstream manifest plus record/pair/audit hashes. It does not rewrite the old
provenance. Subsequent training stages also verify the imported preference hash.

Model weights and generated working files stay under ignored output directories.
The report exporter copies an explicit allowlist of JSON artifacts. Its exports
include response text for inspection and should be reviewed before publication.
No private learning notebooks or credentials belong in the public repository.
