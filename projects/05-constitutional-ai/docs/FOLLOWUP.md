# Feedback and calibration experiment history

This series is paused. The sections below preserve reproduction instructions and
completed outcomes. They are not instructions to resume training. See RESULTS.md
for the final evidence and limitations.

Keep the completed 54-prompt run. Use a fresh checkout of this release and a new
Drive folder: `p5-feedback-followup-01`. Do not reuse the original run folder with
changed code or settings. Follow the environment setup in COLAB.md, then run:

```python
subprocess.run([
    python, 'run_cai.py', 'collect',
    '--config', 'configs/colab-feedback-followup.json',
    '--device', 'cuda', '--out', str(work),
], cwd=project, check=True)
```

Here `work` must point to the new Drive folder. The preset uses 12 non-test prompts,
the existing public model downloads, a stronger critic, and a longer token budget.
It changes several factors together and is intended as a diagnostic pilot. GPU
collection completed on T4; peak memory and elapsed runtime were not recorded.

Export with `scripts/export_report.py` as in COLAB.md. Inspect every accepted pair,
especially wording-only preferences, unsupported claims, unnecessary refusals, and
revisions that answer the evaluator. Keep ties, truncation, and order-disagreement
checks enabled. Do not begin reward/SFT/RL training merely because each split has
one accepted example; assess label quality and sample size first.

The completed follow-up produced 9 ties, 2 identical responses, 1 order disagreement,
and no accepted preferences. No generation was truncated. See RESULTS.md.

## Legacy judge calibration

Before collecting more data, run the eight authored diagnostic cases:

```bash
python scripts/calibrate_judge.py --device cuda --out outputs/judge-calibration-01
```

Use a new Drive-backed output path on Colab. Share `calibration.json` for review.
The cases include clear factual differences, answer-versus-commentary, uncertainty,
privacy consent, and genuine ties. Each case is judged in both candidate orders.
Expected labels are authored diagnostic labels, not independent human annotations.
They are separate from training data. Even passing all eight cases would only
establish basic protocol competence, not reliable safety evaluation. The completed
legacy run scored 3/8; training remains disabled.

## Structured judge redesign

The first calibration has now run: 3/8 correct, consisting only of the three
expected ties. All five clear preferences failed (three ties and two order
disagreements). This matches an always-TIE baseline on these authored cases.
The preserved report is `reports/judge-calibration-legacy/calibration.json`.

The experimental replacement prioritizes factual correctness and relevance, asks
for a brief comparison, and parses strict JSON with `reason` and `decision` fields.
Both candidate orders must still agree. Invalid JSON and cut-off responses remain
invalid, never neutral labels. It has not replaced the pipeline's feedback judge.

```bash
python scripts/calibrate_judge.py --mode structured --device cuda --out outputs/judge-structured-01
```

This runs the original eight cases and four additional diagnostic cases, with a
256-token budget for each judgment. It changes prompt, output schema, and budget
together. The additional cases were written after examining the earlier failures;
they are fresh examples, not independent or blinded evaluation. The completed
structured run scored 0/12 originally and 2/12 after parser repair. Inspect explanations and both raw decisions before
collecting further training labels.

## Formatting repair and stronger-model comparison

The first structured run returned 0/12 accepted correct decisions: all 24 outputs
were JSON wrapped in Markdown fences and were rejected by the original parser.
The repaired parser accepts one complete fenced JSON block, while rejecting prose,
multiple blocks, missing fields, invalid decisions, and empty explanations.
Parser-only rescoring of the saved outputs yields **2/12 correct**. No model was
rerun to obtain that figure. The original and rescored reports remain separately
available under `reports/judge-calibration-structured/`.

Formatting repair does not resolve the incorrect judgments. Compare a larger
checkpoint with the same structured prompt, cases, and generation budget:

```bash
python scripts/calibrate_judge.py --mode structured --device cuda --judge-model Qwen/Qwen2.5-3B-Instruct --out outputs/judge-3b-01
```

The default FP32 command above was followed by an unsuccessful killed process.
A separate FP16 loader override completed the diagnostic and scored 3/12.
The override and exact model revision are recorded in the exported report.
The default command above does not reproduce that precision override.
Peak memory and elapsed runtime were not measured. See RESULTS.md for details.
The 3B checkpoint has its own Qwen research license; consult its model card.
Keep training disabled and review raw outputs before proceeding. The same-family
comparison and development cases do not constitute independent safety validation.
