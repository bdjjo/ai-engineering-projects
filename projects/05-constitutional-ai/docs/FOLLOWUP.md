# Feedback follow-up

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
runtime and memory for these new settings are not measured yet.

Export with `scripts/export_report.py` as in COLAB.md. Inspect every accepted pair,
especially wording-only preferences, unsupported claims, unnecessary refusals, and
revisions that answer the evaluator. Keep ties, truncation, and order-disagreement
checks enabled. Do not begin reward/SFT/RL training merely because each split has
one accepted example; assess label quality and sample size first.

The completed follow-up produced 9 ties, 2 identical responses, 1 order disagreement,
and no accepted preferences. No generation was truncated. See RESULTS.md.

## Next: calibrate the judge

Before collecting more data, run the eight authored diagnostic cases:

```bash
python scripts/calibrate_judge.py --device cuda --out outputs/judge-calibration-01
```

Use a new Drive-backed output path on Colab. Share `calibration.json` for review.
The cases include clear factual differences, answer-versus-commentary, uncertainty,
privacy consent, and genuine ties. Each case is judged in both candidate orders.
Expected labels are authored diagnostic labels, not independent human annotations.
They are separate from training data. Even passing all eight cases would only
establish basic protocol competence, not reliable safety evaluation. No calibration
model run has been completed yet; training remains disabled.

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
they are fresh examples, not independent or blinded evaluation. No measured result
is claimed for the redesign. Inspect explanations and both raw decisions before
collecting further training labels.
