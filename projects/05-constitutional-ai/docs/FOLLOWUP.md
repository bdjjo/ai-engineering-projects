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
