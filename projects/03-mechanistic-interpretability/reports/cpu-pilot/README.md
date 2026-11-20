# Reference experiment artifacts

These are measurements from actual CPU runs, not the example output in the project brief.
See [the results report](../../docs/RESULTS.md) for scope, settings, and interpretation.

- `ioi_results.json`: clean/corrupt scores and all per-prompt head effects.
- `sae_comparisons/`: all four runs, preserving complete logged validation histories.
- `sae_summary.json` and `sae_history.json`: selected 3,000-update run.
- `sae_comparison.csv`: compact comparison table.
- `feature_index.json`: browser examples from the original project probe corpus.
- PNG files: plots of recorded measurements.

The index contains 781 probe tokens and no downloaded dataset text.
Checkpoints and activation caches are excluded from Git. Recreate them with the
commands in the project README. The model/dataset licenses remain separate from
this repository's MIT source license.
