# Archive: results before the 2026-09-22 fixes

Everything produced before the pipeline fixes. **Don't use these numbers in the paper.**
They're kept only as provenance and for rough comparison.

| File | What it is |
|---|---|
| `raw_results.csv` | The original results log, byte-for-byte (including the uncommitted rows). |
| `raw_results_annotated.csv` | Same rows, plus two added columns: `variant_inferred` (the dual-attention ablation arm, recovered from the parameter count because the old logger didn't record it) and `rerun_index` (>0 means the same model/dataset/split/fold/seed was run again and appended). |
| `terminal_res/res1.txt` | Terminal output from the first BiGRU runs (M1 CPU). |
| `checkpoints/` | The old `.pt` checkpoints (git-ignored). |

## Why these results were superseded

1. The dual-attention ablation variant (`full`/`plain`/`channel_only`/`temporal_only`) wasn't logged.
2. 22 runs were repeated and appended as duplicates; WISDM LOSO `temporal_only` was incomplete (4/18).
3. Latency was measured on whichever machine did the training (M1 CPU vs. GPU), so the numbers can't be compared.
4. Early stopping used a random *window-level* validation split. With 50%-overlapping windows that leaks near-duplicate windows, so early stopping rarely triggered.
5. PAMAP2 used only the hand IMU (6 channels) at 100 Hz and included subject 109 (which has almost no protocol data). Accuracy was 55-62%.
6. WISDM was sorted by (user, activity), which joined separate recording sessions into one stream.
7. Labels were encoded per fold instead of with one dataset-wide mapping.
