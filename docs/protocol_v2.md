# Evaluation protocol v2

**Date:** 2026-10-16. **Written before any v2 result exists.** The decision rule in section 3
is fixed now so that the paper's framing can't be chosen after seeing the test numbers.

## 1. Why v1 was replaced

The v1 run (2026-10-15, 483 runs, archived in `results/archive/2026-10-15_protocol_v1/`) was
complete and its numbers are internally consistent, but two weaknesses would not survive review:

1. **Model selection stopped almost immediately.** Early stopping and the LR scheduler used
   validation *loss* on 1–3 held-out subjects. Under subject shift, validation loss rises as soon
   as the model becomes confident, while accuracy/F1 keep improving. 43% of runs picked a best
   epoch ≤ 3; median best epoch was 3 on PAMAP2 and WISDM. Example (WISDM LOSO fold 1, BiGRU):
   val loss 1.76 → 2.38 → … → 2.68 while train loss fell 0.60 → 0.03, so the epoch-1 model was kept.
   Every model was effectively compared after 1–3 epochs of training.
2. **The per-dataset tests could never be significant.** With 6 LOSO folds the smallest possible
   two-sided Wilcoxon p is 0.031, i.e. 0.19 after Holm over 6 comparisons. UCI-HAR CNN+BiGRU beat
   the proposed model on 6/6 folds and still got p = 0.19.

These were diagnosed from training curves and the test design, not from which model won. The
changes below apply identically to every model.

## 2. What changed

| | v1 | v2 |
|---|---|---|
| Best-epoch / early-stopping / LR-plateau metric | val loss | **val macro-F1** (`select_metric`) |
| Validation subjects per fold | 10%, ≥ 1 (PAMAP2: 1 subject) | **20%, ≥ 2** |
| LOSO folds (UCI-HAR / WISDM / PAMAP2) | 6 / 6 / 8 | **10 / 12 / 8** (3 / 3 / 1 subjects held out) |
| Seeds per fold | 3 | **5** |
| Baselines | BiGRU, BiLSTM+Attn, CNN+BiGRU | **+ DeepConvLSTM** (Ordóñez & Roggen 2016, published config) |
| Runs (jobs) | 483 (161) | **1320 (264)** |

Unchanged: architectures, hidden sizes, dropout, lr 1e-3, batch 64, ≤ 50 epochs, patience 8,
grad clip 1.0, preprocessing, the official splits, and the fold-level statistics in
`src/stats_test.py` (seeds averaged per fold, Wilcoxon + paired t, Holm).

Smallest attainable two-sided Wilcoxon p, after Holm over the 7 comparisons: UCI-HAR (10 folds)
0.014, WISDM (12) 0.003, pooled (30 folds) ≪ 0.001. PAMAP2 has only 8 subjects, so its floor is
0.055: on its own it can't reach significance and is reported descriptively (it still counts in
the pooled test).

Each result row carries `protocol = v2`, which is part of the run key: v1 rows are never resumed
from and `load_results` ignores them, so stats and tables only ever use v2 runs.

## 3. Decision rule for the paper (fixed in advance)

**Primary endpoint:** LOSO macro-F1, one value per fold (mean over seeds), pooled over the three
datasets (30 folds); two-sided Wilcoxon signed-rank, Holm-corrected over the 7 comparisons of
`dual_attention_bigru:full` vs every other arm. Per-dataset tests and the official splits are
secondary.

- **Method paper** ("spectral-gated dual attention improves BiGRU HAR"), only if *both* hold:
  1. `full` beats `plain` on the primary endpoint with Holm p < 0.05 (that delta is the claimed
     novelty), and
  2. `full` is not significantly worse (Holm p < 0.05) than any baseline, pooled or on any dataset.
- **Otherwise: empirical / benchmark paper.** Working title: *"Does attention help recurrent HAR
  models under subject-independent evaluation? A controlled ablation on three benchmarks."*
  Contributions: matched-capacity ablation of channel / temporal / spectral attention, compared
  against standard baselines including DeepConvLSTM; a rigorous protocol (LOSO, subject-wise
  validation, 5 seeds, fold-level paired tests); efficiency (params, FLOPs, latency); and the
  finding itself, whatever its direction.

After the v2 test results are seen, no architecture or hyperparameter is changed for the
reported comparison. Any later change must be chosen on validation data only and reported as
such.
