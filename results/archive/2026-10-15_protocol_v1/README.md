# Archive: protocol v1 run (2026-10-15)

The complete 483-run grid trained on the RTX 3050 laptop (commit `4e57de2-dirty`, i.e. with the
`StackedRNN` change from `scripts/Decision.md`). The numbers are internally consistent, but the
protocol was replaced by v2, so **don't use them as the paper's results**. See
`docs/protocol_v2.md` for the reasons: early stopping on val loss kept 1–3-epoch models, and
6-fold per-dataset tests could not reach significance.

| Path | What it is |
|---|---|
| `tables/raw_results.csv` | All 483 runs (no `protocol` column, so `load_results` ignores it). |
| `tables/parts/` | Per-job result files from `run_parallel.py`. |
| `tables/master_comparison.{md,csv}`, `significance.csv`, `efficiency.csv`, `confusion/` | Tables produced from those runs. |
| `figures/`, `preds/` | Confusion / box / efficiency plots and per-run predictions with loss curves. |

Headline v1 results (LOSO macro-F1): CNN+BiGRU best on all three datasets; `full` vs `plain`
pooled ΔF1 = +0.003 (p = 0.73); nothing significant after Holm.
