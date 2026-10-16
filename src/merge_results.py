"""
src/merge_results.py

Merges the per-job result files written by parallel training
(results/tables/parts/<dataset>/*.csv, see src/run_parallel.py) into the single
results/tables/raw_results.csv that stats_test.py and evaluate.py read.

- Rows already in raw_results.csv are kept (e.g. from an earlier sequential run).
- One row per run key (model/variant/dataset/split/fold/seed); if a run appears
  more than once, the row with the latest timestamp wins.
- Rows are written in a fixed order, the same order a sequential run_all.sh
  produces: dataset -> model -> split (official, loso) -> fold -> variant -> seed.
- The output is written to a temp file and then renamed, so a crash mid-merge
  never leaves a truncated raw_results.csv. The part files are left in place.

    python -m src.merge_results
    python -m src.merge_results --datasets ucihar wisdm pamap2
"""

import argparse
import os
from pathlib import Path

import pandas as pd

from src.utils import RAW_RESULTS_CSV, RESULTS_DIR, RESULT_COLUMNS, RUN_KEY

PARTS_DIR = RESULTS_DIR / "tables" / "parts"

DATASET_ORDER = ["ucihar", "wisdm", "pamap2"]
MODEL_ORDER = ["bigru_baseline", "bilstm_attention_baseline", "cnn_bigru_reference", "deepconvlstm",
               "dual_attention_bigru"]
VARIANT_ORDER = ["-", "full", "plain", "channel_only", "temporal_only"]
SPLIT_ORDER = ["official", "loso"]


def _rank(order):
    lookup = {v: i for i, v in enumerate(order)}
    return lambda s: s.map(lambda v: lookup.get(v, len(order)))


def _as_int(s: pd.Series) -> pd.Series:
    # "" (official split's fold_id) sorts first
    return pd.to_numeric(s, errors="coerce").fillna(-1)


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    # everything as text, so values are written back exactly as train.py logged them
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def merge(parts_dir=PARTS_DIR, out_csv=RAW_RESULTS_CSV, dataset_order=None, verbose=True) -> pd.DataFrame:
    parts_dir, out_csv = Path(parts_dir), Path(out_csv)
    part_files = sorted(parts_dir.rglob("*.csv")) if parts_dir.exists() else []
    frames = [f for f in [_read(out_csv)] + [_read(p) for p in part_files] if len(f)]
    if not frames:
        if verbose:
            print(f"merge: nothing to merge (no rows in {out_csv} or {parts_dir})")
        return pd.DataFrame(columns=RESULT_COLUMNS)

    df = pd.concat(frames, ignore_index=True).fillna("")
    n_in = len(df)
    df = (df.sort_values("timestamp", kind="stable")
            .drop_duplicates(subset=RUN_KEY, keep="last"))

    ds_order = list(dataset_order or []) + [d for d in DATASET_ORDER if d not in (dataset_order or [])]
    sort_cols = {
        "_ds": _rank(ds_order)(df["dataset"]),
        "_model": _rank(MODEL_ORDER)(df["model"]),
        "_split": _rank(SPLIT_ORDER)(df["split_mode"]),
        "_fold": _as_int(df["fold_id"]),
        "_variant": _rank(VARIANT_ORDER)(df["variant"]),
        "_seed": _as_int(df["seed"]),
    }
    df = (df.assign(**sort_cols)
            .sort_values(list(sort_cols) + ["dataset", "model", "variant"], kind="stable")
            .drop(columns=list(sort_cols)))
    cols = RESULT_COLUMNS + [c for c in df.columns if c not in RESULT_COLUMNS]
    df = df.reindex(columns=cols, fill_value="")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_name(out_csv.name + ".tmp")
    df.to_csv(tmp, index=False, lineterminator="\n", encoding="utf-8")
    os.replace(tmp, out_csv)
    if verbose:
        print(f"merge: {len(df)} runs -> {out_csv} "
              f"(from {len(part_files)} part files + existing; {n_in - len(df)} duplicate rows dropped)")
    return df


def main():
    p = argparse.ArgumentParser(description="Merge per-job result CSVs into raw_results.csv (fixed order).")
    p.add_argument("--parts_dir", default=str(PARTS_DIR))
    p.add_argument("--out_csv", default=str(RAW_RESULTS_CSV))
    p.add_argument("--datasets", nargs="+", default=None, help="dataset order in the output")
    args = p.parse_args()
    merge(args.parts_dir, args.out_csv, args.datasets)


if __name__ == "__main__":
    main()
