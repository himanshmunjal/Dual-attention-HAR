"""
src/evaluate.py

Builds the paper's tables and figures from saved results -- no retraining:

  results/tables/master_comparison.csv / .md
      one row per (model, variant, dataset, split_mode): mean +/- std accuracy
      and macro-F1 over all runs, std across LOSO folds, params, FLOPs,
      CPU/GPU latency (from efficiency.csv) and Wilcoxon p-values vs. the
      proposed model (from significance.csv, LOSO rows only)
  results/tables/confusion/<model>[_<variant>]_<dataset>_<split>.csv
      confusion matrix summed over all folds and seeds (counts)
  results/figures/
      confusion_*.png            row-normalised confusion matrices (LOSO)
      loso_boxplot.png           per-fold macro-F1 per model, one panel per dataset
      efficiency_tradeoff.png    LOSO macro-F1 vs params and vs FLOPs

Run after training, src/profile_efficiency.py and src/stats_test.py:
    python -m src.evaluate
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.preprocessing import CLASS_NAMES  # noqa: E402
from src.stats_test import label, run_tests  # noqa: E402
from src.utils import load_results, RAW_RESULTS_CSV, PREDS_DIR, RESULTS_DIR, NO_VARIANT  # noqa: E402

TABLES = RESULTS_DIR / "tables"
FIGURES = RESULTS_DIR / "figures"
EFFICIENCY_CSV = TABLES / "efficiency.csv"
PROPOSED = "dual_attention_bigru:full"

MODEL_ORDER = ["bigru_baseline", "bilstm_attention_baseline", "cnn_bigru_reference", "deepconvlstm",
               "dual_attention_bigru"]
VARIANT_ORDER = [NO_VARIANT, "full", "plain", "channel_only", "temporal_only"]
DATASET_ORDER = ["ucihar", "wisdm", "pamap2"]
SHORT = {"bigru_baseline": "BiGRU", "bilstm_attention_baseline": "BiLSTM+Attn",
         "cnn_bigru_reference": "CNN+BiGRU", "deepconvlstm": "DeepConvLSTM",
         "dual_attention_bigru": "DualAttn-BiGRU"}
# fixed categorical order (validated palette, light surface); entity -> colour never changes
MODEL_COLORS = {"bigru_baseline": "#2a78d6", "bilstm_attention_baseline": "#eb6834",
                "cnn_bigru_reference": "#1baf7a", "dual_attention_bigru": "#eda100",
                "deepconvlstm": "#e87ba4"}
MODEL_MARKERS = {"bigru_baseline": "o", "bilstm_attention_baseline": "s",
                 "cnn_bigru_reference": "^", "dual_attention_bigru": "D", "deepconvlstm": "v"}
INK, INK_MUTED, GRID, NEUTRAL = "#1f1f1e", "#6b6a64", "#e6e5e0", "#b9b8b0"
HIGHLIGHT = "#2a78d6"


def pretty(model: str, variant: str) -> str:
    s = SHORT.get(model, model)
    return s if variant == NO_VARIANT else f"{s} ({variant})"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(NEUTRAL)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _sort(df: pd.DataFrame) -> pd.DataFrame:
    key = dict(model=lambda s: s.map({m: i for i, m in enumerate(MODEL_ORDER)}),
               variant=lambda s: s.map({v: i for i, v in enumerate(VARIANT_ORDER)}),
               dataset=lambda s: s.map({d: i for i, d in enumerate(DATASET_ORDER)}))
    cols = [c for c in ["dataset", "split_mode", "model", "variant"] if c in df.columns]
    return df.sort_values(cols, key=lambda s: key[s.name](s) if s.name in key else s).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Master comparison table
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["dataset", "split_mode", "model", "variant"])
    out = g.agg(n_runs=("seed", "size"), n_seeds=("seed", "nunique"),
                n_folds=("fold_id", lambda s: s.nunique() if s.notna().any() else 1),
                accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
                params=("params", "first"), flops=("flops", "first"),
                mean_best_epoch=("best_epoch", "mean"), mean_train_time_s=("train_time_s", "mean")).reset_index()
    # spread across subjects: std of the per-fold (seed-averaged) score
    fold_means = df[df.split_mode == "loso"].groupby(
        ["dataset", "split_mode", "model", "variant", "fold_id"])["macro_f1"].mean()
    fold_std = fold_means.groupby(level=[0, 1, 2, 3]).std().rename("macro_f1_std_across_folds").reset_index()
    return out.merge(fold_std, how="left", on=["dataset", "split_mode", "model", "variant"])


def attach_efficiency(summary: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[evaluate] {path} not found -- run `python -m src.profile_efficiency`; latency columns omitted")
        return summary
    eff = pd.read_csv(path)
    eff["variant"] = eff["variant"].fillna(NO_VARIANT)
    lat_cols = [c for c in eff.columns if c.startswith("latency_ms_")]
    wide = eff.pivot_table(index=["model", "variant", "dataset"], columns="device", values=lat_cols)
    wide.columns = [f"{lat.replace('latency_ms_', 'latency_')}_{dev}_ms" for lat, dev in wide.columns]
    wide = wide.reset_index()
    names = eff.groupby("device")["device_name"].first().to_dict()
    print("[evaluate] latency devices: " + ", ".join(f"{k} = {v}" for k, v in names.items()))
    # profiled params/flops are authoritative (identical architectures), fill if missing
    prof = eff.drop_duplicates(["model", "variant", "dataset"])[["model", "variant", "dataset", "params", "flops"]]
    summary = summary.merge(prof, how="left", on=["model", "variant", "dataset"], suffixes=("", "_prof"))
    for c in ("params", "flops"):
        summary[c] = summary[c].fillna(summary.pop(f"{c}_prof"))
    return summary.merge(wide, how="left", on=["model", "variant", "dataset"])


def attach_significance(summary: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    if sig.empty:
        return summary
    s = sig[(sig.metric == "macro_f1") & (sig.dataset != "ALL")].copy()
    s["arm"] = s["comparator"]
    s = s[["dataset", "arm", "mean_diff", "wilcoxon_p", "wilcoxon_p_holm"]].rename(columns={
        "mean_diff": "f1_diff_proposed_minus_this", "wilcoxon_p": "wilcoxon_p_vs_proposed",
        "wilcoxon_p_holm": "wilcoxon_p_holm_vs_proposed"})
    s["split_mode"] = "loso"
    summary = summary.copy()
    summary["arm"] = [label(m, v) for m, v in zip(summary.model, summary.variant)]
    return summary.merge(s, how="left", on=["dataset", "split_mode", "arm"]).drop(columns="arm")


def write_markdown(summary: pd.DataFrame, path: Path):
    def pm(m, s):
        return f"{m:.4f} ± {s:.4f}" if pd.notna(s) else f"{m:.4f}"
    lines = []
    for (ds, split), block in summary.groupby(["dataset", "split_mode"], sort=False):
        lines += [f"\n### {ds} — {split}\n"]
        cols = ["Model", "Runs", "Accuracy", "Macro-F1", "Params", "MFLOPs"]
        lat = [c for c in block.columns if c.startswith("latency_") and block[c].notna().any()]
        cols += [c.replace("latency_", "").replace("_ms", " (ms)") for c in lat]
        if "wilcoxon_p_vs_proposed" in block and block["wilcoxon_p_vs_proposed"].notna().any():
            cols += ["p vs proposed (Holm)"]
        lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in block.iterrows():
            cells = [pretty(r.model, r.variant), str(int(r.n_runs)),
                     pm(r.accuracy_mean, r.accuracy_std), pm(r.macro_f1_mean, r.macro_f1_std),
                     f"{int(r.params):,}" if pd.notna(r.params) else "",
                     f"{r.flops / 1e6:.2f}" if pd.notna(r.flops) else ""]
            cells += [f"{r[c]:.3f}" if pd.notna(r[c]) else "" for c in lat]
            if "p vs proposed (Holm)" in cols:
                p = r.get("wilcoxon_p_holm_vs_proposed")
                cells += [f"{p:.4f}" if pd.notna(p) else "—"]
            lines.append("| " + " | ".join(cells) + " |")
    path.write_text("# Master comparison\n\nMean ± std over all runs (folds × seeds). "
                    "Latency: median ms per batch (bs1 / bs32). p: two-sided Wilcoxon on "
                    "LOSO fold-level macro-F1, Holm-corrected.\n" + "\n".join(lines) + "\n",
                    encoding="utf-8")


# ---------------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------------
def confusion_matrices(preds_dir: Path, df: pd.DataFrame):
    out_dir = TABLES / "confusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    from src.utils import run_name
    groups = {}
    for r in df.itertuples():
        fold = None if pd.isna(r.fold_id) else int(r.fold_id)
        f = preds_dir / f"{run_name(r.model, r.variant, r.dataset, r.split_mode, fold, int(r.seed))}.npz"
        if f.exists():
            groups.setdefault((r.model, r.variant, r.dataset, r.split_mode), []).append(f)
    for (model, variant, ds, split), files in groups.items():
        cm, classes = None, None
        for f in files:
            z = np.load(f)
            classes = z["classes"]
            k = len(classes)
            m = np.bincount(z["y_true"] * k + z["y_pred"], minlength=k * k).reshape(k, k)
            cm = m if cm is None else cm + m
        names = [CLASS_NAMES[ds].get(int(c), str(c)) for c in classes]
        stem = "_".join([model] + ([variant] if variant != NO_VARIANT else []) + [ds, split])
        pd.DataFrame(cm, index=[f"true_{n}" for n in names], columns=[f"pred_{n}" for n in names]) \
            .to_csv(out_dir / f"{stem}.csv")
        if split == "loso":
            _plot_confusion(cm, names, f"{pretty(model, variant)} — {ds} LOSO ({len(files)} runs)",
                            FIGURES / f"confusion_{stem}.png")
    print(f"[evaluate] confusion matrices for {len(groups)} model/dataset/split groups -> {out_dir}")


def _plot_confusion(cm, names, title, path):
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    k = len(names)
    fig, ax = plt.subplots(figsize=(0.55 * k + 2.2, 0.5 * k + 1.6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)  # sequential: one hue, light -> dark
    ax.set_xticks(range(k), names, rotation=45, ha="right", fontsize=7, color=INK)
    ax.set_yticks(range(k), names, fontsize=7, color=INK)
    ax.set_xlabel("Predicted", fontsize=8, color=INK_MUTED)
    ax.set_ylabel("True", fontsize=8, color=INK_MUTED)
    for i in range(k):
        for j in range(k):
            if norm[i, j] >= 0.01:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.55 else INK)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7, colors=INK_MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def loso_boxplot(df: pd.DataFrame):
    d = df[df.split_mode == "loso"]
    if d.empty:
        return
    fold = d.groupby(["dataset", "model", "variant", "fold_id"])["macro_f1"].mean().reset_index()
    datasets = [x for x in DATASET_ORDER if x in fold.dataset.unique()]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 4.2), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        f = _sort(fold[fold.dataset == ds])
        arms = f.drop_duplicates(["model", "variant"])[["model", "variant"]].itertuples(index=False)
        arms = list(arms)
        data = [f[(f.model == m) & (f.variant == v)]["macro_f1"].to_numpy() for m, v in arms]
        bp = ax.boxplot(data, vert=True, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, linewidth=1.5),
                        whiskerprops=dict(color=INK_MUTED), capprops=dict(color=INK_MUTED))
        for patch, (m, v) in zip(bp["boxes"], arms):
            is_prop = label(m, v) == PROPOSED  # highlight the proposed model, others neutral
            patch.set(facecolor=HIGHLIGHT if is_prop else "#ecebe6",
                      edgecolor=HIGHLIGHT if is_prop else NEUTRAL, linewidth=1)
        rng = np.random.RandomState(0)
        for i, vals in enumerate(data, start=1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals, s=12, color=INK,
                       alpha=0.6, zorder=3, linewidths=0)
        ax.set_xticks(range(1, len(arms) + 1), [pretty(m, v) for m, v in arms],
                      rotation=40, ha="right", fontsize=7, color=INK)
        ax.set_title(f"{ds} — LOSO folds", fontsize=9, color=INK, loc="left")
        ax.set_ylabel("Macro-F1 (seed-averaged, one dot per fold)", fontsize=8, color=INK_MUTED)
        _style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "loso_boxplot.png", dpi=200)
    plt.close(fig)


def efficiency_tradeoff(summary: pd.DataFrame):
    s = summary[(summary.split_mode == "loso") & summary.variant.isin([NO_VARIANT, "full"])]
    if s.empty:
        return
    datasets = [x for x in DATASET_ORDER if x in s.dataset.unique()]
    measures = [("params", "Trainable parameters"), ("flops", "FLOPs per window")]
    fig, axes = plt.subplots(len(datasets), 2, figsize=(9, 3.2 * len(datasets)), squeeze=False)
    for r, ds in enumerate(datasets):
        b = s[s.dataset == ds]
        for c, (col, xlabel) in enumerate(measures):
            ax = axes[r][c]
            for _, row in b.iterrows():
                if pd.isna(row[col]):
                    continue
                ax.errorbar(row[col], row.macro_f1_mean, yerr=row.macro_f1_std_across_folds,
                            fmt=MODEL_MARKERS[row.model], ms=8, color=MODEL_COLORS[row.model],
                            mec="white", mew=1.5, ecolor=NEUTRAL, elinewidth=1, capsize=0)
                ax.annotate(SHORT[row.model], (row[col], row.macro_f1_mean), xytext=(7, 4),
                            textcoords="offset points", fontsize=7, color=INK)
            ax.set_xlabel(xlabel, fontsize=8, color=INK_MUTED)
            if c == 0:
                ax.set_ylabel(f"{ds}\nLOSO macro-F1", fontsize=8, color=INK_MUTED)
            ax.xaxis.set_major_formatter(matplotlib.ticker.EngFormatter())
            ax.margins(x=0.25, y=0.3)
            _style(ax)
    fig.suptitle("Accuracy vs. cost (error bars: std across LOSO folds)", fontsize=10, color=INK, x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "efficiency_tradeoff.png", dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_csv", default=str(RAW_RESULTS_CSV))
    p.add_argument("--preds_dir", default=str(PREDS_DIR))
    p.add_argument("--efficiency_csv", default=str(EFFICIENCY_CSV))
    p.add_argument("--out_dir", default=str(RESULTS_DIR),
                   help="writes <out_dir>/tables and <out_dir>/figures (default: results/)")
    args = p.parse_args()

    global TABLES, FIGURES
    TABLES, FIGURES = Path(args.out_dir) / "tables", Path(args.out_dir) / "figures"

    df = load_results(args.results_csv)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    summary = attach_efficiency(summarize(df), Path(args.efficiency_csv))
    sig = run_tests(df, PROPOSED, ["macro_f1"])
    summary = _sort(attach_significance(summary, sig))
    summary.to_csv(TABLES / "master_comparison.csv", index=False)
    write_markdown(summary, TABLES / "master_comparison.md")
    print(f"[evaluate] master table ({len(summary)} rows) -> {TABLES / 'master_comparison.csv'} (+ .md)")

    confusion_matrices(Path(args.preds_dir), df)
    loso_boxplot(df)
    efficiency_tradeoff(summary)
    print(f"[evaluate] figures -> {FIGURES}")

    with pd.option_context("display.width", 200):
        cols = ["dataset", "split_mode", "model", "variant", "n_runs", "accuracy_mean", "macro_f1_mean",
                "macro_f1_std", "params", "flops"]
        print(summary[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
