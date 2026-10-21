"""
paper/analysis.py

Builds every figure, table and number used in the paper from the protocol-v2
results (results/tables/raw_results.csv, results/tables/efficiency.csv,
results/preds/*.npz) plus the archived v1 run for the protocol comparison.

    python3 paper/analysis.py

Outputs:
    paper/figures/<name>.pdf + .png   (PDF for LaTeX, PNG for the Word version)
    paper/generated/<name>.tex        (LaTeX tables, \\input by paper_src.tex)
    paper/generated/numbers.json      (every number quoted in the text)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"
GEN = ROOT / "paper" / "generated"
FIG.mkdir(parents=True, exist_ok=True)
GEN.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20261021)
NUM = {}  # every number quoted in the paper

# ---------------------------------------------------------------------------
# names, order, colours
# ---------------------------------------------------------------------------
ARMS = [
    "bigru_baseline:-",
    "bilstm_attention_baseline:-",
    "cnn_bigru_reference:-",
    "deepconvlstm:-",
    "dual_attention_bigru:full",
    "dual_attention_bigru:plain",
    "dual_attention_bigru:channel_only",
    "dual_attention_bigru:temporal_only",
]
SHORT = {
    "bigru_baseline:-": "BiGRU",
    "bilstm_attention_baseline:-": "BiLSTM+Attn",
    "cnn_bigru_reference:-": "CNN+BiGRU",
    "deepconvlstm:-": "DeepConvLSTM",
    "dual_attention_bigru:full": "SG-DA-BiGRU",
    "dual_attention_bigru:plain": "DA-BiGRU",
    "dual_attention_bigru:channel_only": "Channel-only",
    "dual_attention_bigru:temporal_only": "Temporal-only",
}
COLOR = {
    "bigru_baseline:-": "#6b6b6b",
    "bilstm_attention_baseline:-": "#a5a5a5",
    "cnn_bigru_reference:-": "#d9822b",
    "deepconvlstm:-": "#b5502a",
    "dual_attention_bigru:full": "#1f5fa8",
    "dual_attention_bigru:plain": "#4f8fd1",
    "dual_attention_bigru:channel_only": "#86b7e6",
    "dual_attention_bigru:temporal_only": "#2a9d8f",
}
PROPOSED = "dual_attention_bigru:full"
DATASETS = ["ucihar", "wisdm", "pamap2"]
DNAME = {"ucihar": "UCI-HAR", "wisdm": "WISDM", "pamap2": "PAMAP2", "ALL": "Pooled"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def write_tex(name, body):
    (GEN / f"{name}.tex").write_text(body)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
raw = pd.read_csv(RES / "tables" / "raw_results.csv")
raw = raw[raw.protocol == "v2"].copy()
raw["arm"] = raw.model + ":" + raw.variant.astype(str)
assert len(raw) == 1320, len(raw)
loso = raw[raw.split_mode == "loso"]
off = raw[raw.split_mode == "official"]

NUM["n_runs"] = int(len(raw))
NUM["n_loso_runs"] = int(len(loso))
NUM["gpu_hours"] = round(raw.train_time_s.sum() / 3600, 1)


def fold_table(metric):
    """rows (dataset, fold), cols arm, value = seed-mean metric."""
    t = loso.groupby(["dataset", "fold_id", "arm"])[metric].mean().unstack()
    return t[ARMS]


F1 = fold_table("macro_f1")
ACC = fold_table("accuracy")
NUM["n_folds"] = {d: int(len(F1.loc[d])) for d in DATASETS}

# ---------------------------------------------------------------------------
# Table: main LOSO results (fold-level mean +- sd)
# ---------------------------------------------------------------------------
def fmt(m, s, bold=False):
    x = f"{m:.3f} $\\pm$ {s:.3f}"
    return f"\\textbf{{{m:.3f}}} $\\pm$ {s:.3f}" if bold else x


rows = []
summary = {}
for arm in ARMS:
    cells = []
    for d in DATASETS:
        f = F1.loc[d, arm]
        a = ACC.loc[d, arm]
        summary[(arm, d)] = (f.mean(), f.std(ddof=1), a.mean(), a.std(ddof=1))
    summary[(arm, "ALL")] = (F1[arm].mean(), F1[arm].std(ddof=1), ACC[arm].mean(), ACC[arm].std(ddof=1))

best = {d: max(ARMS, key=lambda a: summary[(a, d)][0]) for d in DATASETS + ["ALL"]}
best_acc = {d: max(ARMS, key=lambda a: summary[(a, d)][2]) for d in DATASETS + ["ALL"]}
lines = []
for arm in ARMS:
    c = [SHORT[arm]]
    for d in DATASETS + ["ALL"]:
        m, s, am, as_ = summary[(arm, d)]
        c.append(fmt(m, s, best[d] == arm))
    lines.append(" & ".join(c) + r" \\")
    if arm == "deepconvlstm:-":
        lines.append(r"\midrule")
write_tex("tab_loso_f1", "\n".join(lines))

lines = []
for arm in ARMS:
    c = [SHORT[arm]]
    for d in DATASETS + ["ALL"]:
        m, s, am, as_ = summary[(arm, d)]
        c.append(fmt(am, as_, best_acc[d] == arm))
    lines.append(" & ".join(c) + r" \\")
    if arm == "deepconvlstm:-":
        lines.append(r"\midrule")
write_tex("tab_loso_acc", "\n".join(lines))
NUM["loso"] = {SHORT[a]: {d: [round(v, 4) for v in summary[(a, d)]] for d in DATASETS + ["ALL"]} for a in ARMS}

# ---------------------------------------------------------------------------
# Table: official / fixed split
# ---------------------------------------------------------------------------
O = off.groupby(["dataset", "arm"]).macro_f1.agg(["mean", "std"])
lines = []
best_off = {d: O.loc[d]["mean"].idxmax() for d in DATASETS}
for arm in ARMS:
    c = [SHORT[arm]]
    for d in DATASETS:
        m, s = O.loc[(d, arm)]
        c.append(fmt(m, s, best_off[d] == arm))
    lines.append(" & ".join(c) + r" \\")
    if arm == "deepconvlstm:-":
        lines.append(r"\midrule")
write_tex("tab_official", "\n".join(lines))

# rank agreement between the fixed split and LOSO
rank_shift = {}
for d in DATASETS:
    lo = pd.Series({a: summary[(a, d)][0] for a in ARMS})
    of = O.loc[d]["mean"][ARMS]
    tau, p = stats.kendalltau(lo.values, of.values)
    rank_shift[d] = dict(tau=round(tau, 3), p=round(p, 3),
                         loso_rank=lo.rank(ascending=False).astype(int).to_dict(),
                         off_rank=of.rank(ascending=False).astype(int).to_dict())
NUM["rank_shift"] = {d: {"tau": v["tau"], "p": v["p"]} for d, v in rank_shift.items()}
NUM["official"] = {d: {SHORT[a]: [round(O.loc[(d, a)]["mean"], 4), round(O.loc[(d, a)]["std"], 4)] for a in ARMS} for d in DATASETS}

# ---------------------------------------------------------------------------
# significance: Wilcoxon + Holm (recomputed here so the paper is self-contained;
# identical to src/stats_test.py), bootstrap CI of the mean difference
# ---------------------------------------------------------------------------
def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx])
        adj[idx] = min(1.0, run)
    return adj


def boot_ci(x, n=10000):
    x = np.asarray(x)
    b = RNG.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return np.percentile(b, [2.5, 97.5])


sig = []
for scope in DATASETS + ["ALL"]:
    T = F1 if scope == "ALL" else F1.loc[scope]
    comps = [a for a in ARMS if a != PROPOSED]
    ps, recs = [], []
    for c in comps:
        diff = (T[PROPOSED] - T[c]).values
        p = stats.wilcoxon(T[PROPOSED].values, T[c].values, alternative="two-sided").pvalue
        lo, hi = boot_ci(diff)
        recs.append(dict(scope=scope, comparator=c, n=len(diff), mean_diff=diff.mean(), ci_lo=lo, ci_hi=hi,
                         wins=int((diff > 0).sum()), p=p))
        ps.append(p)
    for r, ph in zip(recs, holm(ps)):
        r["p_holm"] = ph
    sig += recs
SIG = pd.DataFrame(sig)

# cross-check against the pipeline's own significance.csv
ref = pd.read_csv(RES / "tables" / "significance.csv")
ref = ref[ref.metric == "macro_f1"].set_index(["dataset", "comparator"])
for r in SIG.itertuples():
    q = ref.loc[(r.scope, r.comparator.removesuffix(":-"))]
    assert abs(q.wilcoxon_p - r.p) < 1e-9 and abs(q.wilcoxon_p_holm - r.p_holm) < 1e-9, (r, q)


def pfmt(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


lines = []
for c in [a for a in ARMS if a != PROPOSED]:
    cells = [SHORT[c]]
    for scope in DATASETS + ["ALL"]:
        r = SIG[(SIG.scope == scope) & (SIG.comparator == c)].iloc[0]
        cells.append(f"{r.mean_diff:+.4f} & {r.wins}/{r.n} & {pfmt(r.p)} & {pfmt(r.p_holm)}")
    lines.append(" & ".join(cells) + r" \\")
# per-dataset is wide; build two tables: pooled detailed + per-dataset compact
lines_pooled = []
for c in [a for a in ARMS if a != PROPOSED]:
    r = SIG[(SIG.scope == "ALL") & (SIG.comparator == c)].iloc[0]
    lines_pooled.append(f"{SHORT[c]} & {r.mean_diff:+.4f} & [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] & {r.wins}/{r.n} & {pfmt(r.p)} & {pfmt(r.p_holm)}" + r" \\")
write_tex("tab_sig_pooled", "\n".join(lines_pooled))

lines_ds = []
for c in [a for a in ARMS if a != PROPOSED]:
    cells = [SHORT[c]]
    for scope in DATASETS:
        r = SIG[(SIG.scope == scope) & (SIG.comparator == c)].iloc[0]
        cells.append(f"{r.mean_diff:+.4f} & {r.wins}/{r.n} & {pfmt(r.p_holm)}")
    lines_ds.append(" & ".join(cells) + r" \\")
write_tex("tab_sig_dataset", "\n".join(lines_ds))
NUM["sig"] = {f"{r.scope}|{SHORT[r.comparator]}": dict(diff=round(r.mean_diff, 4), lo=round(r.ci_lo, 4), hi=round(r.ci_hi, 4),
                                                       wins=f"{r.wins}/{r.n}", p=round(r.p, 4), p_holm=round(r.p_holm, 4))
              for r in SIG.itertuples()}

# smallest attainable two-sided Wilcoxon p per fold count
for d in DATASETS + ["ALL"]:
    n = 30 if d == "ALL" else NUM["n_folds"][d]
    NUM.setdefault("wilcoxon_floor", {})[d] = 2 / 2 ** n

# ---------------------------------------------------------------------------
# Friedman + Nemenyi critical difference
# ---------------------------------------------------------------------------
ranks = F1.rank(axis=1, ascending=False)
mean_rank = ranks.mean()[ARMS]
fr = stats.friedmanchisquare(*[F1[a].values for a in ARMS])
k, N = len(ARMS), len(F1)
q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}[k]
CD = q05 * np.sqrt(k * (k + 1) / (6 * N))
# Iman-Davenport correction
chi = fr.statistic
FF = (N - 1) * chi / (N * (k - 1) - chi)
pFF = 1 - stats.f.cdf(FF, k - 1, (k - 1) * (N - 1))
NUM["friedman"] = dict(chi2=round(chi, 3), p=round(fr.pvalue, 4), FF=round(FF, 3), pFF=round(pFF, 4), CD=round(CD, 3), N=N, k=k)
NUM["mean_rank"] = {SHORT[a]: round(mean_rank[a], 3) for a in ARMS}

# per-dataset ranks too
for d in DATASETS:
    NUM.setdefault("mean_rank_ds", {})[d] = {SHORT[a]: round(v, 2) for a, v in F1.loc[d].rank(axis=1, ascending=False).mean().items()}


def cd_diagram():
    order = mean_rank.sort_values()
    fig, ax = plt.subplots(figsize=(6.6, 1.9))
    lo_r, hi_r = 1, k
    ax.set_xlim(lo_r - 0.3, hi_r + 0.3)
    ax.set_ylim(0.25, 1.06)
    ax.axis("off")
    y0 = 0.82
    ax.plot([lo_r, hi_r], [y0, y0], color="k", lw=0.8)
    for r in range(lo_r, hi_r + 1):
        ax.plot([r, r], [y0, y0 + 0.035], color="k", lw=0.8)
        ax.text(r, y0 + 0.06, str(r), ha="center", va="bottom", fontsize=8)
    # CD bar
    ax.plot([lo_r, lo_r + CD], [0.98, 0.98], color="k", lw=1.2)
    ax.plot([lo_r, lo_r], [0.96, 1.0], color="k", lw=1.2)
    ax.plot([lo_r + CD, lo_r + CD], [0.96, 1.0], color="k", lw=1.2)
    ax.text(lo_r + CD / 2, 1.0, f"CD = {CD:.2f}", ha="center", va="bottom", fontsize=8)
    names = list(order.index)
    half = (len(names) + 1) // 2
    for i, a in enumerate(names):
        r = order[a]
        if i < half:
            yy = y0 - 0.14 - 0.12 * i
            ax.plot([r, r, lo_r - 0.1], [y0, yy, yy], color=COLOR[a], lw=1.0)
            ax.text(lo_r - 0.15, yy, f"{SHORT[a]} ({r:.2f})", ha="right", va="center", fontsize=8, color="k")
        else:
            j = len(names) - 1 - i
            yy = y0 - 0.14 - 0.12 * j
            ax.plot([r, r, hi_r + 0.1], [y0, yy, yy], color=COLOR[a], lw=1.0)
            ax.text(hi_r + 0.15, yy, f"({r:.2f}) {SHORT[a]}", ha="left", va="center", fontsize=8, color="k")
    # cliques: maximal groups whose rank span < CD
    vals = order.values
    cliques = []
    for i in range(len(vals)):
        j = i
        while j + 1 < len(vals) and vals[j + 1] - vals[i] < CD:
            j += 1
        if j > i and not any(i >= a and j <= b for a, b in cliques):
            cliques.append((i, j))
    for n, (i, j) in enumerate(cliques):
        yy = y0 - 0.07 - 0.035 * n
        ax.plot([vals[i] - 0.03, vals[j] + 0.03], [yy, yy], color="k", lw=2.2, solid_capstyle="round")
    save(fig, "fig_cd")
    return cliques


cliques = cd_diagram()
NUM["cd_cliques"] = [[SHORT[mean_rank.sort_values().index[t]] for t in range(i, j + 1)] for i, j in cliques]

# ---------------------------------------------------------------------------
# Bayesian signed-rank test (Benavoli et al. 2017), ROPE = +-0.01 macro-F1
# ---------------------------------------------------------------------------
def bayes_signed_rank(diff, rope=0.01, s=0.5, n_samples=20000):
    z = np.concatenate([[0.0], np.asarray(diff)])
    n = len(z)
    alpha = np.concatenate([[s], np.ones(n - 1)])
    W = RNG.dirichlet(alpha, size=n_samples)          # (S, n)
    pair = (z[:, None] + z[None, :]) / 2               # Walsh averages
    L = (pair < -rope).astype(float)
    R = (pair > rope).astype(float)
    pl = np.einsum("si,ij,sj->s", W, L, W)
    pr = np.einsum("si,ij,sj->s", W, R, W)
    pe = 1 - pl - pr
    win = np.argmax(np.stack([pl, pe, pr], 1), 1)
    return dict(p_worse=float((win == 0).mean()), p_rope=float((win == 1).mean()), p_better=float((win == 2).mean()))


bayes = {}
for c in [a for a in ARMS if a != PROPOSED]:
    bayes[SHORT[c]] = {k_: round(v, 3) for k_, v in bayes_signed_rank((F1[PROPOSED] - F1[c]).values).items()}
NUM["bayes"] = bayes
lines = []
for c in [a for a in ARMS if a != PROPOSED]:
    b = bayes[SHORT[c]]
    lines.append(f"{SHORT[c]} & {b['p_better']:.3f} & {b['p_rope']:.3f} & {b['p_worse']:.3f}" + r" \\")
write_tex("tab_bayes", "\n".join(lines))

# ---------------------------------------------------------------------------
# variance decomposition: seed vs fold (subject) vs model
# ---------------------------------------------------------------------------
var = {}
for d in DATASETS:
    sub = loso[loso.dataset == d]
    seed_sd = sub.groupby(["fold_id", "arm"]).macro_f1.std(ddof=1).mean()
    fold_sd = F1.loc[d].std(ddof=1).mean()
    model_sd = F1.loc[d].mean().std(ddof=1)
    # share of variance with a two-way (fold x arm) decomposition on seed means
    M = F1.loc[d].values
    grand = M.mean()
    ss_fold = M.shape[1] * ((M.mean(1) - grand) ** 2).sum()
    ss_arm = M.shape[0] * ((M.mean(0) - grand) ** 2).sum()
    ss_tot = ((M - grand) ** 2).sum()
    var[d] = dict(seed_sd=seed_sd, fold_sd=fold_sd, model_sd=model_sd,
                  share_fold=ss_fold / ss_tot, share_arm=ss_arm / ss_tot, share_resid=1 - (ss_fold + ss_arm) / ss_tot)
NUM["variance"] = {d: {k_: round(v, 4) for k_, v in x.items()} for d, x in var.items()}

# per-subject macro-F1 of the proposed model (from saved predictions)
PRED = RES / "preds"


def preds(arm, d, split="loso"):
    m, v = arm.split(":")
    stem = m if v == "-" else f"{m}_{v}"
    out = []
    for f in sorted(PRED.glob(f"{stem}_{d}_{split}_*seed*.npz")):
        # guard against e.g. "dual_attention_bigru_full" matching "..._full_..." only
        tail = f.stem[len(stem) + 1:]
        if not tail.startswith(d + "_" + split):
            continue
        z = np.load(f)
        seed = int(f.stem.rsplit("seed", 1)[1])
        out.append((seed, z["y_true"], z["y_pred"], z["subject_test"], z["classes"]))
    return out


subj_f1 = {}
for d in DATASETS:
    rec = {}
    for seed, yt, yp, st, cl in preds(PROPOSED, d):
        for s in np.unique(st):
            m = st == s
            # same definition as the training pipeline: sklearn's default label set
            # (classes present in y_true or y_pred), so absent-class false positives count
            rec.setdefault(int(s), []).append(f1_score(yt[m], yp[m], average="macro"))
    subj_f1[d] = {s: float(np.mean(v)) for s, v in rec.items()}
NUM["subject_f1"] = {d: dict(min=round(min(v.values()), 3), max=round(max(v.values()), 3), n=len(v)) for d, v in subj_f1.items()}


def fig_variance():
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), gridspec_kw=dict(width_ratios=[1, 1.35]))
    ax = axes[0]
    x = np.arange(len(DATASETS))
    w = 0.26
    for i, (key, lab, col) in enumerate([("seed_sd", "seed (same fold)", "#b9c6d6"),
                                         ("fold_sd", "held-out subjects (fold)", "#1f5fa8"),
                                         ("model_sd", "model (arm means)", "#d9822b")]):
        ax.bar(x + (i - 1) * w, [var[d][key] for d in DATASETS], w, label=lab, color=col)
    ax.set_xticks(x, [DNAME[d] for d in DATASETS])
    ax.set_ylabel("SD of macro-F1")
    ax.set_title("(a) Sources of variation")
    ax.legend(frameon=False, loc="upper left")
    ax = axes[1]
    for i, d in enumerate(DATASETS):
        v = np.array(sorted(subj_f1[d].values()))
        jit = RNG.uniform(-0.18, 0.18, len(v))
        ax.scatter(np.full(len(v), i) + jit, v, s=11, color="#1f5fa8", alpha=0.75, lw=0)
        ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color="k", lw=1.2)
    ax.set_xticks(range(len(DATASETS)), [DNAME[d] for d in DATASETS])
    ax.set_ylabel("macro-F1 of held-out subject")
    ax.set_title("(b) SG-DA-BiGRU, per test subject")
    ax.set_ylim(0, 1.02)
    save(fig, "fig_variance")


fig_variance()

# ---------------------------------------------------------------------------
# Figure: per-fold LOSO distribution per dataset
# ---------------------------------------------------------------------------
def fig_loso():
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.9), sharey=False)
    for ax, d in zip(axes, DATASETS):
        T = F1.loc[d]
        data = [T[a].values for a in ARMS]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False,
                        medianprops=dict(color="k", lw=1), whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
        for patch, a in zip(bp["boxes"], ARMS):
            patch.set_facecolor(COLOR[a])
            patch.set_alpha(0.55)
            patch.set_linewidth(0.6)
        for i, a in enumerate(ARMS):
            ax.scatter(np.full(len(T), i + 1) + RNG.uniform(-0.15, 0.15, len(T)), T[a].values, s=6, color=COLOR[a], lw=0, zorder=3)
            ax.scatter([i + 1], [T[a].mean()], marker="D", s=12, color="white", edgecolor="k", lw=0.6, zorder=4)
        ax.set_xticks(range(1, len(ARMS) + 1), [SHORT[a] for a in ARMS], rotation=60, ha="right")
        ax.set_title(f"{DNAME[d]} ({len(T)} folds)")
    axes[0].set_ylabel("LOSO macro-F1 (seed mean per fold)")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_loso")


fig_loso()

# ---------------------------------------------------------------------------
# Figure: paired differences (forest plot), pooled + per dataset
# ---------------------------------------------------------------------------
def fig_forest():
    comps = [a for a in ARMS if a != PROPOSED]
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    ys = np.arange(len(comps))[::-1]
    offs = {"ucihar": -0.24, "wisdm": -0.08, "pamap2": 0.08, "ALL": 0.24}
    mk = {"ucihar": "o", "wisdm": "s", "pamap2": "^", "ALL": "D"}
    col = {"ucihar": "#8a8a8a", "wisdm": "#8a8a8a", "pamap2": "#8a8a8a", "ALL": "#1f5fa8"}
    for y, c in zip(ys, comps):
        for scope in DATASETS + ["ALL"]:
            r = SIG[(SIG.scope == scope) & (SIG.comparator == c)].iloc[0]
            yy = y - offs[scope]
            ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=col[scope], lw=1.6 if scope == "ALL" else 0.9)
            ax.scatter([r.mean_diff], [yy], marker=mk[scope], s=26 if scope == "ALL" else 16,
                       color=col[scope], edgecolor="k" if scope == "ALL" else "none", lw=0.5, zorder=3)
        r = SIG[(SIG.scope == "ALL") & (SIG.comparator == c)].iloc[0]
        ax.text(0.081, y, f"p$_{{Holm}}$={r.p_holm:.3f}" if r.p_holm < 1 else "p$_{Holm}$=1.000",
                va="center", fontsize=7.5, color="#1f5fa8")
    ax.axvspan(-0.01, 0.01, color="#e9eef5", zorder=0)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(ys, [f"vs {SHORT[c]}" for c in comps])
    ax.set_xlim(-0.16, 0.08)
    ax.set_xlabel(r"$\Delta$ macro-F1 = SG-DA-BiGRU $-$ comparator  (mean over folds, 95% bootstrap CI)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=mk[s], ls="-", color=col[s], markeredgecolor="k" if s == "ALL" else "none",
                label=DNAME[s] + (" (30 folds)" if s == "ALL" else "")) for s in DATASETS + ["ALL"]]
    h.append(plt.Rectangle((0, 0), 1, 1, color="#e9eef5", label="ROPE $\\pm$0.01"))
    ax.legend(handles=h, frameon=False, loc="lower left", ncol=1, fontsize=7.5)
    save(fig, "fig_forest")


fig_forest()

# ---------------------------------------------------------------------------
# Figure: full vs plain fold scatter (the pre-registered contrast)
# ---------------------------------------------------------------------------
def fig_full_vs_plain():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw=dict(width_ratios=[1, 1.25]))
    ax = axes[0]
    mk = {"ucihar": "o", "wisdm": "s", "pamap2": "^"}
    for d in DATASETS:
        T = F1.loc[d]
        ax.scatter(T["dual_attention_bigru:plain"], T[PROPOSED], marker=mk[d], s=18, label=DNAME[d],
                   color={"ucihar": "#1f5fa8", "wisdm": "#d9822b", "pamap2": "#2a9d8f"}[d], lw=0, alpha=0.85)
    lim = [0.35, 1.0]
    ax.plot(lim, lim, color="k", lw=0.6, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("DA-BiGRU (no spectral gate)")
    ax.set_ylabel("SG-DA-BiGRU (spectral gate)")
    ax.set_title("(a) per-fold macro-F1")
    ax.legend(frameon=False, loc="upper left")
    ax = axes[1]
    diff = (F1[PROPOSED] - F1["dual_attention_bigru:plain"]).values
    # posterior of the mean difference under a Bayesian bootstrap
    W = RNG.dirichlet(np.ones(len(diff)), 20000)
    post = W @ diff
    ax.hist(post, bins=70, color="#4f8fd1", alpha=0.85, density=True)
    ax.axvspan(-0.01, 0.01, color="#e9eef5", zorder=0, label="ROPE $\\pm$0.01")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel(r"mean $\Delta$ macro-F1 (full $-$ plain), 30 folds")
    ax.set_ylabel("posterior density")
    ax.set_title("(b) Bayesian bootstrap of the mean difference")
    p_in = ((post > -0.01) & (post < 0.01)).mean()
    ax.text(0.03, 0.95, f"P(|$\\Delta$|<0.01) = {p_in:.3f}\nP($\\Delta$>0) = {(post > 0).mean():.3f}",
            transform=ax.transAxes, va="top", fontsize=8)
    ax.legend(frameon=False, loc="upper right")
    NUM["full_vs_plain_post"] = dict(p_in_rope=round(float(p_in), 3), p_pos=round(float((post > 0).mean()), 3),
                                     ci=[round(float(np.percentile(post, 2.5)), 4), round(float(np.percentile(post, 97.5)), 4)])
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_full_vs_plain")


fig_full_vs_plain()

# ---------------------------------------------------------------------------
# efficiency
# ---------------------------------------------------------------------------
eff = pd.read_csv(RES / "tables" / "efficiency.csv")
eff["arm"] = eff.model + ":" + eff.variant.astype(str)


def eff_val(arm, d, dev, col):
    r = eff[(eff.arm == arm) & (eff.dataset == d) & (eff.device == dev)]
    return float(r[col].iloc[0])


lines = []
for arm in ARMS:
    cells = [SHORT[arm]]
    for d in DATASETS:
        cells.append(f"{eff_val(arm, d, 'cuda', 'params') / 1000:.1f}")
    for d in DATASETS:
        cells.append(f"{eff_val(arm, d, 'cuda', 'flops') / 1e6:.1f}")
    cells.append(f"{eff_val(arm, 'ucihar', 'cuda', 'latency_ms_bs1'):.2f}")
    cells.append(f"{eff_val(arm, 'ucihar', 'cuda', 'latency_ms_bs32'):.2f}")
    cells.append(f"{eff_val(arm, 'ucihar', 'cpu', 'latency_ms_bs1'):.2f}")
    lines.append(" & ".join(cells) + r" \\")
    if arm == "deepconvlstm:-":
        lines.append(r"\midrule")
write_tex("tab_eff", "\n".join(lines))
NUM["eff"] = {SHORT[a]: {d: dict(params=int(eff_val(a, d, "cuda", "params")), mflops=round(eff_val(a, d, "cuda", "flops") / 1e6, 2),
                                 gpu_bs1=round(eff_val(a, d, "cuda", "latency_ms_bs1"), 3), gpu_bs32=round(eff_val(a, d, "cuda", "latency_ms_bs32"), 3),
                                 cpu_bs1=round(eff_val(a, d, "cpu", "latency_ms_bs1"), 3), cpu_bs32=round(eff_val(a, d, "cpu", "latency_ms_bs32"), 3))
                         for d in DATASETS} for a in ARMS}
NUM["gate_params"] = int(eff_val(PROPOSED, "ucihar", "cuda", "params") - eff_val("dual_attention_bigru:plain", "ucihar", "cuda", "params"))


def fig_eff():
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    for ax, (xcol, dev, xl) in zip(axes, [("flops", "cuda", "MFLOPs per window (log)"),
                                         ("latency_ms_bs32", "cuda", "GPU latency, batch 32 [ms] (log)")]):
        for a in ARMS:
            xs = [eff_val(a, d, dev, xcol) / (1e6 if xcol == "flops" else 1) for d in DATASETS]
            ys = [summary[(a, d)][0] for d in DATASETS]
            for d, x_, y_, m in zip(DATASETS, xs, ys, ["o", "s", "^"]):
                ax.scatter(x_, y_, marker=m, s=26, color=COLOR[a], edgecolor="k", lw=0.3, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel("LOSO macro-F1")
        ax.grid(alpha=0.25, lw=0.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=COLOR[a], label=SHORT[a]) for a in ARMS]
    h += [Line2D([], [], marker=m, ls="", color="w", markeredgecolor="k", label=DNAME[d]) for d, m in zip(DATASETS, "os^")]
    fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.12), fontsize=7.5)
    axes[0].set_title("(a) accuracy vs compute")
    axes[1].set_title("(b) accuracy vs measured latency")
    fig.tight_layout()
    save(fig, "fig_eff")


fig_eff()

# ---------------------------------------------------------------------------
# protocol v1 vs v2: best epoch
# ---------------------------------------------------------------------------
v1 = pd.read_csv(RES / "archive" / "2026-10-15_protocol_v1" / "tables" / "raw_results.csv")
NUM["v1_runs"] = int(len(v1))
be = {}
for name, df in [("v1", v1), ("v2", raw)]:
    be[name] = dict(median={d: float(df[df.dataset == d].best_epoch.median()) for d in DATASETS},
                    le3={d: round(float((df[df.dataset == d].best_epoch <= 3).mean()), 3) for d in DATASETS},
                    le3_all=round(float((df.best_epoch <= 3).mean()), 3),
                    median_all=float(df.best_epoch.median()))
NUM["best_epoch"] = be
NUM["hit_max_epochs"] = int((raw.epochs_trained >= 50).sum())


def fig_epochs():
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.2), sharey=True)
    bins = np.arange(0.5, 51.5, 1)
    for ax, d in zip(axes, DATASETS):
        ax.hist(v1[v1.dataset == d].best_epoch, bins=bins, density=True, color="#c9c9c9", label="v1: select on val loss")
        ax.hist(raw[raw.dataset == d].best_epoch, bins=bins, density=True, histtype="step", color="#1f5fa8", lw=1.2,
                label="v2: select on val macro-F1")
        ax.set_title(DNAME[d])
        ax.set_xlabel("selected (best) epoch")
        ax.set_xlim(0, 50)
    axes[0].set_ylabel("share of runs")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig_epochs")


fig_epochs()

# ---------------------------------------------------------------------------
# fixed split vs LOSO ranks (slope chart)
# ---------------------------------------------------------------------------
def fig_rankshift():
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.7), sharey=True)
    for ax, d in zip(axes, DATASETS):
        lr, orr = rank_shift[d]["loso_rank"], rank_shift[d]["off_rank"]
        for a in ARMS:
            ax.plot([0, 1], [orr[a], lr[a]], color=COLOR[a], lw=1.6 if a == PROPOSED else 1.0, marker="o", ms=3.5)
        ax.set_xticks([0, 1], ["fixed split", "LOSO"])
        ax.set_xlim(-0.15, 1.15)
        ax.set_title(f"{DNAME[d]}  ($\\tau$ = {rank_shift[d]['tau']:.2f})")
        ax.invert_yaxis() if d == "ucihar" else None
    axes[0].set_yticks(range(1, 9))
    axes[0].set_ylabel("rank by macro-F1 (1 = best)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COLOR[a], marker="o", ms=3.5, label=SHORT[a]) for a in ARMS]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.13), fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_rankshift")


fig_rankshift()

# ---------------------------------------------------------------------------
# per-class F1 (LOSO predictions pooled over folds, averaged over seeds)
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    "ucihar": {1: "walk", 2: "upstairs", 3: "downstairs", 4: "sit", 5: "stand", 6: "lie"},
    "wisdm": {0: "walk", 1: "jog", 2: "upstairs", 3: "downstairs", 4: "sit", 5: "stand"},
    "pamap2": {1: "lie", 2: "sit", 3: "stand", 4: "walk", 5: "run", 6: "cycle", 7: "Nordic walk", 12: "asc. stairs",
               13: "desc. stairs", 16: "vacuum", 17: "iron", 24: "rope jump"},
}
perclass = {}
for d in DATASETS:
    for a in ARMS:
        by_seed = {}
        for seed, yt, yp, st, cl in preds(a, d):
            by_seed.setdefault(seed, ([], []))
            by_seed[seed][0].append(yt)
            by_seed[seed][1].append(yp)
        # predictions are stored as encoded indices 0..K-1; `classes` maps them to raw labels
        labs = list(range(len(CLASS_NAMES[d])))
        f = []
        for seed, (yt, yp) in by_seed.items():
            yt, yp = np.concatenate(yt), np.concatenate(yp)
            f.append(f1_score(yt, yp, labels=labs, average=None, zero_division=0))
        perclass[(d, a)] = np.mean(f, 0)
        assert len(by_seed) == 5, (d, a, len(by_seed))


def fig_perclass():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.3), gridspec_kw=dict(width_ratios=[6, 6, 12]))
    for ax, d in zip(axes, DATASETS):
        labs = sorted(CLASS_NAMES[d])
        base = perclass[(d, "bigru_baseline:-")]
        M = np.stack([perclass[(d, a)] - base for a in ARMS if a != "bigru_baseline:-"])
        im = ax.imshow(M, cmap="RdBu", vmin=-0.15, vmax=0.15, aspect="auto")
        ax.set_xticks(range(len(labs)), [CLASS_NAMES[d][l] for l in labs], rotation=60, ha="right")
        ax.set_yticks(range(len(ARMS) - 1), [SHORT[a] for a in ARMS if a != "bigru_baseline:-"] if d == "ucihar" else [""] * 7)
        ax.set_title(DNAME[d])
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{100 * M[i, j]:+.0f}", ha="center", va="center", fontsize=5.6,
                        color="white" if abs(M[i, j]) > 0.09 else "k")
        ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01)
    cb.set_label("$\\Delta$ per-class F1 vs BiGRU")
    save(fig, "fig_perclass")


fig_perclass()
NUM["perclass_bigru"] = {d: {CLASS_NAMES[d][l]: round(float(v), 3) for l, v in zip(sorted(CLASS_NAMES[d]), perclass[(d, "bigru_baseline:-")])} for d in DATASETS}
NUM["perclass_delta"] = {d: {SHORT[a]: {CLASS_NAMES[d][l]: round(float(v), 3) for l, v in zip(sorted(CLASS_NAMES[d]), perclass[(d, a)] - perclass[(d, "bigru_baseline:-")])}
                             for a in ["cnn_bigru_reference:-", "deepconvlstm:-", PROPOSED, "dual_attention_bigru:plain"]} for d in DATASETS}

# ---------------------------------------------------------------------------
# what does the spectral gate see? descriptors on the raw windows
# (UCI-HAR and WISDM only: PAMAP2 is not preprocessed on this machine)
# ---------------------------------------------------------------------------
SR = {"ucihar": 50, "wisdm": 20}
CHN = {"ucihar": ["acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z"], "wisdm": ["acc x", "acc y", "acc z"]}


def descriptors(X, sr, band=(0.3, 3.5)):
    P = np.abs(np.fft.rfft(X, axis=1)) ** 2
    tot = P.sum(1, keepdims=True) + 1e-8
    f = np.fft.rfftfreq(X.shape[1], 1 / sr)
    m = (f >= band[0]) & (f <= band[1])
    ratio = P[:, m, :].sum(1) / tot[:, 0, :]
    p = P / tot
    ent = -(p * np.log(p + 1e-8)).sum(1) / np.log(P.shape[1] + 1e-8)
    return ratio, ent


desc_stats = {}


def fig_descriptors():
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), gridspec_kw=dict(width_ratios=[6, 3]))
    for ax, d in zip(axes, ["ucihar", "wisdm"]):
        X = np.load(ROOT / "data" / "processed" / d / "X.npy").astype(np.float32)
        y = np.load(ROOT / "data" / "processed" / d / "y.npy")
        # the model sees per-channel z-scored inputs (training-fold statistics);
        # global z-scoring is a close stand-in for this descriptive figure
        Xn = (X - X.mean((0, 1))) / (X.std((0, 1)) + 1e-8)
        ratio, ent = descriptors(Xn, SR[d])
        labs = sorted(CLASS_NAMES[d])
        M = np.stack([ratio[y == l].mean(0) for l in labs])  # (classes, channels)
        im = ax.imshow(M.T, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(labs)), [CLASS_NAMES[d][l] for l in labs], rotation=45, ha="right")
        ax.set_yticks(range(len(CHN[d])), CHN[d])
        ax.set_title(DNAME[d])
        for i in range(M.shape[1]):
            for j in range(M.shape[0]):
                ax.text(j, i, f"{M[j, i]:.2f}", ha="center", va="center", fontsize=6, color="w" if M[j, i] < 0.6 else "k")
        ax.spines[:].set_visible(False)
        # how much class information is in the 2C descriptor alone? (5-fold, subject-grouped)
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        s = np.load(ROOT / "data" / "processed" / d / "subject.npy")
        Z = np.concatenate([ratio, ent], 1)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        sc = cross_val_score(clf, Z, y, groups=s, cv=GroupKFold(5), scoring="f1_macro")
        desc_stats[d] = dict(f1_descriptor_only=round(float(sc.mean()), 3), n_features=int(Z.shape[1]))
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("mean low-band (0.3–3.5 Hz) energy ratio $\\rho_c$")
    save(fig, "fig_descriptors")


fig_descriptors()
NUM["descriptor_only"] = desc_stats

# ---------------------------------------------------------------------------
# architecture and protocol diagrams
# ---------------------------------------------------------------------------
def fig_arch():
    """architecture diagrams are Graphviz sources in paper/diagrams/ (boxes size to their text)."""
    import subprocess
    for name in ["arch", "gate"]:
        src = ROOT / "paper" / "diagrams" / f"{name}.dot"
        subprocess.run(["dot", "-Tpdf", str(src), "-o", str(FIG / f"fig_{name}.pdf")], check=True)
        subprocess.run(["dot", "-Tpng", "-Gdpi=300", str(src), "-o", str(FIG / f"fig_{name}.png")], check=True)


fig_arch()


def fig_protocol():
    fig, ax = plt.subplots(figsize=(6.8, 2.5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    nS = 30
    cw = 0.85 / nS
    folds = 10
    show = [0, 1, 9]
    for r, f in enumerate(show):
        y = 0.75 - r * 0.22
        test = set(range(3 * f, 3 * f + 3))
        rest = [s for s in range(nS) if s not in test]
        val = set(np.random.default_rng(f).choice(rest, 5, replace=False))
        for s in range(nS):
            col = "#d9822b" if s in test else ("#86b7e6" if s in val else "#e4e4e4")
            ax.add_patch(plt.Rectangle((0.12 + s * cw, y), cw * 0.9, 0.12, color=col, lw=0))
        ax.text(0.11, y + 0.06, f"fold {f + 1}" if f != 9 else "fold 10", ha="right", va="center", fontsize=7.5)
        if r == 1:
            ax.text(0.12 + 0.85 / 2, y - 0.05, "⋮", ha="center", va="center", fontsize=10, family="DejaVu Sans")
    for (c, t, x0, y0) in [("#e4e4e4", "training subjects", 0.12, 0.13), ("#d9822b", "held-out test subjects", 0.42, 0.13),
                           ("#86b7e6", "validation subjects (≥2): early stopping on validation macro-F1", 0.12, 0.02)]:
        ax.add_patch(plt.Rectangle((x0, y0), 0.018, 0.06, color=c, lw=0))
        ax.text(x0 + 0.024, y0 + 0.03, t, va="center", fontsize=6.8, family="DejaVu Sans")
    ax.text(0.12, 0.92, "subjects 1 … 30 (UCI-HAR shown; WISDM: 12 folds × 3 subjects, PAMAP2: 8 folds × 1 subject)", fontsize=7.5)
    save(fig, "fig_protocol")


fig_protocol()

# ---------------------------------------------------------------------------
# dataset table numbers
# ---------------------------------------------------------------------------
ds_info = {}
for d in DATASETS:
    r = raw[(raw.dataset == d)].iloc[0]
    tot = loso[loso.dataset == d].groupby("fold_id").n_test.first().sum()
    ds_info[d] = dict(windows=int(tot), classes=int(r.num_classes))
NUM["datasets"] = ds_info

(GEN / "numbers.json").write_text(json.dumps(NUM, indent=1, default=float))
print(json.dumps(NUM, indent=1, default=float))
