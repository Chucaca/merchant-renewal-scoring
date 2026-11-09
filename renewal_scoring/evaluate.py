"""Evaluation: discrimination, lift, calibration, ablation and a black-box benchmark.

All metrics are computed on the *test cohort* of a time-based split (contracts
ending later in the year), with the "positive" class being *renewed* (a higher
score means healthier). For the sales team the list is read from the bottom:
decile 1 = lowest scores = highest churn risk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")  # silence a joblib warning on Windows

SPLIT_DATE = "2025-09-01"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUC-ROC via the Mann-Whitney U statistic (ties get half credit)."""
    y = np.asarray(y).astype(bool)
    r = rankdata(s)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ks_statistic(y: np.ndarray, s: np.ndarray) -> float:
    """Max distance between the score CDFs of renewed vs churned merchants."""
    y = np.asarray(y).astype(bool)
    grid = np.unique(s)
    c1 = np.searchsorted(np.sort(s[y]), grid, side="right") / y.sum()
    c0 = np.searchsorted(np.sort(s[~y]), grid, side="right") / (~y).sum()
    return float(np.max(np.abs(c1 - c0)))


def bootstrap_auc_ci(y, s, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        vals.append(auc(yy, s[idx]))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_bootstrap_auc_diff(y, s_a, s_b, n_boot: int = 500, seed: int = 1) -> tuple[float, float, float]:
    """AUC(b) - AUC(a) with a paired bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    y, s_a, s_b = map(np.asarray, (y, s_a, s_b))
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(auc(y[idx], s_b[idx]) - auc(y[idx], s_a[idx]))
    return float(auc(y, s_b) - auc(y, s_a)), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def decile_table(y, s, n_bins: int = 10) -> pd.DataFrame:
    """Decile 1 = lowest score (riskiest). Lift = churn rate / overall churn rate."""
    df = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(s)})
    df["decile"] = pd.qcut(df["s"].rank(method="first"), n_bins, labels=range(1, n_bins + 1)).astype(int)
    base_churn = 1 - df["y"].mean()
    t = df.groupby("decile").agg(merchants=("y", "size"), score_min=("s", "min"), score_max=("s", "max"), renewal_rate=("y", "mean"))
    t["churn_rate"] = 1 - t["renewal_rate"]
    t["lift"] = t["churn_rate"] / base_churn
    churners = (1 - df["y"]).sum()
    t["cum_churn_captured"] = ((1 - df["y"]).groupby(df["decile"]).sum().cumsum() / churners).to_numpy()
    return t.reset_index()


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    return df.groupby("bin").agg(mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index()


def roc_points(y, s) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(bool)
    order = np.argsort(-np.asarray(s), kind="mergesort")
    ys = y[order]
    tpr = np.concatenate([[0], np.cumsum(ys) / ys.sum()])
    fpr = np.concatenate([[0], np.cumsum(~ys) / (~ys).sum()])
    return fpr, tpr


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
def fit_gbm(X_train: pd.DataFrame, y_train: np.ndarray, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    cat = [c for c in ["industry", "city_tier", "branch_tier", "contract_type"] if c in X_train]
    model = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        categorical_features=[X_train.columns.get_loc(c) for c in cat],
        random_state=seed,
    )
    return model.fit(X_train, y_train)


ABLATION = [
    ("score", "Final score (0.45 Dep + 0.55 Mom, with override)"),
    ("score_uncapped", "Score without churn override"),
    ("dependency", "Dependency alone"),
    ("breadth", "  breadth"),
    ("core_consistency", "  core daily consistency"),
    ("fis_dependency", "  FIS-weighted feature dependency"),
    ("momentum", "Momentum alone"),
    ("current_engagement", "  current engagement"),
    ("trend", "  recency-weighted trend"),
    ("bas", "Business Activity Score (BAS)"),
]


@dataclass
class EvaluationResult:
    test: pd.DataFrame  # scored test cohort with gbm_proba
    metrics: dict
    ablation: pd.DataFrame
    deciles: pd.DataFrame
    gbm_deciles: pd.DataFrame
    calibration_score: pd.DataFrame
    calibration_gbm: pd.DataFrame
    band_table: pd.DataFrame
    trajectory_table: pd.DataFrame


def evaluate(scores: pd.DataFrame, X: pd.DataFrame, merchants: pd.DataFrame, split_date: str = SPLIT_DATE) -> EvaluationResult:
    is_train = (pd.to_datetime(merchants["contract_end"]) < pd.Timestamp(split_date)).to_numpy()
    y = merchants["renewed"].to_numpy()

    gbm = fit_gbm(X[is_train], y[is_train])
    proba = gbm.predict_proba(X[~is_train])[:, 1]

    test = scores[~is_train].copy()
    test["gbm_proba"] = proba
    test["trajectory"] = merchants.loc[~is_train, "trajectory"].to_numpy()
    yt = test["renewed"].to_numpy()

    rows = []
    for col, label in ABLATION + [("gbm_proba", "Black-box GBM (HistGradientBoosting)")]:
        rows.append({"component": col, "description": label, "auc": auc(yt, test[col].to_numpy()), "ks": ks_statistic(yt, test[col].to_numpy())})
    ablation = pd.DataFrame(rows)

    s = test["score"].to_numpy()
    ci = bootstrap_auc_ci(yt, s)
    ci_gbm = bootstrap_auc_ci(yt, proba, seed=2)
    diff = paired_bootstrap_auc_diff(yt, s, proba)
    deciles = decile_table(yt, s)
    gbm_deciles = decile_table(yt, proba)

    band_order = ["critical", "high", "medium", "low"]
    band_table = (
        test.groupby("band")
        .agg(merchants=("renewed", "size"), renewal_rate=("renewed", "mean"), mean_score=("score", "mean"))
        .reindex(band_order)
        .reset_index()
    )
    band_table["share"] = band_table["merchants"] / band_table["merchants"].sum()
    traj = (
        test.groupby("trajectory")
        .agg(merchants=("renewed", "size"), renewal_rate=("renewed", "mean"), mean_score=("score", "mean"),
             share_critical_or_high=("band", lambda b: b.isin(["critical", "high"]).mean()))
        .reset_index()
    )

    metrics = {
        "n_train": int(is_train.sum()),
        "n_test": int((~is_train).sum()),
        "split_date": split_date,
        "renewal_rate_test": float(yt.mean()),
        "auc_score": auc(yt, s),
        "auc_score_ci95": ci,
        "ks_score": ks_statistic(yt, s),
        "auc_gbm": auc(yt, proba),
        "auc_gbm_ci95": ci_gbm,
        "ks_gbm": ks_statistic(yt, proba),
        "auc_gap_gbm_minus_score": diff[0],
        "auc_gap_ci95": (diff[1], diff[2]),
        "top_decile_lift_score": float(deciles.loc[0, "lift"]),
        "top_decile_lift_gbm": float(gbm_deciles.loc[0, "lift"]),
        "top2_decile_churn_captured_score": float(deciles.loc[1, "cum_churn_captured"]),
        "top2_decile_churn_captured_gbm": float(gbm_deciles.loc[1, "cum_churn_captured"]),
        "churn_override_rate": float(test["churn_override"].mean()),
        "churn_override_renewal_rate": float(test.loc[test["churn_override"], "renewed"].mean()),
        "gbm_iterations": int(gbm.n_iter_),
        "gbm_n_features": int(X.shape[1]),
    }
    return EvaluationResult(
        test=test,
        metrics=metrics,
        ablation=ablation,
        deciles=deciles,
        gbm_deciles=gbm_deciles,
        calibration_score=calibration_table(yt, s / 100),
        calibration_gbm=calibration_table(yt, proba),
        band_table=band_table,
        trajectory_table=traj,
    )


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)


def make_charts(res: EvaluationResult, out_dir) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t = res.test
    y = t["renewed"].to_numpy()
    files = []
    note = "Synthetic data - test cohort (time-based split)"

    # 1. ROC curves
    fig, ax = plt.subplots(figsize=(6.4, 5.6), facecolor=SURFACE)
    curves = [
        ("gbm_proba", "Black-box GBM", ORANGE, "-"),
        ("score", "Interpretable score", BLUE, "-"),
        ("momentum", "Momentum only", AQUA, "--"),
        ("dependency", "Dependency only", VIOLET, ":"),
    ]
    for col, name, color, ls in curves:
        fpr, tpr = roc_points(y, t[col].to_numpy())
        ax.plot(fpr, tpr, color=color, lw=2, ls=ls, label=f"{name}  (AUC {auc(y, t[col].to_numpy()):.3f})")
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=(0, (2, 3)))
    _style(ax, "ROC: interpretable score vs black-box vs components", "False positive rate", "True positive rate (renewed)")
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK)
    fig.text(0.01, 0.01, note, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = out / "roc_curves.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    files.append(str(p))

    # 2. Decile lift bars
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)
    x = res.deciles["decile"].to_numpy()
    wdt = 0.4
    ax.bar(x - wdt / 2 - 0.01, res.deciles["lift"], width=wdt, color=BLUE, label="Interpretable score")
    ax.bar(x + wdt / 2 + 0.01, res.gbm_deciles["lift"], width=wdt, color=ORANGE, label="Black-box GBM")
    ax.axhline(1.0, color=INK2, lw=1, ls=(0, (2, 3)))
    for xi, v in zip(x[:2], res.deciles["lift"].to_numpy()[:2]):
        ax.text(xi - wdt / 2, v + 0.03, f"{v:.2f}x", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x)
    _style(ax, "Churn lift by score decile (1 = riskiest 10%)", "Score decile", "Churn rate / average churn rate")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.text(0.01, 0.01, note, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = out / "decile_lift.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    files.append(str(p))

    # 3. Score distribution by label
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)
    bins = np.linspace(0, 100, 41)
    ax.hist(t.loc[t["renewed"] == 1, "score"], bins=bins, color=BLUE, alpha=0.65, label="Renewed")
    ax.hist(t.loc[t["renewed"] == 0, "score"], bins=bins, color=ORANGE, alpha=0.65, label="Did not renew")
    for b in (30, 50, 70):
        ax.axvline(b, color=INK2, lw=1, ls=(0, (2, 3)))
    for xpos, name in [(15, "critical"), (40, "high"), (60, "medium"), (85, "low")]:
        ax.text(xpos, ax.get_ylim()[1] * 0.95, name, ha="center", fontsize=8, color=INK2)
    _style(ax, "Score distribution by renewal outcome", "Score (0-100)", "Merchants")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, 0.9))
    fig.text(0.01, 0.01, note, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = out / "score_distribution.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    files.append(str(p))

    # 4. Calibration
    fig, ax = plt.subplots(figsize=(5.6, 5.0), facecolor=SURFACE)
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=(0, (2, 3)))
    ax.plot(res.calibration_score["mean_pred"], res.calibration_score["observed"], "o-", color=BLUE, lw=2, ms=6, label="Score / 100")
    ax.plot(res.calibration_gbm["mean_pred"], res.calibration_gbm["observed"], "s-", color=ORANGE, lw=2, ms=6, label="GBM probability")
    _style(ax, "Calibration (deciles)", "Mean predicted / score", "Observed renewal rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.text(0.01, 0.01, note, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = out / "calibration.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    files.append(str(p))

    # 5. Ablation bars
    fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor=SURFACE)
    ab = res.ablation.iloc[::-1]
    colors = [ORANGE if c == "gbm_proba" else BLUE if c in ("score", "score_uncapped") else AQUA if c in ("momentum", "current_engagement", "trend") else VIOLET if c in ("dependency", "breadth", "core_consistency", "fis_dependency") else YELLOW for c in ab["component"]]
    ax.barh(range(len(ab)), ab["auc"] - 0.5, left=0.5, color=colors, height=0.7)
    ax.set_yticks(range(len(ab)))
    ax.set_yticklabels([d.strip() for d in ab["description"]], fontsize=8)
    for i, v in enumerate(ab["auc"]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax.set_xlim(0.5, max(ab["auc"]) + 0.05)
    _style(ax, "Ablation: AUC of each component on its own", "AUC-ROC (0.5 = random)", "")
    fig.text(0.01, 0.01, note, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = out / "ablation.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    files.append(str(p))
    return files
