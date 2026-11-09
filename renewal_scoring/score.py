"""Interpretable renewal score.

    score = 0.45 * Dependency + 0.55 * Momentum            (0..1, reported 0..100)

    Dependency = 0.22 * breadth + 0.39 * core_consistency + 0.39 * fis_dependency
                 (each peer-normalized against the peer group's P25..P80)
    Momentum   = 0.70 * current_engagement + 0.30 * trend

    Churn override: if latest-week engagement < 0.10 -> score = min(score, 0.15 * Momentum)

A separate Business Activity Score (BAS) describes the merchant's *trading*
health, independent of how much of the product it uses:

    BAS = 0.40 * continuity + 0.15 * intensity + 0.30 * trend + 0.15 * stability
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .synthetic import SyntheticData

W_DEPENDENCY, W_MOMENTUM = 0.45, 0.55
DEP_WEIGHTS = {"breadth": 0.22, "core_consistency": 0.39, "fis_dependency": 0.39}
MOM_WEIGHTS = {"current_engagement": 0.70, "trend": 0.30}
BAS_WEIGHTS = {"continuity": 0.40, "intensity": 0.15, "trend": 0.30, "stability": 0.15}
CHURN_LATEST_THRESHOLD = 0.10
CHURN_CAP_FACTOR = 0.15

DEP_WEEKS = 26  # window for dependency metrics (structural reliance: the full window)
RECENT_WEEKS = 8  # window for "unused feature" reason codes
CURRENT_WEEKS = 4  # window for current engagement
BASELINE_WEEKS = 13  # merchant's own baseline (first half of the window)
TREND_HALF_LIFE = 8.0  # weeks; recency weighting of the trend regression
TREND_SCALE = 0.02  # slope (engagement units / week) mapped through tanh
STREAK_SCALE = 10.0  # days; continuity = 1 - exp(-streak / STREAK_SCALE)

MIN_PEER_SIZE = 100
PEER_LEVELS: list[list[str]] = [
    ["industry", "city_tier", "branch_tier"],
    ["industry", "branch_tier"],
    ["industry"],
    [],  # global
]

BANDS = [  # (lower bound on 0..100 score, band)
    (70, "low"),
    (50, "medium"),
    (30, "high"),
    (0, "critical"),
]


# --------------------------------------------------------------------------- #
# Peer normalization
# --------------------------------------------------------------------------- #
@dataclass
class PeerNormalizer:
    """Normalize a metric to [0, 1] against the merchant's peer group P25..P80.

    Peer groups are tried from most to least specific (``PEER_LEVELS``); the
    first level whose group has at least ``min_peer`` merchants is used.
    """

    min_peer: int = MIN_PEER_SIZE
    lo_q: float = 0.25
    hi_q: float = 0.80
    levels: list[list[str]] = field(default_factory=lambda: [list(x) for x in PEER_LEVELS])
    stats_: dict = field(default_factory=dict)

    def fit(self, segments: pd.DataFrame, metrics: pd.DataFrame) -> "PeerNormalizer":
        self.stats_ = {}
        for col in metrics.columns:
            per_level = []
            for keys in self.levels:
                if keys:
                    g = metrics[col].groupby([segments[k] for k in keys])
                    tbl = pd.DataFrame({"n": g.size(), "lo": g.quantile(self.lo_q), "hi": g.quantile(self.hi_q)})
                else:
                    s = metrics[col]
                    tbl = pd.DataFrame({"n": [len(s)], "lo": [s.quantile(self.lo_q)], "hi": [s.quantile(self.hi_q)]})
                per_level.append(tbl)
            self.stats_[col] = per_level
        return self

    def _lookup(self, segments: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(segments)
        lo = np.full(n, np.nan)
        hi = np.full(n, np.nan)
        level = np.full(n, -1)
        for li, (keys, tbl) in enumerate(zip(self.levels, self.stats_[col])):
            todo = level < 0
            if not todo.any():
                break
            if keys:
                idx = pd.MultiIndex.from_frame(segments[keys]) if len(keys) > 1 else pd.Index(segments[keys[0]])
                t = tbl.reindex(idx)
                ok = todo & (t["n"].fillna(0).to_numpy() >= self.min_peer)
                lo[ok] = t["lo"].to_numpy()[ok]
                hi[ok] = t["hi"].to_numpy()[ok]
            else:
                ok = todo
                lo[ok] = tbl["lo"].iloc[0]
                hi[ok] = tbl["hi"].iloc[0]
            level[ok] = li
        return lo, hi, level

    def transform(self, segments: pd.DataFrame, metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        out = {}
        level = None
        for col in metrics.columns:
            lo, hi, level = self._lookup(segments, col)
            span = np.maximum(hi - lo, 1e-9)
            out[col] = np.clip((metrics[col].to_numpy() - lo) / span, 0.0, 1.0)
        return pd.DataFrame(out, index=metrics.index), pd.Series(level, index=metrics.index, name="peer_level")


def peer_percentile(values: pd.Series, segments: pd.DataFrame, min_peer: int = MIN_PEER_SIZE) -> np.ndarray:
    """Percentile rank (0..1) within the most specific peer group of size >= min_peer."""
    out = np.full(len(values), np.nan)
    for keys in PEER_LEVELS:
        todo = np.isnan(out)
        if not todo.any():
            break
        if keys:
            grp = [segments[k] for k in keys]
            size = values.groupby(grp).transform("size").to_numpy()
            pct = values.groupby(grp).rank(pct=True).to_numpy()
            ok = todo & (size >= min_peer)
        else:
            pct = values.rank(pct=True).to_numpy()
            ok = todo
        out[ok] = pct[ok]
    return out


# --------------------------------------------------------------------------- #
# Component metrics
# --------------------------------------------------------------------------- #
def weekly_engagement(daily_txn: np.ndarray) -> np.ndarray:
    """Engagement per week (0..1) relative to the merchant's own baseline.

    E_w = 0.5 * min(1, active_days_w / baseline_active_days)
        + 0.5 * min(1, txn_w / baseline_txn)
    with baselines = mean of the first ``BASELINE_WEEKS`` weeks.
    """
    n = daily_txn.shape[0]
    d = daily_txn.astype(float).reshape(n, -1, 7)
    act = (d > 0).sum(2)
    txn = d.sum(2)
    base_act = np.maximum(act[:, :BASELINE_WEEKS].mean(1, keepdims=True), 1.0)
    base_txn = np.maximum(txn[:, :BASELINE_WEEKS].mean(1, keepdims=True), 1.0)
    return 0.5 * np.minimum(act / base_act, 1.0) + 0.5 * np.minimum(txn / base_txn, 1.0)


def recency_weighted_slope(series: np.ndarray, half_life: float = TREND_HALF_LIFE) -> np.ndarray:
    """Weighted least-squares slope per row; recent weeks weigh more."""
    T = series.shape[1]
    t = np.arange(T, dtype=float)
    w = 0.5 ** ((T - 1 - t) / half_life)
    w = w / w.sum()
    t_bar = (w * t).sum()
    y_bar = (series * w).sum(1, keepdims=True)
    cov = ((t - t_bar) * (series - y_bar) * w).sum(1)
    var = (w * (t - t_bar) ** 2).sum()
    return cov / var


def trend_component(eng: np.ndarray) -> np.ndarray:
    slope = recency_weighted_slope(eng)
    return 0.5 + 0.5 * np.tanh(slope / TREND_SCALE)


def current_streak(daily_txn: np.ndarray) -> np.ndarray:
    """Number of consecutive active days ending on the snapshot day."""
    inactive = daily_txn[:, ::-1] == 0
    has_gap = inactive.any(1)
    first_gap = inactive.argmax(1)
    return np.where(has_gap, first_gap, daily_txn.shape[1]).astype(float)


def churn_override(score: np.ndarray, momentum: np.ndarray, latest: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cap the score at 0.15 * momentum when the latest engagement is < 0.10."""
    flag = latest < CHURN_LATEST_THRESHOLD
    capped = np.where(flag, np.minimum(score, CHURN_CAP_FACTOR * momentum), score)
    return capped, flag


def risk_band(score_0_100: np.ndarray | float) -> np.ndarray:
    s = np.atleast_1d(np.asarray(score_0_100, dtype=float))
    out = np.full(s.shape, "critical", dtype=object)
    for lower, name in reversed(BANDS):
        out[s >= lower] = name
    return out


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _feature_explanations(fu: np.ndarray, names: list[str], weight: np.ndarray, top_k: int = 3) -> dict[str, list[str]]:
    """Per-merchant feature-level facts used by the reason codes."""
    first = fu[:, :8, :].mean(1, dtype=np.float64)
    recent = fu[:, -CURRENT_WEEKS:, :].mean(1, dtype=np.float64)
    drop = np.where(first > 0, np.clip(1 - recent / np.maximum(first, 1e-9), 0, 1), 0.0)
    dropped_score = np.where(drop >= 0.5, drop * weight[None, :], 0.0)
    support_score = (recent > 0) * weight[None, :] * np.log1p(recent)
    top_value = np.argsort(-weight)[:10]
    unused_mask = np.zeros_like(weight, dtype=bool)
    unused_mask[top_value] = True
    unused_score = np.where((fu[:, -RECENT_WEEKS:, :].sum(1, dtype=np.int64) == 0) & unused_mask[None, :], weight[None, :], 0.0)

    def top(mat: np.ndarray) -> list[str]:
        idx = np.argsort(-mat, axis=1)[:, :top_k]
        vals = np.take_along_axis(mat, idx, 1)
        arr = np.array(names, dtype=object)
        return [";".join(arr[i][v > 0]) for i, v in zip(idx, vals)]

    return {
        "dropped_features": top(dropped_score),
        "supporting_features": top(support_score),
        "unused_high_value_features": top(unused_score),
    }


def compute_raw_metrics(data: SyntheticData, fis: pd.DataFrame) -> pd.DataFrame:
    n = len(data.merchants)
    fu = data.feature_usage
    weight = fis.set_index("feature").loc[data.feature_names, "weight"].to_numpy()
    recent_use = fu[:, -DEP_WEEKS:, :].sum(1, dtype=np.float64)
    breadth = (recent_use > 0).sum(1).astype(float)
    consistency = (data.daily_txn[:, -DEP_WEEKS * 7 :] > 0).mean(1)
    # per-feature intensity scaled by the population P80 among users of that feature
    ref = np.array(
        [np.quantile(recent_use[recent_use[:, j] > 0, j], 0.8) if (recent_use[:, j] > 0).any() else 1.0 for j in range(fu.shape[2])]
    )
    intensity = np.minimum(np.log1p(recent_use) / np.log1p(np.maximum(ref, 1.0))[None, :], 1.0)
    fis_dep = intensity @ weight

    eng = weekly_engagement(data.daily_txn)
    current = eng[:, -CURRENT_WEEKS:].mean(1)
    latest = eng[:, -1]
    trend = trend_component(eng)

    d = data.daily_txn.astype(float)
    weekly_txn = d.reshape(n, -1, 7).sum(2)[:, -RECENT_WEEKS:]
    cv = weekly_txn.std(1) / np.maximum(weekly_txn.mean(1), 1e-9)
    return pd.DataFrame(
        {
            "breadth_raw": breadth,
            "core_consistency_raw": consistency,
            "fis_dependency_raw": fis_dep,
            "current_engagement": current,
            "latest_engagement": latest,
            "trend": trend,
            "streak_days": current_streak(data.daily_txn),
            "avg_daily_txn_28d": d[:, -28:].mean(1),
            "stability": np.clip(1 - cv, 0, 1),
        },
        index=data.merchants.index,
    )


def score_merchants(
    data: SyntheticData, fis: pd.DataFrame, normalizer: PeerNormalizer | None = None
) -> tuple[pd.DataFrame, PeerNormalizer]:
    m = data.merchants
    seg = m[["industry", "city_tier", "branch_tier"]]
    raw = compute_raw_metrics(data, fis)
    dep_raw = raw[["breadth_raw", "core_consistency_raw", "fis_dependency_raw"]]
    if normalizer is None:
        normalizer = PeerNormalizer().fit(seg, dep_raw)
    dep_norm, peer_level = normalizer.transform(seg, dep_raw)
    dep_norm.columns = ["breadth", "core_consistency", "fis_dependency"]

    dependency = sum(DEP_WEIGHTS[k] * dep_norm[k] for k in DEP_WEIGHTS).to_numpy()
    momentum = (MOM_WEIGHTS["current_engagement"] * raw["current_engagement"] + MOM_WEIGHTS["trend"] * raw["trend"]).to_numpy()
    uncapped = W_DEPENDENCY * dependency + W_MOMENTUM * momentum
    final, overridden = churn_override(uncapped, momentum, raw["latest_engagement"].to_numpy())

    continuity = 1 - np.exp(-raw["streak_days"].to_numpy() / STREAK_SCALE)
    intensity_pct = peer_percentile(raw["avg_daily_txn_28d"], seg)
    bas = (
        BAS_WEIGHTS["continuity"] * continuity
        + BAS_WEIGHTS["intensity"] * intensity_pct
        + BAS_WEIGHTS["trend"] * raw["trend"].to_numpy()
        + BAS_WEIGHTS["stability"] * raw["stability"].to_numpy()
    )

    weight = fis.set_index("feature").loc[data.feature_names, "weight"].to_numpy()
    expl = _feature_explanations(data.feature_usage, data.feature_names, weight)

    out = pd.DataFrame(
        {
            "merchant_id": m["merchant_id"],
            "peer_group": m["peer_group"],
            "peer_level": peer_level.map({0: "industry|city|branch", 1: "industry|branch", 2: "industry", 3: "global"}),
            "contract_type": m["contract_type"],
            "contract_end": m["contract_end"],
            "score": np.round(100 * final, 1),
            "score_uncapped": np.round(100 * uncapped, 1),
            "churn_override": overridden,
            "dependency": dependency,
            "momentum": momentum,
            **{k: dep_norm[k].to_numpy() for k in DEP_WEIGHTS},
            "current_engagement": raw["current_engagement"].to_numpy(),
            "latest_engagement": raw["latest_engagement"].to_numpy(),
            "trend": raw["trend"].to_numpy(),
            "bas": np.round(100 * bas, 1),
            "bas_continuity": continuity,
            "bas_intensity": intensity_pct,
            "bas_stability": raw["stability"].to_numpy(),
            "streak_days": raw["streak_days"].to_numpy(),
            **expl,
        }
    )
    out["band"] = risk_band(out["score"].to_numpy())
    if "renewed" in m:
        out["renewed"] = m["renewed"].to_numpy()
    return out, normalizer
