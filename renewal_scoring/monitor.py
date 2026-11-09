"""Daily snapshot drift checks (run after each scoring batch, before publishing).

Three rule families, each returning human-readable alerts:

1. **Band volume** - merchant count per risk band moved by more than +/-10%
   versus the previous snapshot (pipeline bug, upstream data outage, or a real
   shift that the sales team should hear about).
2. **Large score drops** - merchants whose score fell by more than 50% *and*
   at least 20 points day-over-day (these go to the account manager directly).
3. **Rule checks** - invariants of the scoring logic, e.g. dependency
   components in [0, 1], overridden merchants capped at 15, no missing scores.
"""

from __future__ import annotations

import pandas as pd

from .score import CHURN_CAP_FACTOR

BAND_TOLERANCE = 0.10
DROP_RELATIVE = 0.50
DROP_ABSOLUTE = 20.0


def band_volume_alerts(prev: pd.DataFrame, curr: pd.DataFrame, tol: float = BAND_TOLERANCE) -> list[str]:
    a, b = prev["band"].value_counts(), curr["band"].value_counts()
    alerts = []
    for band in sorted(set(a.index) | set(b.index)):
        before, after = int(a.get(band, 0)), int(b.get(band, 0))
        if before == 0 or abs(after - before) / before > tol:
            alerts.append(f"band '{band}': {before} -> {after} merchants")
    return alerts


def score_drop_alerts(prev: pd.DataFrame, curr: pd.DataFrame) -> pd.DataFrame:
    j = prev[["merchant_id", "score"]].merge(curr[["merchant_id", "score"]], on="merchant_id", suffixes=("_prev", "_curr"))
    drop = j["score_prev"] - j["score_curr"]
    mask = (drop >= DROP_ABSOLUTE) & (drop > DROP_RELATIVE * j["score_prev"])
    return j[mask].assign(drop=drop[mask]).sort_values("drop", ascending=False)


def rule_check_alerts(curr: pd.DataFrame) -> list[str]:
    alerts = []
    for col in ["breadth", "core_consistency", "fis_dependency", "dependency", "momentum"]:
        bad = (~curr[col].between(0, 1)).sum()
        if bad:
            alerts.append(f"{col}: {bad} values outside [0, 1]")
    if curr["score"].isna().any():
        alerts.append(f"score: {int(curr['score'].isna().sum())} missing")
    over = curr[curr["churn_override"].astype(bool)]
    if (over["score"] > 100 * CHURN_CAP_FACTOR + 1e-6).any():
        alerts.append("churn override: overridden merchants above the cap")
    return alerts


def drift_report(prev: pd.DataFrame, curr: pd.DataFrame) -> dict:
    drops = score_drop_alerts(prev, curr)
    return {
        "band_volume": band_volume_alerts(prev, curr),
        "score_drops": int(len(drops)),
        "score_drop_examples": drops.head(10).to_dict("records"),
        "rule_checks": rule_check_alerts(curr),
    }
