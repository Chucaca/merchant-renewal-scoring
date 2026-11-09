"""Feature importance scores (FIS) and model matrices.

FIS answers "how much does relying on this product feature tell us about a
merchant's stickiness?". It is derived from historical renewal outcomes (the
training cohort only) and is used to weight per-feature dependency in the
score.

For each feature ``f`` three signals are computed, each in ``[0, 1]``:

* **active_signal (weight 0.40)** - how much better merchants who are *currently*
  using ``f`` (last 4 weeks) renew than the base rate:
  ``clip((rr_active - rr_base) / (1 - rr_base), 0, 1)``.
* **retention_lift (weight 0.25)** - relative renewal lift of adopters over
  non-adopters, tanh-capped so a feature cannot dominate on lift alone:
  ``tanh(max(rr_adopt / rr_non_adopt - 1, 0))``.
* **aha_signal (weight 0.35)** - renewal advantage of merchants that reached an
  "aha" habit early (used ``f`` in >= 3 of the first 8 weeks):
  ``clip((rr_aha - rr_base) / (1 - rr_base), 0, 1)``.

``FIS = 0.40 * active + 0.25 * lift + 0.35 * aha``. A signal is only computed
when every sub-sample it relies on has more than ``MIN_SAMPLE = 100``
merchants; otherwise it is 0 (we prefer "no evidence" to a noisy estimate).
Weights are then normalized to sum to 1 for use in the score.

The weights (0.40 / 0.25 / 0.35) favour *current* behaviour, then the early
habit, and only lightly the raw adopter/non-adopter lift, which is the signal
most confounded by merchant size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .synthetic import FEATURE_GROUP_OF, SyntheticData

FIS_WEIGHTS = {"active": 0.40, "lift": 0.25, "aha": 0.35}
MIN_SAMPLE = 100
RECENT_WEEKS = 8
CURRENT_WEEKS = 4
AHA_WEEKS = 8
AHA_MIN_WEEKS = 3


def _rate(y: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    k = int(mask.sum())
    return (float(y[mask].mean()) if k else float("nan")), k


def compute_fis(feature_usage: np.ndarray, renewed: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Compute FIS per feature from a (n, weeks, features) usage cube and labels."""
    y = np.asarray(renewed).astype(float)
    rr_base = y.mean()
    used_any = (feature_usage > 0).any(1)
    used_current = (feature_usage[:, -CURRENT_WEEKS:, :] > 0).any(1)
    aha = (feature_usage[:, :AHA_WEEKS, :] > 0).sum(1) >= AHA_MIN_WEEKS

    rows = []
    for j, f in enumerate(feature_names):
        rr_act, n_act = _rate(y, used_current[:, j])
        rr_ad, n_ad = _rate(y, used_any[:, j])
        rr_non, n_non = _rate(y, ~used_any[:, j])
        rr_aha, n_aha = _rate(y, aha[:, j])

        active = np.clip((rr_act - rr_base) / (1 - rr_base), 0, 1) if n_act > MIN_SAMPLE else 0.0
        lift = (
            float(np.tanh(max(rr_ad / rr_non - 1, 0.0)))
            if n_ad > MIN_SAMPLE and n_non > MIN_SAMPLE and rr_non > 0
            else 0.0
        )
        aha_s = np.clip((rr_aha - rr_base) / (1 - rr_base), 0, 1) if n_aha > MIN_SAMPLE else 0.0
        fis = FIS_WEIGHTS["active"] * active + FIS_WEIGHTS["lift"] * lift + FIS_WEIGHTS["aha"] * aha_s
        rows.append(
            {
                "feature": f,
                "group": FEATURE_GROUP_OF.get(f, "other"),
                "n_adopters": n_ad,
                "n_current": n_act,
                "n_aha": n_aha,
                "active_signal": float(active),
                "retention_lift": float(lift),
                "aha_signal": float(aha_s),
                "fis": float(fis),
            }
        )
    out = pd.DataFrame(rows)
    total = out["fis"].sum()
    out["weight"] = out["fis"] / total if total > 0 else 1.0 / len(out)
    return out.sort_values("fis", ascending=False).reset_index(drop=True)


def build_model_matrix(data: SyntheticData) -> pd.DataFrame:
    """Raw-ish feature matrix for the black-box benchmark (same inputs as the score)."""
    m = data.merchants
    n = len(m)
    d = data.daily_txn.astype(float).reshape(n, -1, 7)
    weekly_txn = d.sum(2)
    weekly_active = (d > 0).sum(2)
    fu = data.feature_usage  # int16; sums below upcast without copying the cube
    cols: dict[str, np.ndarray] = {}
    for w in range(weekly_txn.shape[1]):
        cols[f"txn_w{w:02d}"] = weekly_txn[:, w]
        cols[f"active_days_w{w:02d}"] = weekly_active[:, w]
    for j, f in enumerate(data.feature_names):
        cols[f"{f}_first8"] = fu[:, :8, j].sum(1, dtype=np.float64)
        cols[f"{f}_mid"] = fu[:, 8:-8, j].sum(1, dtype=np.float64)
        cols[f"{f}_last8"] = fu[:, -8:, j].sum(1, dtype=np.float64)
        cols[f"{f}_weeks_used"] = (fu[:, :, j] > 0).sum(1)
    X = pd.DataFrame(cols)
    for c in ["industry", "city_tier", "branch_tier", "contract_type"]:
        X[c] = pd.Categorical(m[c]).codes
    return X
