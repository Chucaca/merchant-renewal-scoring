import numpy as np
import pandas as pd
import pytest

from renewal_scoring.evaluate import auc, decile_table, ks_statistic
from renewal_scoring.features import compute_fis
from renewal_scoring.score import (
    CHURN_CAP_FACTOR,
    PeerNormalizer,
    churn_override,
    current_streak,
    risk_band,
    score_merchants,
)
from renewal_scoring.synthetic import generate


@pytest.fixture(scope="module")
def small():
    data = generate(n_merchants=1500, seed=3)
    fis = compute_fis(data.feature_usage, data.merchants["renewed"].to_numpy(), data.feature_names)
    scores, _ = score_merchants(data, fis)
    return data, fis, scores


def _segments(sizes: dict[str, int]) -> pd.DataFrame:
    rows = []
    for ind, k in sizes.items():
        rows += [{"industry": ind, "city_tier": "tier1", "branch_tier": "single"}] * k
    return pd.DataFrame(rows)


def test_peer_normalization_is_bounded():
    rng = np.random.default_rng(0)
    seg = _segments({"a": 300, "b": 250})
    x = pd.DataFrame({"m": rng.normal(0, 1, len(seg)) * 10})
    norm, _ = PeerNormalizer().fit(seg, x).transform(seg, x)
    assert norm["m"].between(0, 1).all()
    # values at or below the peer P25 map to 0, at or above P80 map to 1
    a = seg["industry"] == "a"
    p25, p80 = x.loc[a, "m"].quantile([0.25, 0.80])
    assert (norm.loc[a & (x["m"] <= p25), "m"] == 0).all()
    assert (norm.loc[a & (x["m"] >= p80), "m"] == 1).all()


def test_peer_normalization_falls_back_for_small_groups():
    seg = _segments({"big": 400, "tiny": 20})
    x = pd.DataFrame({"m": np.r_[np.linspace(0, 100, 400), np.full(20, 1000.0)]})
    norm, level = PeerNormalizer(min_peer=100).fit(seg, x).transform(seg, x)
    tiny = seg["industry"] == "tiny"
    assert (level[~tiny] == 0).all()  # most specific level used
    assert (level[tiny] == 3).all()  # tiny group -> global fallback
    assert (norm.loc[tiny, "m"] == 1).all()  # compared against everyone, they are on top


def test_peer_normalization_constant_metric_does_not_blow_up():
    seg = _segments({"a": 200})
    x = pd.DataFrame({"m": np.ones(200)})
    norm, _ = PeerNormalizer().fit(seg, x).transform(seg, x)
    assert np.isfinite(norm["m"]).all() and norm["m"].between(0, 1).all()


def test_churn_override_caps_score():
    score = np.array([0.80, 0.80, 0.80])
    momentum = np.array([0.40, 0.40, 0.40])
    latest = np.array([0.05, 0.10, 0.50])
    capped, flag = churn_override(score, momentum, latest)
    assert flag.tolist() == [True, False, False]
    assert capped[0] == pytest.approx(CHURN_CAP_FACTOR * 0.40)
    assert capped[1] == 0.80 and capped[2] == 0.80


def test_churn_override_never_raises_score():
    score = np.array([0.01])
    capped, flag = churn_override(score, np.array([0.9]), np.array([0.0]))
    assert flag[0] and capped[0] == 0.01


def test_stopped_merchants_are_overridden_and_scored_low(small):
    data, _, scores = small
    stopped = (data.merchants["trajectory"] == "sudden_stop").to_numpy()
    assert stopped.sum() > 0
    assert scores.loc[stopped, "churn_override"].mean() > 0.9
    assert scores.loc[stopped, "score"].max() <= 100 * CHURN_CAP_FACTOR + 1e-9
    assert (scores.loc[scores["churn_override"], "score"] <= 15.0 + 1e-9).all()


def test_scores_in_range_and_banded(small):
    _, _, scores = small
    assert scores["score"].between(0, 100).all()
    assert scores["bas"].between(0, 100).all()
    assert set(scores["band"]) <= {"critical", "high", "medium", "low"}
    assert risk_band(np.array([0, 29.9, 30, 50, 70, 100])).tolist() == ["critical", "critical", "high", "medium", "low", "low"]


def test_score_is_predictive_on_synthetic_data(small):
    _, _, scores = small
    assert auc(scores["renewed"], scores["score"]) > 0.70


def test_fis_requires_minimum_sample():
    rng = np.random.default_rng(1)
    n = 500
    usage = np.zeros((n, 26, 2), dtype=np.int16)
    usage[:50, :, 0] = 3  # only 50 adopters -> below MIN_SAMPLE -> no signal
    usage[:300, :, 1] = 3
    y = np.r_[np.ones(250), np.zeros(250)].astype(int)
    rng.shuffle(y[300:])
    fis = compute_fis(usage, y, ["rare", "common"]).set_index("feature")
    assert fis.loc["rare", "fis"] == 0
    assert fis.loc["common", "fis"] > 0
    assert fis["weight"].sum() == pytest.approx(1.0)


def test_streak_and_metrics_helpers():
    d = np.array([[1, 1, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]])
    assert current_streak(d).tolist() == [3, 0, 6]
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    t = decile_table(np.r_[np.zeros(50), np.ones(50)], np.arange(100), n_bins=10)
    assert t.loc[0, "lift"] == pytest.approx(2.0)


def test_monitoring_drift_rules(small):
    from renewal_scoring.monitor import drift_report

    _, _, scores = small
    assert drift_report(scores, scores) == {"band_volume": [], "score_drops": 0, "score_drop_examples": [], "rule_checks": []}
    curr = scores.copy()
    curr.loc[curr.index[:400], "band"] = "critical"
    top = curr["score"].idxmax()
    curr.loc[top, "score"] = 0.0
    rep = drift_report(scores, curr)
    assert rep["band_volume"] and rep["score_drops"] >= 1
