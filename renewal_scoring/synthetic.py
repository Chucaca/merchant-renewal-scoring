"""Seeded synthetic data generator.

Simulates a POS SaaS with yearly contracts. Each merchant has:

* a segment (industry x city_tier x branch_tier -> peer group),
* a contract_type and a contract end date (spread over one calendar year),
* a latent ``health`` that drives both product usage and the renewal decision,
* a usage *trajectory* over the 26 weeks before the scoring snapshot
  (stable, growing, gradual decay, sudden stop, seasonal business),
* 26 weeks x 7 days of core daily activity (transactions per day),
* 26 weeks of usage counts for 40 product features grouped into 10 groups.

The renewal label is drawn from a logistic model of the latent health at the
end of the window, the hidden "aha" features, contract type and noise, with the
intercept solved so that the overall renewal rate is ~50%.

Nothing here is real data; all parameters are invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

N_WEEKS = 26
N_DAYS = N_WEEKS * 7

INDUSTRIES = {
    # name: (share, health shift, seasonal share)
    "grocery": (0.22, 0.10, 0.02),
    "fashion": (0.16, -0.10, 0.10),
    "cafe_restaurant": (0.18, -0.15, 0.08),
    "pharmacy": (0.08, 0.30, 0.01),
    "electronics": (0.08, 0.00, 0.04),
    "cosmetics": (0.10, -0.05, 0.05),
    "building_materials": (0.06, 0.15, 0.12),
    "souvenir_tourism": (0.05, -0.20, 0.55),
    "bookstore_stationery": (0.07, 0.05, 0.25),
}
CITY_TIERS = {"tier1": 0.45, "tier2": 0.35, "tier3": 0.20}
BRANCH_TIERS = {"single": 0.72, "small_chain": 0.21, "chain": 0.07}
CONTRACT_TYPES = {"1y": (0.62, 0.0), "2y": (0.25, 0.35), "3y": (0.13, 0.55)}

FEATURE_GROUPS: dict[str, list[str]] = {
    "sales": ["pos_checkout", "barcode_scan", "returns", "discount_at_till"],
    "inventory": ["stock_receive", "stock_count", "low_stock_alert", "supplier_order"],
    "customers": ["customer_profile", "customer_debt", "sms_campaign", "customer_segment"],
    "promotions": ["voucher", "combo_pricing", "price_book", "happy_hour"],
    "reports": ["daily_report", "profit_report", "inventory_report", "export_excel"],
    "staff": ["staff_accounts", "shift_handover", "commission", "permission_roles"],
    "online": ["online_store", "marketplace_sync", "social_orders", "delivery_partner"],
    "finance": ["cashbook", "e_invoice", "bank_reconcile", "tax_report"],
    "loyalty": ["points_program", "membership_tier", "gift_card", "referral"],
    "multi_branch": ["stock_transfer", "branch_report", "central_pricing", "branch_compare"],
}
FEATURES = [f for fs in FEATURE_GROUPS.values() for f in fs]
FEATURE_GROUP_OF = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}

# Features that (unknown to the scorer) genuinely raise renewal propensity once
# a merchant gets value from them. FIS should rediscover most of these.
AHA_FEATURES = {
    "stock_count": 0.30,
    "customer_debt": 0.30,
    "profit_report": 0.30,
    "e_invoice": 0.35,
    "points_program": 0.20,
    "stock_transfer": 0.25,
    "shift_handover": 0.20,
}

TRAJECTORIES = ["stable", "growth", "gradual_decay", "sudden_stop", "seasonal"]


@dataclass
class SyntheticData:
    merchants: pd.DataFrame  # one row per merchant (static + label)
    daily_txn: np.ndarray  # (n, N_DAYS) int16 transactions per day
    feature_usage: np.ndarray  # (n, N_WEEKS, n_features) int16 weekly counts
    feature_names: list[str]

    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.merchants.to_csv(folder / "merchants.csv", index=False)
        np.savez_compressed(
            folder / "usage.npz",
            daily_txn=self.daily_txn,
            feature_usage=self.feature_usage,
            feature_names=np.array(self.feature_names),
        )

    @classmethod
    def load(cls, folder: str | Path) -> "SyntheticData":
        folder = Path(folder)
        merchants = pd.read_csv(folder / "merchants.csv", parse_dates=["contract_end", "snapshot_date"])
        z = np.load(folder / "usage.npz")
        return cls(merchants, z["daily_txn"], z["feature_usage"], [str(x) for x in z["feature_names"]])


def _choice(rng: np.random.Generator, mapping: dict, n: int, idx: int | None = None) -> np.ndarray:
    keys = list(mapping)
    p = np.array([v[idx] if idx is not None else v for v in mapping.values()], dtype=float)
    return rng.choice(keys, size=n, p=p / p.sum())


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_merchants: int = 20_000, seed: int = 42, target_renewal_rate: float = 0.50) -> SyntheticData:
    rng = np.random.default_rng(seed)
    n = n_merchants

    industry = _choice(rng, INDUSTRIES, n, 0)
    city = _choice(rng, CITY_TIERS, n)
    branch = _choice(rng, BRANCH_TIERS, n)
    contract = _choice(rng, CONTRACT_TYPES, n, 0)

    ind_shift = np.array([INDUSTRIES[i][1] for i in industry])
    seasonal_share = np.array([INDUSTRIES[i][2] for i in industry])
    branch_shift = pd.Series(branch).map({"single": 0.0, "small_chain": 0.35, "chain": 0.6}).to_numpy()
    city_shift = pd.Series(city).map({"tier1": 0.1, "tier2": 0.0, "tier3": -0.1}).to_numpy()
    contract_shift = np.array([CONTRACT_TYPES[c][1] for c in contract])

    # Contract end dates spread over 2025; snapshot is taken 14 days before the end.
    end_offset = rng.integers(0, 365, size=n)
    contract_end = pd.Timestamp("2025-01-01") + pd.to_timedelta(end_offset, unit="D")
    snapshot = contract_end - pd.Timedelta(days=14)
    window_start_doy = ((snapshot - pd.Timedelta(days=N_DAYS)).dayofyear.to_numpy() - 1)

    # Latent health at the start of the observation window.
    h0 = rng.normal(0, 1, n) + ind_shift + branch_shift + city_shift

    # Trajectory choice depends on health: weak merchants decay/stop more often.
    p_decay = 0.10 + 0.14 * _sigmoid(-1.4 * h0)
    p_stop = 0.04 + 0.12 * _sigmoid(-1.6 * h0 - 0.5)
    p_growth = 0.06 + 0.08 * _sigmoid(1.2 * h0)
    p_seasonal = seasonal_share
    p_stable = np.clip(1 - p_decay - p_stop - p_growth - p_seasonal, 0.05, None)
    P = np.vstack([p_stable, p_growth, p_decay, p_stop, p_seasonal]).T
    P /= P.sum(1, keepdims=True)
    u = rng.random(n)[:, None]
    traj_idx = (u > np.cumsum(P, 1)).sum(1)
    trajectory = np.array(TRAJECTORIES)[traj_idx]

    # Weekly engagement multiplier m[w] (the "true" trajectory) and a seasonal
    # factor that changes observed usage but not underlying health.
    w = np.arange(N_WEEKS)[None, :].astype(float)
    m = 1.0 + rng.normal(0, 0.05, (n, N_WEEKS))
    true_end = np.ones(n)
    seasonal_factor = np.ones((n, N_WEEKS))

    g = trajectory == "growth"
    start = rng.uniform(0.6, 0.85, n)
    m[g] = (start[g, None] + (1.15 - start[g, None]) * w / (N_WEEKS - 1)) + rng.normal(0, 0.05, (g.sum(), N_WEEKS))
    true_end[g] = 1.1

    d = trajectory == "gradual_decay"
    d_start = rng.integers(2, 16, n)
    d_floor = rng.uniform(0.05, 0.55, n)
    frac = np.clip((w - d_start[:, None]) / np.maximum(N_WEEKS - 1 - d_start[:, None], 1), 0, 1)
    decay_curve = 1 - (1 - d_floor[:, None]) * frac
    m[d] = decay_curve[d] + rng.normal(0, 0.05, (d.sum(), N_WEEKS))
    true_end[d] = d_floor[d]

    s = trajectory == "sudden_stop"
    stop_week = rng.integers(10, N_WEEKS, n)
    stop_curve = np.where(w < stop_week[:, None], 1.0, 0.02)
    m[s] = stop_curve[s]
    true_end[s] = 0.0

    se = trajectory == "seasonal"
    peak = rng.uniform(0, 365, n)
    doy = window_start_doy[:, None] + w * 7
    seasonal_factor[se] = (0.55 + 0.75 * (0.5 + 0.5 * np.cos(2 * np.pi * (doy - peak[:, None]) / 365)))[se]
    # Recent drift for everybody except stopped merchants: a quiet change in
    # health over the last ~10 weeks (staff turnover, a new competitor, ...).
    drift = rng.normal(0, 0.35, n)
    drift[s] = 0.0
    ramp = np.clip((w - 15) / 10, 0, 1)
    m = m * np.exp(drift[:, None] * ramp)
    m = np.clip(m, 0.0, 1.5)
    observed_m = np.clip(m * seasonal_factor, 0, 2.0)

    # ---------------- core daily activity ----------------
    base_active = _sigmoid(1.4 + 0.9 * h0)  # probability of trading on a given day
    weekday = np.tile(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.05, 0.85]), N_WEEKS)
    om_daily = np.repeat(observed_m, 7, axis=1)
    p_active = np.clip(base_active[:, None] * np.minimum(om_daily, 1.15) * weekday[None, :], 0, 0.99)
    active = rng.random((n, N_DAYS)) < p_active
    lam = np.exp(2.3 + 0.35 * h0 + 0.8 * branch_shift + rng.normal(0, 0.5, n))
    txn = rng.poisson(lam[:, None] * np.clip(om_daily, 0.05, None)) * active
    daily_txn = np.minimum(txn, 32_000).astype(np.int16)

    # ---------------- feature usage ----------------
    n_f = len(FEATURES)
    f_base = rng.normal(-0.6, 0.9, n_f)  # adoption intercept per feature
    f_base[FEATURES.index("pos_checkout")] = 4.0
    f_base[FEATURES.index("barcode_scan")] = 1.5
    f_base[FEATURES.index("daily_report")] = 1.2
    f_health = rng.uniform(0.3, 1.0, n_f)
    f_rate = rng.normal(1.2, 0.6, n_f)  # log weekly use rate when adopted
    f_rate[FEATURES.index("pos_checkout")] = 3.5
    ind_codes = pd.Categorical(industry, categories=list(INDUSTRIES)).codes
    ind_aff = rng.normal(0, 0.5, (len(INDUSTRIES), n_f))
    adopt_logit = f_base[None, :] + f_health[None, :] * h0[:, None] + ind_aff[ind_codes]
    mb = np.array([FEATURE_GROUP_OF[f] == "multi_branch" for f in FEATURES])
    adopt_logit[:, mb] += np.where(branch == "single", -4.0, 1.0)[:, None]
    adopted = rng.random((n, n_f)) < _sigmoid(adopt_logit)
    # Some features are adopted part way through the window.
    adopt_week = np.where(rng.random((n, n_f)) < 0.75, 0, rng.integers(1, N_WEEKS, (n, n_f)))
    personal_rate = np.exp(f_rate[None, :] + 0.25 * h0[:, None] + rng.normal(0, 0.6, (n, n_f)))

    weekly_active_frac = active.reshape(n, N_WEEKS, 7).mean(2)
    usage = np.zeros((n, N_WEEKS, n_f), dtype=np.int16)
    for wk in range(N_WEEKS):
        on = adopted & (adopt_week <= wk)
        lam_w = personal_rate * observed_m[:, wk, None] * (0.2 + 0.8 * weekly_active_frac[:, wk, None])
        usage[:, wk, :] = np.minimum(rng.poisson(lam_w) * on, 32_000)

    # Did the merchant get real value from each aha feature (used in >= 6 weeks)?
    aha_value = np.zeros(n)
    for f, beta in AHA_FEATURES.items():
        j = FEATURES.index(f)
        aha_value += 2.2 * beta * ((usage[:, :, j] > 0).sum(1) >= 6)

    # ---------------- renewal label ----------------
    end_health = np.where(
        trajectory == "sudden_stop", -3.0, 2.0 * np.log(np.clip(true_end, 0.05, None)) + 2.6 * drift
    )
    noise = rng.normal(0, 0.4, n)
    core = 0.10 * h0 + end_health + aha_value + contract_shift + noise
    lo, hi = -10.0, 10.0
    for _ in range(60):  # solve intercept for the target renewal rate
        mid = (lo + hi) / 2
        if _sigmoid(mid + core).mean() < target_renewal_rate:
            lo = mid
        else:
            hi = mid
    p_renew = _sigmoid(mid + core)
    renewed = (rng.random(n) < p_renew).astype(int)

    merchants = pd.DataFrame(
        {
            "merchant_id": [f"M{i:06d}" for i in range(n)],
            "industry": industry,
            "city_tier": city,
            "branch_tier": branch,
            "contract_type": contract,
            "contract_end": contract_end,
            "snapshot_date": snapshot,
            "trajectory": trajectory,  # ground-truth pattern, NOT used by the scorer
            "latent_health": h0.round(4),  # ground truth, NOT used by the scorer
            "renewed": renewed,
        }
    )
    merchants["peer_group"] = merchants["industry"] + "|" + merchants["city_tier"] + "|" + merchants["branch_tier"]
    return SyntheticData(merchants, daily_txn, usage, list(FEATURES))
