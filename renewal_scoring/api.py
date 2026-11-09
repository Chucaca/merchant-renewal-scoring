"""FastAPI service: score lookup with reasons and a summary.

Run::

    python -m renewal_scoring.cli all          # produces data/scores.csv
    uvicorn renewal_scoring.api:app --port 8000

Endpoints: ``GET /health``, ``GET /merchants/{merchant_id}/score``, ``GET /metrics``.

The cache here is an in-process TTL dict. In production the same interface is
backed by Redis (shared across workers, TTL aligned with the daily snapshot).
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, Response

from . import __version__
from .explain import explain

DEFAULT_SCORES = Path(os.getenv("RENEWAL_SCORES_PATH", "data/scores.csv"))
CACHE_TTL_SECONDS = float(os.getenv("RENEWAL_CACHE_TTL", "300"))


class TTLCache:
    """Tiny thread-safe TTL cache (swap for Redis in production)."""

    def __init__(self, ttl: float = 300.0, max_items: int = 10_000):
        self.ttl, self.max_items = ttl, max_items
        self._data: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            if item[0] < time.monotonic():
                del self._data[key]
                return None
            return item[1]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if len(self._data) >= self.max_items:
                self._data.pop(next(iter(self._data)))
            self._data[key] = (time.monotonic() + self.ttl, value)


class Metrics:
    """Prometheus metrics if prometheus_client is installed, else a minimal text exposition."""

    def __init__(self) -> None:
        try:
            import prometheus_client as pc  # lazy optional dependency

            self._pc = pc
            self.registry = pc.CollectorRegistry()
            self.requests = pc.Counter("renewal_requests_total", "Score requests", ["status"], registry=self.registry)
            self.cache_hits = pc.Counter("renewal_cache_hits_total", "Cache hits", registry=self.registry)
            self.latency = pc.Histogram("renewal_request_seconds", "Score latency", registry=self.registry)
        except ImportError:
            self._pc = None
            self._counts: dict[str, float] = {}
            self._lock = threading.Lock()

    def inc(self, name: str, label: str = "") -> None:
        if self._pc:
            (self.requests.labels(status=label) if name == "requests" else self.cache_hits).inc()
            return
        key = f'renewal_requests_total{{status="{label}"}}' if name == "requests" else "renewal_cache_hits_total"
        with self._lock:
            self._counts[key] = self._counts.get(key, 0) + 1

    def observe(self, seconds: float) -> None:
        if self._pc:
            self.latency.observe(seconds)
            return
        with self._lock:
            self._counts["renewal_request_seconds_sum"] = self._counts.get("renewal_request_seconds_sum", 0) + seconds
            self._counts["renewal_request_seconds_count"] = self._counts.get("renewal_request_seconds_count", 0) + 1

    def render(self) -> tuple[bytes, str]:
        if self._pc:
            return self._pc.generate_latest(self.registry), self._pc.CONTENT_TYPE_LATEST
        with self._lock:
            lines = [f"{k} {v}" for k, v in sorted(self._counts.items())]
        return ("\n".join(lines) + "\n").encode(), "text/plain; version=0.0.4"


def _load_or_build_scores(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, keep_default_na=False, na_values=[""])
    # Demo fallback: build a small synthetic population on the fly.
    from .features import compute_fis
    from .score import score_merchants
    from .synthetic import generate

    data = generate(n_merchants=2_000, seed=7)
    fis = compute_fis(data.feature_usage, data.merchants["renewed"].to_numpy(), data.feature_names)
    scores, _ = score_merchants(data, fis)
    return scores


def _clean(v: Any) -> Any:
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def create_app(scores: Optional[pd.DataFrame] = None, scores_path: Path = DEFAULT_SCORES, cache_ttl: float = CACHE_TTL_SECONDS) -> FastAPI:
    app = FastAPI(title="Merchant renewal scoring", version=__version__)
    state: dict[str, Any] = {"scores": scores}
    cache = TTLCache(ttl=cache_ttl)
    metrics = Metrics()
    lock = threading.Lock()

    def table() -> pd.DataFrame:
        with lock:
            if state["scores"] is None:
                state["scores"] = _load_or_build_scores(scores_path)
            if state["scores"].index.name != "merchant_id":
                state["scores"] = state["scores"].set_index("merchant_id", drop=False)
            return state["scores"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__, "merchants_loaded": int(len(table()))}

    @app.get("/merchants/{merchant_id}/score")
    def merchant_score(merchant_id: str) -> dict:
        t0 = time.perf_counter()
        cached = cache.get(merchant_id)
        if cached is not None:
            metrics.inc("cache")
            metrics.inc("requests", "200")
            metrics.observe(time.perf_counter() - t0)
            return cached
        df = table()
        if merchant_id not in df.index:
            metrics.inc("requests", "404")
            raise HTTPException(status_code=404, detail=f"merchant {merchant_id} not found")
        row = {k: _clean(v) for k, v in df.loc[merchant_id].to_dict().items()}
        ex = explain(row)
        body = {
            "merchant_id": merchant_id,
            "score": row["score"],
            "band": row["band"],
            "churn_override": row["churn_override"],
            "components": {
                "dependency": round(row["dependency"], 4),
                "momentum": round(row["momentum"], 4),
                "breadth": round(row["breadth"], 4),
                "core_consistency": round(row["core_consistency"], 4),
                "fis_dependency": round(row["fis_dependency"], 4),
                "current_engagement": round(row["current_engagement"], 4),
                "trend": round(row["trend"], 4),
            },
            "business_activity_score": row.get("bas"),
            "reasons": ex["facts"]["reasons"],
            "strengths": ex["facts"]["strengths"],
            "summary": ex["summary"],
            "summary_source": ex["summary_source"],
        }
        cache.set(merchant_id, body)
        metrics.inc("requests", "200")
        metrics.observe(time.perf_counter() - t0)
        return body

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        payload, ctype = metrics.render()
        if ctype.startswith("text/plain"):
            return PlainTextResponse(payload.decode(), media_type=ctype)
        return Response(payload, media_type=ctype)

    return app


app = create_app()
