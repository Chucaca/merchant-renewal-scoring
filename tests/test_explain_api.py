import pandas as pd
import pytest
from fastapi.testclient import TestClient

from renewal_scoring.api import TTLCache, create_app
from renewal_scoring.explain import MerchantSummary, build_facts, check_summary, summarize
from renewal_scoring.features import compute_fis
from renewal_scoring.score import score_merchants
from renewal_scoring.synthetic import generate


@pytest.fixture(scope="module")
def scores() -> pd.DataFrame:
    data = generate(n_merchants=800, seed=11)
    fis = compute_fis(data.feature_usage, data.merchants["renewed"].to_numpy(), data.feature_names)
    s, _ = score_merchants(data, fis)
    return s


@pytest.fixture(scope="module")
def facts(scores):
    row = scores.sort_values("score").iloc[0].to_dict()
    return build_facts(row)


def test_facts_are_json_friendly(facts):
    assert 0 <= facts["score"] <= 100
    assert facts["band"] in {"critical", "high", "medium", "low"}
    assert facts["reasons"], "the riskiest merchant must have at least one reason code"


def test_guardrail_rejects_invented_numbers(facts):
    bad = MerchantSummary(headline="Risk score 87.5", why="Usage fell 64% last month.", next_action="Call them.")
    problems = check_summary(bad, facts)
    assert any("87.5" in p for p in problems) and any("64" in p for p in problems)


def test_guardrail_rejects_banned_wording(facts):
    bad = MerchantSummary(headline="This merchant will churn", why="They definitely stopped.", next_action="Call.")
    problems = check_summary(bad, facts)
    assert len(problems) >= 2


def test_guardrail_accepts_numbers_from_facts(facts):
    ok = MerchantSummary(headline=f"Score {facts['score']} of 100", why="Activity dropped.", next_action="Call the owner.")
    assert check_summary(ok, facts) == []


def test_retries_then_template_fallback(facts):
    calls = []

    def bad_llm(f, feedback=None):
        calls.append(feedback)
        return MerchantSummary(headline="Guaranteed churn, 99% sure", why="x", next_action="y")

    summary, source, rejected = summarize(facts, generator=bad_llm)
    assert source == "template_fallback"
    assert len(calls) == 3  # 1 attempt + 2 retries
    assert calls[0] is None and calls[1]  # feedback passed on retry
    assert rejected
    assert check_summary(summary, facts) == []  # the template itself passes the guardrail


def test_llm_output_accepted_after_retry(facts):
    answers = iter(
        [
            MerchantSummary(headline="Usage fell 77%", why="x", next_action="y"),
            MerchantSummary(headline=f"{facts['band'].capitalize()} risk", why="Recent activity is down.", next_action="Call the owner."),
        ]
    )
    summary, source, rejected = summarize(facts, generator=lambda f, fb=None: next(answers))
    assert source == "llm" and len(rejected) == 1


def test_offline_default_uses_template(facts, monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    _, source, _ = summarize(facts)
    assert source == "template"


def test_ttl_cache_expires():
    c = TTLCache(ttl=0.0)
    c.set("a", 1)
    assert c.get("a") is None
    c2 = TTLCache(ttl=60)
    c2.set("a", 1)
    assert c2.get("a") == 1


def test_api_smoke(scores):
    client = TestClient(create_app(scores=scores))
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

    mid = scores["merchant_id"].iloc[0]
    r = client.get(f"/merchants/{mid}/score")
    assert r.status_code == 200
    body = r.json()
    for key in ["score", "band", "components", "reasons", "summary"]:
        assert key in body
    assert set(body["summary"]) == {"headline", "why", "next_action"}
    assert 0 <= body["components"]["dependency"] <= 1

    assert client.get(f"/merchants/{mid}/score").json() == body  # served from cache
    assert client.get("/merchants/NOPE/score").status_code == 404

    m = client.get("/metrics")
    assert m.status_code == 200 and "renewal_requests_total" in m.text
