"""Reason codes and a guarded LLM summary layer.

Design rule: **code computes every fact; the LLM only writes prose.**

1. ``build_facts`` turns a scored row into a small, JSON-serializable fact sheet
   (score, band, component values, ranked reason codes, feature-level signals).
2. ``summarize`` asks an LLM (Gemini or OpenAI through ``instructor``, imported
   lazily) to fill exactly three short fields of ``MerchantSummary``.
3. ``check_summary`` rejects any output that contains a number not present in
   the facts or banned wording (over-confident or judgmental language). The
   call is retried up to 2 times with the rejection reason as feedback, then the
   deterministic template is used.

Offline (the default, ``LLM_PROVIDER`` unset or ``none``) the template is used
directly, so the pipeline and the API never depend on a network call.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from .score import CHURN_LATEST_THRESHOLD, DEP_WEIGHTS, MOM_WEIGHTS, W_DEPENDENCY, W_MOMENTUM

MAX_RETRIES = 2

BANNED_PATTERNS = [
    r"\bguarantee",
    r"\bcertain(ly)?\b",
    r"\bdefinitely\b",
    r"\bwill (churn|leave|cancel|not renew)\b",
    r"\bnever\b",
    r"\balways\b",
    r"\blazy\b",
    r"\bfraud",
    r"\bfail(ed|ing)? business\b",
    r"\bbankrupt",
    r"\bthreat",
]

COMPONENT_TEXT = {
    "breadth": ("feature breadth", "uses fewer product features than most peers", "uses many product features"),
    "core_consistency": ("core daily consistency", "trades on fewer days than most peers", "trades consistently, day after day"),
    "fis_dependency": ("dependency on high-value features", "relies little on the features that predict renewal", "relies on features that predict renewal"),
    "current_engagement": ("current engagement", "recent activity is well below its own baseline", "keeps recent activity at or above its own baseline"),
    "trend": ("activity trend", "activity has been trending down in recent weeks", "shows an upward activity trend"),
}
COMPONENT_WEIGHT = {
    **{k: W_DEPENDENCY * v for k, v in DEP_WEIGHTS.items()},
    **{k: W_MOMENTUM * v for k, v in MOM_WEIGHTS.items()},
}
ACTIONS = {
    "critical": "Call the owner this week; confirm whether the store is still trading before discussing renewal.",
    "high": "Schedule a check-in call and offer a short training session on the features they stopped using.",
    "medium": "Send a usage review and suggest one high-value feature they have not tried yet.",
    "low": "Standard renewal reminder; consider an upsell conversation.",
}


class MerchantSummary(BaseModel):
    """The only thing the LLM is allowed to write."""

    headline: str = Field(..., max_length=120, description="One-line status for the account manager.")
    why: str = Field(..., max_length=320, description="Two sentences explaining the main risk drivers, using only the facts.")
    next_action: str = Field(..., max_length=200, description="One concrete next step for the renewal team.")


# --------------------------------------------------------------------------- #
# Facts and reason codes
# --------------------------------------------------------------------------- #
def _split(v: Any) -> list[str]:
    if not isinstance(v, str) or not v:
        return []
    return [x for x in v.split(";") if x]


def reason_codes(row: dict | Any, top_k: int = 3) -> list[dict]:
    """Rank components by weighted shortfall ``weight * (1 - value)``."""
    out = []
    if bool(row["churn_override"]):
        out.append({"code": "CHURN_OVERRIDE", "component": "latest_engagement", "text": "activity in the latest week is close to zero, so the score is capped"})
    shortfalls = sorted(((COMPONENT_WEIGHT[k] * (1 - float(row[k])), k) for k in COMPONENT_WEIGHT), reverse=True)
    for gap, k in shortfalls:
        if len(out) >= top_k or gap < 0.02:
            break
        out.append({"code": f"LOW_{k.upper()}", "component": k, "text": COMPONENT_TEXT[k][1]})
    dropped = _split(row.get("dropped_features"))
    if dropped:
        out.append({"code": "FEATURE_DROP", "component": "features", "text": "stopped or sharply reduced use of " + ", ".join(dropped)})
    return out


def strengths(row: dict | Any, top_k: int = 2) -> list[dict]:
    ranked = sorted(((COMPONENT_WEIGHT[k] * float(row[k]), k) for k in COMPONENT_WEIGHT), reverse=True)
    return [{"component": k, "text": COMPONENT_TEXT[k][2]} for c, k in ranked[:top_k] if float(row[k]) >= 0.6]


def build_facts(row: dict | Any) -> dict:
    pct = lambda v: int(round(100 * float(v)))  # noqa: E731
    return {
        "merchant_id": str(row["merchant_id"]),
        "score": int(round(float(row["score"]))),
        "score_scale_max": 100,
        "band": str(row["band"]),
        "dependency_pct": pct(row["dependency"]),
        "momentum_pct": pct(row["momentum"]),
        "components_pct": {k: pct(row[k]) for k in COMPONENT_WEIGHT},
        "churn_override": bool(row["churn_override"]),
        "reasons": reason_codes(row),
        "strengths": strengths(row),
        "supporting_features": _split(row.get("supporting_features")),
        "unused_high_value_features": _split(row.get("unused_high_value_features")),
        "suggested_action": ACTIONS[str(row["band"])],
    }


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def allowed_numbers(facts: dict) -> set[str]:
    nums = set(_NUM.findall(json.dumps(facts)))
    return {n.replace(",", ".") for n in nums}


def check_summary(summary: MerchantSummary, facts: dict) -> list[str]:
    """Return a list of violations (empty = accepted)."""
    problems = []
    allowed = allowed_numbers(facts)
    for field_name, text in summary.model_dump().items():
        for n in _NUM.findall(text):
            if n.replace(",", ".") not in allowed:
                problems.append(f"{field_name}: number '{n}' is not in the facts")
        for pat in BANNED_PATTERNS:
            if re.search(pat, text, flags=re.IGNORECASE):
                problems.append(f"{field_name}: banned wording /{pat}/")
    return problems


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #
def template_summary(facts: dict) -> MerchantSummary:
    reasons = [r["text"] for r in facts["reasons"]][:2]
    strengths_ = [s["text"] for s in facts["strengths"]][:1]
    band = facts["band"]
    headline = f"{band.capitalize()} renewal risk: score {facts['score']} of 100"
    if reasons:
        lead = "Watch points: " if band == "low" else "Main drivers: "
        why = lead + "; ".join(reasons) + "."
    else:
        why = "No major risk driver stands out."
    if strengths_:
        why += " On the positive side, it " + strengths_[0] + "."
    return MerchantSummary(headline=headline[:120], why=why[:320], next_action=facts["suggested_action"][:200])


PROMPT = """You write short renewal-risk notes for a sales team at a POS software company.
Use ONLY the facts below. Do not introduce any number that is not in the facts.
Do not predict outcomes with certainty and do not judge the merchant.
Fill three fields: headline (<=120 chars), why (<=2 sentences), next_action (1 concrete step).

FACTS:
{facts}
"""


def llm_generator_from_env() -> Optional[Callable[[dict, Optional[str]], MerchantSummary]]:
    """Build an LLM generator from env vars, or None if unavailable.

    ``LLM_PROVIDER`` = ``gemini`` | ``openai`` | ``none`` (default),
    ``LLM_MODEL`` optionally overrides the model name.
    """
    provider = os.getenv("LLM_PROVIDER", "none").lower()
    if provider not in {"gemini", "openai"}:
        return None
    try:
        import instructor  # lazy: optional dependency
    except ImportError:
        return None
    model = os.getenv("LLM_MODEL") or ("gemini-2.0-flash" if provider == "gemini" else "gpt-4o-mini")
    prefix = "google" if provider == "gemini" else "openai"
    try:
        client = instructor.from_provider(f"{prefix}/{model}")
    except Exception:  # missing SDK or credentials
        return None

    def generate(facts: dict, feedback: Optional[str] = None) -> MerchantSummary:
        content = PROMPT.format(facts=json.dumps(facts, indent=1))
        if feedback:
            content += f"\nYour previous answer was rejected: {feedback}. Fix it."
        return client.chat.completions.create(
            response_model=MerchantSummary,
            messages=[{"role": "user", "content": content}],
            max_retries=1,
        )

    return generate


def summarize(
    facts: dict,
    generator: Optional[Callable[[dict, Optional[str]], MerchantSummary]] = None,
    use_env: bool = True,
) -> tuple[MerchantSummary, str, list[str]]:
    """Return (summary, source, rejected_reasons).

    ``source`` is ``"llm"``, ``"template"`` (no LLM configured) or
    ``"template_fallback"`` (LLM output rejected 1 + MAX_RETRIES times or errored).
    """
    if generator is None and use_env:
        generator = llm_generator_from_env()
    if generator is None:
        return template_summary(facts), "template", []
    rejected: list[str] = []
    feedback = None
    for _ in range(1 + MAX_RETRIES):
        try:
            candidate = generator(facts, feedback)
            if not isinstance(candidate, MerchantSummary):
                candidate = MerchantSummary.model_validate(candidate)
        except Exception as exc:  # network / validation errors count as a failed attempt
            rejected.append(f"error: {type(exc).__name__}")
            feedback = "the answer could not be parsed"
            continue
        problems = check_summary(candidate, facts)
        if not problems:
            return candidate, "llm", rejected
        rejected.extend(problems)
        feedback = "; ".join(problems)
    return template_summary(facts), "template_fallback", rejected


def explain(row: dict | Any, generator=None, use_env: bool = True) -> dict:
    facts = build_facts(row)
    summary, source, rejected = summarize(facts, generator=generator, use_env=use_env)
    return {"facts": facts, "summary": summary.model_dump(), "summary_source": source, "guardrail_rejections": rejected}


__all__ = [
    "MerchantSummary",
    "build_facts",
    "check_summary",
    "explain",
    "reason_codes",
    "summarize",
    "template_summary",
    "CHURN_LATEST_THRESHOLD",
]
