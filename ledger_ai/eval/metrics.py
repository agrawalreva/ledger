"""Faithfulness, relevance, business alignment, and composite scorers."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
logger = logging.getLogger(__name__)

_PASS_THRESHOLD_COMPOSITE = 0.75
_PASS_THRESHOLD_SINGLE = 0.60

_EMBED_MODEL = None


def _embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL


def relevance_score(question: str, answer: str) -> float:
    model = _embed_model()
    q_emb = model.encode(question)
    a_emb = model.encode(answer)
    return float(cosine_similarity([q_emb], [a_emb])[0][0])


def _relevance_lexical(question: str, answer: str) -> float:
    """Lightweight overlap score when embeddings are unavailable (e.g. pipeline --mock)."""
    q = set(question.lower().split())
    a = set(answer.lower().split())
    if not q or not a:
        return 0.5
    inter = len(q & a)
    union = len(q | a)
    return float(inter) / float(union) if union else 0.0


def business_alignment_score(answer: str) -> float:
    """0–2 rubric, normalized to 0.0–1.0."""
    score = 0
    if re.search(r"\d", answer):
        score += 1
    rec_markers = (
        "recommend",
        "next step",
        "should",
        "consider",
        "suggest",
        "action",
        "follow up",
        "review",
        "monitor",
    )
    lower = answer.lower()
    if any(m in lower for m in rec_markers):
        score += 1
    return score / 2.0


def faithfulness_score_llm(
    question: str,
    reference_context: str,
    answer: str,
    mock: bool = False,
) -> dict[str, Any]:
    """LLM-as-judge faithfulness 0.0–1.0."""
    if mock:
        # Deterministic pseudo-score from hash for stable tests
        h = abs(hash((question[:80], answer[:80]))) % 1000
        score = 0.55 + (h / 1000) * 0.4
        return {
            "score": round(score, 3),
            "rationale": "mock judge",
            "unsupported_claims": [],
        }

    from langchain_groq import ChatGroq

    from ledger_ai.utils.groq_invoke import invoke_with_retry, is_rate_limit_error, truncate_for_llm

    judge_model = os.environ.get(
        "LEDGER_GROQ_JUDGE_MODEL",
        "llama-3.1-8b-instant",
    )
    llm = ChatGroq(model=judge_model, temperature=0)
    judge_prompt = """
You are an evaluation judge. Given a question, a reference context, and a model answer,
score the answer's faithfulness to the context on a scale of 0.0 to 1.0.
Faithfulness means: every factual claim in the answer is supported by the context.
Return JSON only, no preamble: {"score": float, "rationale": str, "unsupported_claims": list}
"""
    user = (
        f"Question:\n{truncate_for_llm(question, 1200)}\n\n"
        f"Reference context:\n{truncate_for_llm(reference_context, 2800)}\n\n"
        f"Model answer:\n{truncate_for_llm(answer, 2800)}"
    )
    try:
        msg = invoke_with_retry(
            llm,
            [("system", judge_prompt), ("human", user)],
            max_attempts=8,
            operation="faithfulness_judge",
        )
    except Exception as e:
        if is_rate_limit_error(e):
            logger.error("Faithfulness judge gave up after retries (rate limit): %s", e)
            return {
                "score": 0.5,
                "rationale": (
                    "Judge skipped: Groq rate limit or token budget exhausted. "
                    "Retry tomorrow, use Mock LLM for the optimizer, set LEDGER_EVAL_MAX_CASES, "
                    "or upgrade Groq tier. See README."
                ),
                "unsupported_claims": [],
            }
        raise
    text = msg.content if hasattr(msg, "content") else str(msg)
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Judge JSON parse failed; defaulting score 0.5")
        return {"score": 0.5, "rationale": text[:500], "unsupported_claims": []}
    score = float(data.get("score", 0.5))
    score = max(0.0, min(1.0, score))
    return {
        "score": score,
        "rationale": str(data.get("rationale", "")),
        "unsupported_claims": list(data.get("unsupported_claims") or []),
    }


def composite_score(faithfulness: float, relevance: float, business_alignment: float) -> float:
    return (faithfulness * 0.4) + (relevance * 0.3) + (business_alignment * 0.3)


def passed_eval(faithfulness: float, relevance: float, business_alignment: float, composite: float) -> bool:
    if composite < _PASS_THRESHOLD_COMPOSITE:
        return False
    if min(faithfulness, relevance, business_alignment) < _PASS_THRESHOLD_SINGLE:
        return False
    return True


def score_answer(
    question: str,
    reference_context: str,
    answer: str,
    mock: bool = False,
) -> dict[str, Any]:
    f = faithfulness_score_llm(question, reference_context, answer, mock=mock)
    rel = _relevance_lexical(question, answer) if mock else relevance_score(question, answer)
    biz = business_alignment_score(answer)
    comp = composite_score(f["score"], rel, biz)
    return {
        "faithfulness": f["score"],
        "relevance": rel,
        "business_alignment": biz,
        "composite": comp,
        "passed": passed_eval(f["score"], rel, biz, comp),
        "llm_judge_rationale": f["rationale"],
        "unsupported_claims": f.get("unsupported_claims", []),
    }
