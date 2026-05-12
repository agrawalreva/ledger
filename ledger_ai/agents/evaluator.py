"""Agent 2 — Evaluate prompt variants against the full test suite."""

from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from langchain_groq import ChatGroq

_pkg_root = Path(__file__).resolve().parents[2]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from ledger_ai.eval.metrics import score_answer
from ledger_ai.eval.test_cases import TEST_CASES
from ledger_ai.utils.groq_invoke import invoke_with_retry, truncate_for_llm

logger = logging.getLogger(__name__)

_ANSWER_MODEL = os.environ.get("LEDGER_GROQ_ANSWER_MODEL", "llama-3.3-70b-versatile")


def cases_for_eval() -> list[dict]:
    """Full suite by default; set LEDGER_EVAL_MAX_CASES for a shorter first Groq run."""
    raw = os.environ.get("LEDGER_EVAL_MAX_CASES", "").strip()
    if not raw:
        return list(TEST_CASES)
    try:
        n = int(raw)
    except ValueError:
        return list(TEST_CASES)
    n = max(1, min(n, len(TEST_CASES)))
    if n < len(TEST_CASES):
        logger.warning("Using first %s / %s eval cases (LEDGER_EVAL_MAX_CASES)", n, len(TEST_CASES))
    return list(TEST_CASES)[:n]


def _generate_model_answer(prompt_text: str, case: dict, mock: bool) -> str:
    if mock:
        ctx = case["reference_context"][:500]
        return (
            f"Based on the excerpt: {ctx} "
            f"I recommend validating figures in Snowflake. For the question about "
            f"\"{case['question'][:120]}\", use SUM(amount) grouped by quarter. "
            f"Next step: review flagged rows where is_flagged=true (approx 8% in sample)."
        )

    llm = ChatGroq(model=_ANSWER_MODEL, temperature=0.2)
    user = (
        f"Question:\n{case['question']}\n\n"
        "Reference context (ground truth excerpt for evaluation):\n"
        f"{truncate_for_llm(case['reference_context'], 4000)}\n\n"
        "Answer faithfully using only the reference context. Include at least one numeric "
        "statement when numbers exist, and end with one explicit recommendation."
    )
    msg = invoke_with_retry(
        llm,
        [("system", prompt_text), ("human", user)],
        max_attempts=8,
        operation="eval_answer",
    )
    text = msg.content if hasattr(msg, "content") else str(msg)
    time.sleep(2)
    return text


def _aggregate_category(results_for_variant: list[dict]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = {c: [] for c in ("factual", "edge_case", "adversarial", "stakeholder")}
    pass_buckets: dict[str, list[int]] = {c: [] for c in buckets}
    for r in results_for_variant:
        cat = r["test_category"]
        buckets[cat].append(float(r["composite"]))
        pass_buckets[cat].append(1 if r["passed"] else 0)
    out: dict[str, dict[str, float]] = {}
    for cat in buckets:
        scores = buckets[cat]
        passes = pass_buckets[cat]
        if not scores:
            out[cat] = {"avg_composite": 0.0, "pass_rate": 0.0}
        else:
            out[cat] = {
                "avg_composite": sum(scores) / len(scores),
                "pass_rate": sum(passes) / len(passes),
            }
    return out


def _dominant_failure(category_scores: dict[str, dict[str, float]]) -> str:
    worst = None
    worst_rate = 1.0
    for cat, row in category_scores.items():
        pr = row["pass_rate"]
        if pr < worst_rate:
            worst_rate = pr
            worst = cat
    return worst or "factual"


def evaluate_variant(
    variant: dict,
    mock: bool,
    call_counter: list[int],
) -> dict[str, Any]:
    variant_id = variant["variant_id"]
    prompt_text = variant["prompt_text"]
    per_case: list[dict] = []
    cases = cases_for_eval()
    for idx, case in enumerate(cases):
        if idx > 0 and idx % 10 == 0 and not mock:
            logger.info("[Agent2] batch pause 60s after %s cases", idx)
            time.sleep(60)
        answer = _generate_model_answer(prompt_text, case, mock=mock)
        if not mock:
            time.sleep(2)
        scored = score_answer(
            case["question"],
            case["reference_context"],
            answer,
            mock=mock,
        )
        per_case.append(
            {
                "eval_id": f"ev-{uuid.uuid4().hex[:12]}",
                "version_id": variant_id,
                "test_case_id": case["test_case_id"],
                "test_category": case["category"],
                **scored,
            }
        )
        call_counter[0] += 1
        logger.info(
            "[%s] case=%s composite=%.3f passed=%s",
            variant_id,
            case["test_case_id"],
            scored["composite"],
            scored["passed"],
        )

    cat_scores = _aggregate_category(per_case)
    dominant = _dominant_failure(cat_scores)
    overall_pass = all(row["pass_rate"] >= 0.75 for row in cat_scores.values())
    avg_all = sum(r["composite"] for r in per_case) / max(len(per_case), 1)
    return {
        "variant_id": variant_id,
        "category_scores": cat_scores,
        "dominant_failure_category": dominant,
        "overall_pass": overall_pass,
        "avg_composite": avg_all,
        "per_case": per_case,
    }


def run(variants: list[dict], mock: bool = False) -> dict[str, Any]:
    """
    Returns:
      variants: list of evaluation summaries per variant
      best_variant_id: str
      all_pass: bool if any variant overall_pass
    """
    counter = [0]
    summaries = []
    for v in variants:
        summaries.append(evaluate_variant(v, mock=mock, call_counter=counter))
    best = max(summaries, key=lambda s: s["avg_composite"])
    any_pass = any(s["overall_pass"] for s in summaries)
    return {
        "variant_summaries": summaries,
        "best_variant_id": best["variant_id"],
        "all_pass": any_pass,
        "failure_report": best,
    }
