"""Agent 4 — Final optimization recommendation and stakeholder narrative."""

from __future__ import annotations

import json
import logging
import statistics
import time
from typing import Any

from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    try:
        mx = statistics.mean(xs)
        my = statistics.mean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        denx = sum((x - mx) ** 2 for x in xs) ** 0.5
        deny = sum((y - my) ** 2 for y in ys) ** 0.5
        if denx == 0 or deny == 0:
            return None
        return round(num / (denx * deny), 3)
    except statistics.StatisticsError:
        return None


def run(
    run_id: str,
    history: dict,
    best_version: dict,
    human_feedback_rows: list[dict] | None = None,
    mock: bool = False,
) -> dict[str, Any]:
    human_feedback_rows = human_feedback_rows or []
    ratings_by_version: dict[str, list[float]] = {}
    for row in human_feedback_rows:
        vid = row.get("version_id")
        if not vid:
            continue
        ratings_by_version.setdefault(vid, []).append(float(row.get("rating", 0)))

    xs: list[float] = []
    ys: list[float] = []
    for vid, rs in ratings_by_version.items():
        if vid == best_version.get("version_id"):
            xs.append(float(best_version.get("avg_composite", 0.0)))
            ys.append(statistics.mean(rs))

    corr = _correlation(xs, ys)

    perf = best_version.get("category_scores", {})
    score_hist = history.get("score_history", [])

    if mock:
        return {
            "run_id": run_id,
            "best_prompt": best_version.get("prompt_text", ""),
            "best_version_id": best_version.get("version_id"),
            "performance_summary": perf,
            "iteration_history": score_hist,
            "stakeholder_summary": (
                "Mock recommendation: prompt quality trended upward across iterations. "
                "Human–auto correlation uses a placeholder when feedback volume is low."
            ),
            "human_auto_correlation": corr if corr is not None else 0.73,
        }

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    sys = """You are a principal AI strategist writing for CFO and product leadership.
Given structured JSON about prompt optimization results, produce:
1) A concise stakeholder_summary (plain English, 120–200 words).
2) bullets for key risks and mitigations.
Return JSON only: {"stakeholder_summary": str, "risk_bullets": [str, ...]}"""
    payload = {
        "run_id": run_id,
        "best_version_id": best_version.get("version_id"),
        "category_scores": perf,
        "score_history": score_hist,
        "human_auto_correlation": corr,
        "strategies_tried": history.get("tried_strategies", []),
    }
    msg = llm.invoke([("system", sys), ("human", json.dumps(payload, indent=2))])
    text = msg.content if hasattr(msg, "content") else str(msg)
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    data = json.loads(text)
    logger.info("[Agent4] recommendation complete for run %s", run_id)
    time.sleep(0.3)
    return {
        "run_id": run_id,
        "best_prompt": best_version.get("prompt_text", ""),
        "best_version_id": best_version.get("version_id"),
        "performance_summary": perf,
        "iteration_history": score_hist,
        "stakeholder_summary": data.get("stakeholder_summary", ""),
        "risk_bullets": data.get("risk_bullets", []),
        "human_auto_correlation": corr,
    }
