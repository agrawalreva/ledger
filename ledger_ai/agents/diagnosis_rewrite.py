"""Agent 3 — Diagnose failures and rewrite prompt with iteration memory."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid

from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

STRATEGY_MAP = {
    "factual": "add_grounding",
    "edge_case": "add_null_handling",
    "adversarial": "add_skepticism",
    "stakeholder": "simplify_language",
}


def run(failure_report: dict, memory: dict, best_prompt_text: str, mock: bool = False) -> dict:
    tried = memory.get("tried_strategies", [])
    dominant = failure_report.get("dominant_failure_category", "factual")
    default_strategy = STRATEGY_MAP.get(dominant, "add_grounding")

    if mock:
        strat = default_strategy
        if strat in tried:
            for s in ("add_grounding", "add_null_handling", "add_skepticism", "simplify_language"):
                if s not in tried:
                    strat = s
                    break
        rewritten = (
            best_prompt_text
            + "\n\n--- Optimizer patch ---\n"
            + f"Strategy: {strat}. Reinforce: cite only query results; handle nulls; refuse unsupported claims; "
            "use plain-language executive summary with one numbered recommendation."
        )
        out = {
            "diagnosis": f"Dominant weakness in {dominant} category under target pass rate.",
            "selected_strategy": strat,
            "rewritten_prompt": rewritten,
            "rationale": "Mock rewrite for UI and pipeline testing.",
        }
        logger.info("[Agent3] mock rewrite strategy=%s", strat)
        return out

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    tried_strategies = ", ".join(tried) if tried else "(none yet)"
    system_prompt = f"""
You are an expert prompt optimizer with memory of past iterations.
Given a structured failure report and the history of strategies already tried,
diagnose the root cause of failures and select the most targeted rewrite strategy.
Do NOT repeat strategies in: [{tried_strategies}].
Valid strategies are exactly: add_grounding, add_null_handling, add_skepticism, simplify_language.
Return JSON only, no preamble: {{"diagnosis": str, "selected_strategy": str, "rewritten_prompt": str, "rationale": str}}
"""
    human = json.dumps(
        {
            "failure_report": {
                "variant_id": failure_report.get("variant_id"),
                "category_scores": failure_report.get("category_scores"),
                "dominant_failure_category": failure_report.get("dominant_failure_category"),
                "overall_pass": failure_report.get("overall_pass"),
                "avg_composite": failure_report.get("avg_composite"),
            },
            "current_prompt": best_prompt_text,
            "memory": memory,
        },
        indent=2,
    )
    msg = llm.invoke([("system", system_prompt), ("human", human)])
    text = msg.content if hasattr(msg, "content") else str(msg)
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    strat = data.get("selected_strategy", default_strategy)
    if strat in tried:
        for candidate in ("add_grounding", "add_null_handling", "add_skepticism", "simplify_language"):
            if candidate not in tried:
                strat = candidate
                break
    logger.info("[Agent3] strategy=%s", strat)
    time.sleep(0.5)
    return {
        "diagnosis": data.get("diagnosis", ""),
        "selected_strategy": strat,
        "rewritten_prompt": data.get("rewritten_prompt", best_prompt_text),
        "rationale": data.get("rationale", ""),
    }


def update_memory(memory: dict, rewrite_result: dict, eval_summary: dict) -> None:
    memory["iteration"] = memory.get("iteration", 0) + 1
    memory.setdefault("tried_strategies", []).append(rewrite_result["selected_strategy"])
    memory.setdefault("score_history", []).append(
        {
            "avg_composite": eval_summary.get("avg_composite"),
            "variant_id": eval_summary.get("variant_id"),
        }
    )
    memory.setdefault("worst_category_history", []).append(
        eval_summary.get("dominant_failure_category", "")
    )


def new_child_variant(rewrite_result: dict, parent_version_id: str) -> dict:
    return {
        "variant_id": f"v-{uuid.uuid4().hex[:10]}",
        "strategy_name": rewrite_result["selected_strategy"],
        "prompt_text": rewrite_result["rewritten_prompt"],
        "rationale": rewrite_result.get("rationale", ""),
        "parent_version_id": parent_version_id,
    }
