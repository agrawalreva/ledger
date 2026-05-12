"""Orchestrates the 4-agent optimization loop."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

_pkg_root = Path(__file__).resolve().parents[2]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

import ledger_ai.agents.diagnosis_rewrite as diagnosis_rewrite
import ledger_ai.agents.evaluator as evaluator
import ledger_ai.agents.prompt_engineer as prompt_engineer
import ledger_ai.agents.recommendation as recommendation
from ledger_ai.db import snowflake_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_run_id() -> str:
    return f"run-{uuid.uuid4().hex[:12]}"


def init_memory() -> dict:
    return {
        "iteration": 0,
        "tried_strategies": [],
        "score_history": [],
        "worst_category_history": [],
    }


def save_variants_to_snowflake(
    variants: list[dict],
    run_id: str,
    iteration: int,
    parent_map: dict[str, str | None] | None = None,
) -> None:
    parent_map = parent_map or {}
    if not snowflake_client.snowflake_configured():
        logger.info("[pipeline] Snowflake not configured; skip save_variants")
        return
    for v in variants:
        vid = v["variant_id"]
        parent = v.get("parent_version_id") or parent_map.get(vid)
        try:
            snowflake_client.insert_prompt_version(
                version_id=vid,
                parent_version_id=parent,
                prompt_text=v["prompt_text"],
                agent_task="merchant_insights",
                iteration_number=iteration,
                rewrite_strategy=v.get("strategy_name"),
            )
        except Exception as e:
            logger.warning("insert_prompt_version failed: %s", e)


def save_eval_results_to_snowflake(eval_block: dict, run_id: str, iteration: int) -> None:
    if not snowflake_client.snowflake_configured():
        logger.info("[pipeline] Snowflake not configured; skip save_eval_results")
        return
    for summary in eval_block.get("variant_summaries", []):
        for row in summary.get("per_case", []):
            try:
                snowflake_client.insert_eval_result(
                    eval_id=row["eval_id"],
                    version_id=row["version_id"],
                    test_case_id=row["test_case_id"],
                    test_category=row["test_category"],
                    faithfulness=row["faithfulness"],
                    relevance=row["relevance"],
                    business_alignment=row["business_alignment"],
                    composite=row["composite"],
                    passed=bool(row["passed"]),
                    rationale=row.get("llm_judge_rationale", ""),
                )
            except Exception as e:
                logger.warning("insert_eval_result failed: %s", e)


def all_pass(eval_results: dict) -> bool:
    return bool(eval_results.get("all_pass"))


def get_best_variant(eval_results: dict) -> dict:
    summaries = eval_results["variant_summaries"]
    best_summary = max(summaries, key=lambda s: s["avg_composite"])
    vid = best_summary["variant_id"]
    match = next((v for v in eval_results.get("_variants_snapshot", []) if v["variant_id"] == vid), None)
    prompt_text = (match or {}).get("prompt_text", "")
    return {
        "version_id": best_summary["variant_id"],
        "prompt_text": prompt_text,
        "category_scores": best_summary["category_scores"],
        "avg_composite": best_summary["avg_composite"],
        "variant_id": best_summary["variant_id"],
    }


def log_run_completion(
    run_id: str,
    reason: str,
    iteration: int,
    best_version: dict | None,
    threshold_met: bool,
) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    if not snowflake_client.snowflake_configured():
        logger.info(
            "[pipeline] run complete run_id=%s reason=%s iterations=%s best=%s",
            run_id,
            reason,
            iteration,
            (best_version or {}).get("version_id"),
        )
        return
    try:
        snowflake_client.insert_optimization_run(
            run_id=run_id,
            started_at=now,
            completed_at=now,
            total_iterations=iteration,
            final_version_id=(best_version or {}).get("version_id"),
            threshold_met=threshold_met,
            termination_reason=reason,
        )
    except Exception as e:
        logger.warning("insert_optimization_run failed: %s", e)


def fetch_human_feedback_from_snowflake(_run_id: str) -> list[dict]:
    if not snowflake_client.snowflake_configured():
        return []
    try:
        df = snowflake_client.execute_sql_df(
            "SELECT version_id, rating FROM human_feedback ORDER BY created_at DESC LIMIT 500"
        )
        if df.empty:
            return []
        return df.to_dict("records")
    except Exception:
        return []


def run_optimization_pipeline(task_description: str, max_iterations: int = 10, mock: bool = False) -> dict:
    run_id = generate_run_id()
    memory = init_memory()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    logger.info("[pipeline] starting run_id=%s mock=%s", run_id, mock)

    variants = prompt_engineer.run(task_description, n=5, mock=mock)
    save_variants_to_snowflake(variants, run_id, iteration=0)

    iteration = 0
    best_version: dict | None = None
    eval_results: dict | None = None

    while iteration < max_iterations:
        eval_results = evaluator.run(variants, mock=mock)
        eval_results["_variants_snapshot"] = [dict(v) for v in variants]
        save_eval_results_to_snowflake(eval_results, run_id, iteration)

        if all_pass(eval_results):
            best_version = get_best_variant(eval_results)
            log_run_completion(run_id, "threshold_met", iteration, best_version, True)
            break

        best_version = get_best_variant(eval_results)
        failure_report = next(
            s for s in eval_results["variant_summaries"] if s["variant_id"] == best_version["variant_id"]
        )
        rewrite_result = diagnosis_rewrite.run(
            failure_report=failure_report,
            memory=memory,
            best_prompt_text=best_version["prompt_text"],
            mock=mock,
        )
        diagnosis_rewrite.update_memory(memory, rewrite_result, failure_report)

        parent_id = best_version["version_id"]
        new_variant = diagnosis_rewrite.new_child_variant(rewrite_result, parent_id)
        variants = [new_variant]
        save_variants_to_snowflake(variants, run_id, iteration=iteration + 1, parent_map={new_variant["variant_id"]: parent_id})

        iteration += 1
        logger.info("[pipeline] iteration=%s strategy=%s", iteration, rewrite_result["selected_strategy"])
    else:
        if eval_results is None:
            eval_results = evaluator.run(variants, mock=mock)
            eval_results["_variants_snapshot"] = [dict(v) for v in variants]
        best_version = get_best_variant(eval_results)
        log_run_completion(run_id, "max_iterations", iteration, best_version, False)

    rec = recommendation.run(
        run_id=run_id,
        history=memory,
        best_version=best_version or {},
        human_feedback_rows=fetch_human_feedback_from_snowflake(run_id),
        mock=mock,
    )

    return {
        "run_id": run_id,
        "started_at": started,
        "best_version": best_version,
        "recommendation": rec,
        "memory": memory,
        "last_eval": eval_results,
        "mock_mode": mock,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ledger AI optimization pipeline")
    parser.add_argument("--mock", action="store_true", help="Skip live LLM judge pressure; synthetic scores")
    parser.add_argument("--task", type=str, default="Improve merchant spend insights chat assistant prompts.")
    parser.add_argument("--max-iterations", type=int, default=3, help="Default lowered for local smoke tests")
    args = parser.parse_args()
    out = run_optimization_pipeline(args.task, max_iterations=args.max_iterations, mock=args.mock)
    print(json.dumps({k: out[k] for k in ("run_id", "best_version", "recommendation")}, indent=2, default=str))


if __name__ == "__main__":
    main()
