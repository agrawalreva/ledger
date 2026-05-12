"""Snowflake connection and read/write helpers."""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from dotenv import load_dotenv

_root = Path(__file__).resolve().parents[2]
load_dotenv(_root / ".env")
load_dotenv(_root / "ledger_ai" / ".env")

REQUIRED_ENV = (
    "SNOWFLAKE_USER",
    "SNOWFLAKE_PASSWORD",
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_SCHEMA",
    "SNOWFLAKE_WAREHOUSE",
)


def snowflake_configured() -> bool:
    return all(os.getenv(k) for k in REQUIRED_ENV)


def _conn_params() -> dict[str, str]:
    return {
        "user": os.environ["SNOWFLAKE_USER"],
        "password": os.environ["SNOWFLAKE_PASSWORD"],
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "database": os.environ["SNOWFLAKE_DATABASE"],
        "schema": os.environ["SNOWFLAKE_SCHEMA"],
        "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
    }


@contextmanager
def get_connection():
    import snowflake.connector

    conn = snowflake.connector.connect(**_conn_params())
    try:
        yield conn
    finally:
        conn.close()


def execute_sql(sql: str, params: Sequence[Any] | None = None) -> list[tuple]:
    """Run SQL that returns rows."""
    if not snowflake_configured():
        raise RuntimeError("Snowflake environment variables are not fully set.")
    with get_connection() as conn:
        cur = conn.cursor()
        try:
            cur.execute(sql, params or [])
            if cur.description:
                return cur.fetchall()
            return []
        finally:
            cur.close()


def execute_sql_df(sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
    rows = execute_sql(sql, params)
    if not rows:
        return pd.DataFrame()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, params or [])
        cols = [c[0] for c in cur.description]
        cur.close()
    return pd.DataFrame(rows, columns=cols)


def execute_write(sql: str, params: Sequence[Any] | None = None) -> int:
    """Run DML / DDL; returns rowcount when available."""
    if not snowflake_configured():
        raise RuntimeError("Snowflake environment variables are not fully set.")
    with get_connection() as conn:
        cur = conn.cursor()
        try:
            cur.execute(sql, params or [])
            conn.commit()
            return cur.rowcount or 0
        finally:
            cur.close()


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:16]}"


def insert_prompt_version(
    version_id: str,
    parent_version_id: str | None,
    prompt_text: str,
    agent_task: str,
    iteration_number: int,
    rewrite_strategy: str | None,
) -> None:
    sql = """
        INSERT INTO prompt_versions (
            version_id, parent_version_id, prompt_text, agent_task,
            iteration_number, rewrite_strategy
        ) VALUES (%s, %s, %s, %s, %s, %s)
    """
    execute_write(
        sql,
        (
            version_id,
            parent_version_id,
            prompt_text,
            agent_task,
            iteration_number,
            rewrite_strategy,
        ),
    )


def insert_eval_result(
    eval_id: str,
    version_id: str,
    test_case_id: str,
    test_category: str,
    faithfulness: float,
    relevance: float,
    business_alignment: float,
    composite: float,
    passed: bool,
    rationale: str,
) -> None:
    sql = """
        INSERT INTO eval_results (
            eval_id, version_id, test_case_id, test_category,
            faithfulness_score, relevance_score, business_alignment_score,
            composite_score, passed, llm_judge_rationale
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    execute_write(
        sql,
        (
            eval_id,
            version_id,
            test_case_id,
            test_category,
            faithfulness,
            relevance,
            business_alignment,
            composite,
            passed,
            rationale[:16000] if rationale else "",
        ),
    )


def insert_human_feedback(
    feedback_id: str,
    version_id: str,
    question: str,
    answer: str,
    rating: int,
    feedback_note: str,
) -> None:
    sql = """
        INSERT INTO human_feedback (
            feedback_id, version_id, question, answer, rating, feedback_note
        ) VALUES (%s, %s, %s, %s, %s, %s)
    """
    execute_write(sql, (feedback_id, version_id, question, answer, rating, feedback_note))


def insert_optimization_run(
    run_id: str,
    started_at: str,
    completed_at: str | None,
    total_iterations: int,
    final_version_id: str | None,
    threshold_met: bool,
    termination_reason: str,
) -> None:
    sql = """
        INSERT INTO optimization_runs (
            run_id, started_at, completed_at, total_iterations,
            final_version_id, threshold_met, termination_reason
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    execute_write(
        sql,
        (
            run_id,
            started_at,
            completed_at,
            total_iterations,
            final_version_id,
            threshold_met,
            termination_reason,
        ),
    )


def insert_chat_log(
    log_id: str,
    version_id: str,
    merchant_id: str,
    question: str,
    answer: str,
) -> None:
    sql = """
        INSERT INTO chat_logs (log_id, version_id, merchant_id, question, answer)
        VALUES (%s, %s, %s, %s, %s)
    """
    execute_write(sql, (log_id, version_id, merchant_id, question, answer))


def load_transactions_csv(csv_path: str) -> int:
    """Bulk load CSV into merchant_transactions (replace)."""
    from snowflake.connector.pandas_tools import write_pandas

    df = pd.read_csv(csv_path)
    if not snowflake_configured():
        raise RuntimeError("Snowflake not configured.")
    with get_connection() as conn:
        cur = conn.cursor()
        try:
            cur.execute("DELETE FROM merchant_transactions")
            _, _, nrows, _ = write_pandas(conn, df, "MERCHANT_TRANSACTIONS")
            conn.commit()
            return int(nrows)
        finally:
            cur.close()


def fetch_human_feedback_correlation() -> pd.DataFrame:
    sql = """
        SELECT f.version_id,
               AVG(f.rating) AS avg_rating,
               AVG(e.composite_score) AS avg_composite
        FROM human_feedback f
        LEFT JOIN (
            SELECT version_id, AVG(composite_score) AS composite_score
            FROM eval_results
            GROUP BY version_id
        ) e ON e.version_id = f.version_id
        GROUP BY f.version_id
    """
    try:
        return execute_sql_df(sql)
    except Exception:
        return pd.DataFrame()


def fetch_prompt_lineage_scores() -> pd.DataFrame:
    sql = """
        SELECT p.version_id, p.parent_version_id, p.rewrite_strategy, p.iteration_number,
               AVG(e.composite_score) AS avg_composite,
               SUM(CASE WHEN e.passed THEN 1 ELSE 0 END)::FLOAT
                 / NULLIF(COUNT(*), 0) AS pass_rate
        FROM prompt_versions p
        LEFT JOIN eval_results e ON e.version_id = p.version_id
        GROUP BY p.version_id, p.parent_version_id, p.rewrite_strategy, p.iteration_number
    """
    try:
        return execute_sql_df(sql)
    except Exception:
        return pd.DataFrame()


def fetch_latest_active_prompt() -> dict[str, Any] | None:
    sql = """
        SELECT version_id, prompt_text, iteration_number, rewrite_strategy
        FROM prompt_versions
        ORDER BY created_at DESC
        LIMIT 1
    """
    try:
        rows = execute_sql(sql)
        if not rows:
            return None
        return {
            "version_id": rows[0][0],
            "prompt_text": rows[0][1],
            "iteration_number": rows[0][2],
            "rewrite_strategy": rows[0][3],
        }
    except Exception:
        return None


def fetch_merchants() -> list[dict[str, str]]:
    sql = """
        SELECT DISTINCT merchant_id, merchant_name
        FROM merchant_transactions
        ORDER BY merchant_name
        LIMIT 100
    """
    try:
        df = execute_sql_df(sql)
        if df.empty:
            return []
        return df.to_dict("records")
    except Exception:
        return []
