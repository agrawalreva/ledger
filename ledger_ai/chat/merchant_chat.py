"""Merchant-facing chat agent with Snowflake tool use (local SQLite fallback for dev)."""

from __future__ import annotations

import logging
import re
import sqlite3
import sys
import uuid
from pathlib import Path

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

_pkg_root = Path(__file__).resolve().parents[2]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from ledger_ai.db import snowflake_client

logger = logging.getLogger(__name__)

_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "transactions.csv"
_sqlite_conn: sqlite3.Connection | None = None


def _local_conn() -> sqlite3.Connection:
    global _sqlite_conn
    if _sqlite_conn is None:
        _sqlite_conn = sqlite3.connect(":memory:")
        if _CSV_PATH.is_file():
            df = pd.read_csv(_CSV_PATH)
            df.to_sql("merchant_transactions", _sqlite_conn, if_exists="replace", index=False)
        else:
            pd.DataFrame(
                {
                    "transaction_id": [],
                    "merchant_id": [],
                    "merchant_name": [],
                    "amount": [],
                }
            ).to_sql("merchant_transactions", _sqlite_conn, if_exists="replace", index=False)
    return _sqlite_conn


def _run_sql(sql: str) -> str:
    if not re.match(r"^\s*select", sql, re.I):
        return "Error: only read-only SELECT statements are allowed."
    if snowflake_client.snowflake_configured():
        try:
            df = snowflake_client.execute_sql_df(sql)
            if df.empty:
                return "(no rows)"
            return df.head(200).to_string(index=False)
        except Exception as e:
            logger.exception("Snowflake query failed")
            return f"Snowflake error: {e}"
    try:
        df = pd.read_sql_query(sql, _local_conn())
        if df.empty:
            return "(no rows)"
        return df.head(200).to_string(index=False)
    except Exception as e:
        return f"Local SQL error: {e}"


@tool
def query_snowflake(sql: str) -> str:
    """Query the merchant transaction database. Use this to answer questions about
    spending, vendors, budgets, flags, and payment patterns."""
    return _run_sql(sql)


def _run_tool_agent(system_prompt: str, question: str, max_steps: int = 18) -> str:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2).bind_tools([query_snowflake])
    messages: list = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]
    for step in range(max_steps):
        if step == max_steps - 2:
            messages.append(
                SystemMessage(
                    content=(
                        "If you still need data, run at most ONE more SQL query. "
                        "Otherwise answer immediately from results already in this thread — "
                        "plain language, cite numbers from tool output only."
                    )
                )
            )
        ai: AIMessage = llm.invoke(messages)
        messages.append(ai)
        if not getattr(ai, "tool_calls", None):
            return str(ai.content or "").strip() or "(empty model reply)"
        for call in ai.tool_calls:
            if isinstance(call, dict):
                name = call.get("name")
                args = call.get("args") or {}
                tid = call.get("id") or "call"
            else:
                name = getattr(call, "name", "")
                args = getattr(call, "args", {}) or {}
                tid = getattr(call, "id", "call")
            if name == "query_snowflake":
                if isinstance(args, dict) and not args.get("sql"):
                    out = "Error: query_snowflake requires a non-empty 'sql' string argument."
                else:
                    out = query_snowflake.invoke(args if isinstance(args, dict) else {"sql": str(args)})
                logger.info("tool query_snowflake step=%s sql_preview=%s", step, str(args)[:200])
            else:
                out = f"Unknown tool {name}"
            messages.append(ToolMessage(content=str(out), tool_call_id=str(tid)))

    plain = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    messages.append(
        HumanMessage(
            content=(
                "The user is waiting for a final answer. Using ONLY the numeric results "
                "already returned from query_snowflake above, write a concise answer. "
                "If prior queries failed or are missing, say exactly what went wrong — "
                "do not invent figures. Do not request more tools."
            )
        )
    )
    final = plain.invoke(messages)
    return str(final.content or "").strip() or "Could not produce a final answer from tool results."


def run_merchant_chat(
    question: str,
    system_prompt: str,
    merchant_id: str | None = None,
    merchant_name: str | None = None,
    version_id: str = "local",
    mock: bool = False,
) -> str:
    if mock:
        ql = question.lower()
        # Still hit the real DB / local SQLite so you can verify data wiring without Groq.
        if any(k in ql for k in ("how many", "count ", "number of", "total rows")) and any(
            k in ql for k in ("trans", "row", "record", "dataset")
        ):
            raw = _run_sql("SELECT COUNT(*) AS transaction_count FROM merchant_transactions")
            merchant_note = ""
            if merchant_id:
                merchant_note = (
                    f"\n\nThe merchant dropdown is set to `{merchant_id}`, but this mock shortcut **does not** "
                    "apply that filter — the count is still over the whole table. "
                    "Turn off **Mock LLM** for merchant-scoped SQL."
                )
            return (
                "**Mock mode** — no LLM; ran read-only SQL only.\n\n"
                f"```\n{raw.strip()}\n```\n"
                f"That count is **all** rows in `merchant_transactions`."
                f"{merchant_note}\n\n"
                "Turn off **Mock LLM** in the sidebar and set `GROQ_API_KEY` for natural-language answers."
            )
        return (
            f"[mock] For merchant scope {merchant_id or 'ALL'}: "
            "Example: `SELECT COUNT(*) FROM merchant_transactions` — "
            f"turn off Mock LLM + set `GROQ_API_KEY` to answer: {question!r}"
        )

    q = question
    if merchant_id:
        scope = (
            f"\n\n[Merchant scope — use this in SQL when the user says 'this merchant' or implies one company: "
            f"merchant_id = '{merchant_id}'"
            + (f", merchant_name = '{merchant_name}'" if merchant_name else "")
            + ". Prefer filtering WHERE merchant_id = that literal.]"
        )
    else:
        scope = ""
        if any(p in question.lower() for p in ("this merchant", "this company", "for them")):
            q = (
                f"{question}\n\n[Note: No merchant is selected in the UI. "
                "Either ask the user to pick one in 'Merchant scope', or run SQL for all merchants "
                "and explain that the answer is not scoped.]"
            )
    text = _run_tool_agent(system_prompt + scope, q)
    try:
        if snowflake_client.snowflake_configured():
            snowflake_client.insert_chat_log(
                f"log-{uuid.uuid4().hex[:12]}",
                version_id,
                merchant_id or "",
                question,
                text,
            )
    except Exception as e:
        logger.warning("chat log insert skipped: %s", e)
    return text
