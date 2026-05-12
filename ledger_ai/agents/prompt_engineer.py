"""Agent 1 — Prompt engineer: generate N distinct prompt variants."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid

from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are an expert prompt engineer specializing in business intelligence assistants.
Generate {n} distinct prompt variants for a merchant insights chat assistant.
Each variant must differ meaningfully in: role framing, context injection strategy,
output format, and constraint specificity.
Return a JSON array of variant objects with keys: variant_id, strategy_name, prompt_text, rationale.
Return JSON only — no preamble, no markdown fences.
"""


def run(task_description: str, n: int = 5, mock: bool = False) -> list[dict]:
    if mock:
        base = task_description[:200]
        out = []
        strategies = [
            "structured_bullets",
            "narrative_executive",
            "sql_first_grounding",
            "risk_forward",
            "minimalist_cfo",
        ]
        for i in range(n):
            vid = f"var-mock-{uuid.uuid4().hex[:6]}"
            out.append(
                {
                    "variant_id": vid,
                    "strategy_name": strategies[i % len(strategies)],
                    "prompt_text": (
                        f"Role: Ledger analyst variant {i+1}.\n"
                        f"Context: {base}\n"
                        "Format: Answer with sections Role/Findings/Next steps.\n"
                        "Constraints: Use query_snowflake only for numbers; cite rows.\n"
                        "Few-shot: If spend unknown, run SELECT SUM(amount) ... LIMIT 50."
                    ),
                    "rationale": f"Mock variant {i+1} for pipeline testing.",
                }
            )
        logger.info("[Agent1] mock mode: returned %s variants", len(out))
        return out

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
    sys = SYSTEM_PROMPT.format(n=n)
    msg = llm.invoke(
        [
            ("system", sys),
            ("human", f"Task description:\n{task_description}\n\nProduce exactly {n} variants."),
        ]
    )
    text = msg.content if hasattr(msg, "content") else str(msg)
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Agent1 expected JSON array")
    logger.info("[Agent1] generated %s variants", len(data))
    time.sleep(0.5)
    return data
