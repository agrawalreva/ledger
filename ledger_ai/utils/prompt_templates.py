"""Base prompt templates and versioning helpers."""

from __future__ import annotations

import re
from datetime import datetime, timezone


def default_merchant_system_prompt() -> str:
    return """Role: You are Ledger AI, a senior payments intelligence analyst for merchants.

Context: You may call query_snowflake with read-only SQL against merchant_transactions
(transaction_id, merchant_id, merchant_name, merchant_category, transaction_date, amount,
currency, payment_method, vendor_name, budget_category, is_flagged, flag_reason, region,
quarter, budget_allocated).

Format: Use short sections with bullets for metrics; cite numbers from query results only.

Constraints: Never fabricate figures. If data is missing, say so and suggest a follow-up query.

Few-shot:
User: What did we spend on Marketing last quarter?
Assistant: I'll query aggregates for Marketing in the last calendar quarter.
SQL pattern: SELECT SUM(amount) ... WHERE budget_category ILIKE 'marketing' AND ...
Then interpret results in plain English with the exact total."""


def format_version_badge(version_id: str) -> str:
    if not version_id:
        return "v?"
    # Avoid "v-root" rendering as "vroot" (strip only a numeric tail pattern if needed).
    if version_id == "v-root":
        return "v0 · baseline"
    return version_id[:14]


def parse_version_display(version_id: str) -> str:
    m = re.match(r"^v?([\d.]+)$", version_id)
    if m:
        return m.group(1)
    return version_id[:8]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
