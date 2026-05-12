"""Groq invoke helpers: rate-limit retries and safe truncation."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Sequence

logger = logging.getLogger(__name__)


def truncate_for_llm(text: str, max_chars: int = 2800) -> str:
    if not text or len(text) <= max_chars:
        return text
    return text[: max_chars - 30] + "\n...[truncated]"


def retry_after_seconds(exc: BaseException) -> float | None:
    """Parse Groq's human-readable retry hint when present."""
    s = str(exc).lower()
    m = re.search(r"try again in (\d+)m([\d.]+)s", s)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m = re.search(r"try again in ([\d.]+)s", s)
    if m:
        return float(m.group(1))
    return None


def is_rate_limit_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    if "RateLimit" in name:
        return True
    low = str(exc).lower()
    return "429" in str(exc) or "rate_limit" in low or ("token" in low and "limit" in low)


def invoke_with_retry(
    llm: Any,
    messages: Sequence[Any],
    *,
    max_attempts: int = 8,
    operation: str = "groq",
) -> Any:
    """Invoke LangChain chat model; on Groq-style rate limits sleep and retry."""
    last: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return llm.invoke(messages)
        except Exception as e:
            last = e
            if not is_rate_limit_error(e):
                raise
            wait = retry_after_seconds(e)
            if wait is None:
                wait = min(120.0, 5.0 * (2**attempt))
            wait = max(wait, 2.0)
            logger.warning(
                "[%s] rate limited attempt %s/%s sleeping %.1fs: %s",
                operation,
                attempt + 1,
                max_attempts,
                wait,
                e,
            )
            time.sleep(wait)
    assert last is not None
    raise last
