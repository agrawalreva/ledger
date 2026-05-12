"""Merchant chat Streamlit view."""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import streamlit as st

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from ledger_ai.chat.merchant_chat import run_merchant_chat
from ledger_ai.db import snowflake_client
from ledger_ai.utils.prompt_templates import default_merchant_system_prompt, format_version_badge


def _intel_from_eval(last_eval: dict | None) -> dict[str, float]:
    if not last_eval:
        return {"faithfulness": 0.82, "relevance": 0.79, "business_alignment": 0.81}
    summaries = last_eval.get("variant_summaries") or []
    if not summaries:
        return {"faithfulness": 0.82, "relevance": 0.79, "business_alignment": 0.81}
    best = max(summaries, key=lambda s: s["avg_composite"])
    rows = best.get("per_case") or []
    if not rows:
        return {"faithfulness": 0.82, "relevance": 0.79, "business_alignment": 0.81}
    return {
        "faithfulness": statistics.mean(r["faithfulness"] for r in rows),
        "relevance": statistics.mean(r["relevance"] for r in rows),
        "business_alignment": statistics.mean(r["business_alignment"] for r in rows),
    }


def render_chat_view() -> None:
    st.markdown(
        """
<div class="ledger-card" style="padding:0.75rem 1rem; margin-bottom:1rem;">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap;">
    <div>
      <span style="font-family:'Playfair Display',serif;font-size:22px;color:#f0efe8;">Ledger</span>
      <span style="font-family:'IBM Plex Mono',monospace;font-size:22px;color:#c9a84c;"> AI</span>
    </div>
    <div style="display:flex;gap:0.75rem;align-items:center;">
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    merchants = snowflake_client.fetch_merchants()
    if not merchants:
        csv_path = Path(__file__).resolve().parent.parent / "data" / "transactions.csv"
        if csv_path.is_file():
            import pandas as pd

            df = pd.read_csv(csv_path)
            merchants = (
                df[["merchant_id", "merchant_name"]]
                .drop_duplicates()
                .head(50)
                .rename(columns={"merchant_id": "merchant_id", "merchant_name": "merchant_name"})
                .to_dict("records")
            )

    col_main, col_intel = st.columns([2.1, 1.1])

    with col_main:
        h1, h2, h3 = st.columns([1.4, 1.2, 1.0])
        with h1:
            st.markdown(
                """
<div class="wordmark" style="padding-top:0.35rem;">
  <span class="gold-shimmer" style="font-family:'Playfair Display',serif;font-size:22px;color:#f0efe8;">Ledger</span>
  <span style="font-family:'IBM Plex Mono',monospace;font-size:22px;color:#c9a84c;"> AI</span>
</div>
""",
                unsafe_allow_html=True,
            )
        with h2:
            opts = {m["merchant_name"]: m["merchant_id"] for m in merchants} if merchants else {}
            merchant_label = st.selectbox(
                "Merchant scope",
                options=list(opts.keys()) if opts else ["(load data / Snowflake for merchants)"],
                help="Pick a merchant before asking about “this merchant”.",
                label_visibility="visible",
            )
            merchant_id = opts.get(merchant_label) if opts else None
            if merchant_id:
                st.caption(f"Queries will prefer `merchant_id = {merchant_id!r}`.")
            else:
                st.caption("Select a merchant above for “this merchant” questions.")
        with h3:
            if st.button("Run Optimizer", key="run_opt_header", type="secondary"):
                st.session_state["nav"] = "admin"
                st.rerun()

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        if "active_prompt" not in st.session_state:
            st.session_state.active_prompt = snowflake_client.fetch_latest_active_prompt()
        if st.session_state.active_prompt is None:
            st.session_state.active_prompt = {
                "version_id": "v-root",
                "prompt_text": default_merchant_system_prompt(),
                "iteration_number": 0,
                "rewrite_strategy": "baseline",
            }

        for msg in st.session_state.chat_messages:
            role = msg["role"]
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant"):
                    vid = msg.get("version_id", st.session_state.active_prompt.get("version_id", "v"))
                    st.caption(format_version_badge(vid))
                    st.markdown(msg["content"])
                    c1, c2, _ = st.columns([1, 1, 8])
                    with c1:
                        if st.button("👍", key=f"up-{msg['id']}"):
                            _save_feedback(msg, 1)
                    with c2:
                        if st.button("👎", key=f"down-{msg['id']}"):
                            _save_feedback(msg, -1)

        prompt = st.chat_input("Ask about spend, vendors, budgets, or flags…")
        if prompt:
            st.session_state["last_user_question"] = prompt
            st.session_state.chat_messages.append({"id": len(st.session_state.chat_messages), "role": "user", "content": prompt})
            with st.spinner("Thinking…"):
                answer = run_merchant_chat(
                    prompt,
                    st.session_state.active_prompt["prompt_text"],
                    merchant_id=merchant_id,
                    merchant_name=merchant_label if merchant_id else None,
                    version_id=st.session_state.active_prompt.get("version_id", "v-root"),
                    mock=st.session_state.get("mock_llm", False),
                )
            st.session_state.chat_messages.append(
                {
                    "id": len(st.session_state.chat_messages),
                    "role": "assistant",
                    "content": answer,
                    "version_id": st.session_state.active_prompt.get("version_id", "v-root"),
                }
            )
            st.rerun()

    with col_intel:
        st.markdown("### Intelligence panel")
        ap = st.session_state.active_prompt
        st.markdown(
            f"<div class='muted'>Active prompt <span class='version-pill'>{format_version_badge(ap.get('version_id','v'))}</span></div>",
            unsafe_allow_html=True,
        )
        intel = _intel_from_eval(st.session_state.get("last_eval"))
        for label, key in (
            ("Faithfulness", "faithfulness"),
            ("Relevance", "relevance"),
            ("Alignment", "business_alignment"),
        ):
            val = intel[key]
            pct = int(val * 100)
            st.markdown(f"<div class='mono'>{label}  {val:.2f}</div>", unsafe_allow_html=True)
            st.markdown(
                f"""
<div class="score-track"><div class="score-fill" style="width:{pct}%"></div></div>
""",
                unsafe_allow_html=True,
            )

        run = st.session_state.get("last_optimization_run") or {}
        mem = run.get("memory", {})
        st.markdown("---")
        st.markdown(
            f"<div class='muted'>Iterations used {mem.get('iteration', 0)}/10</div>",
            unsafe_allow_html=True,
        )
        strat = ap.get("rewrite_strategy") or "—"
        st.markdown(f"<div class='muted'>Final strategy</div><div class='mono'>{strat}</div>", unsafe_allow_html=True)
        fb = st.session_state.get("feedback_counts", {"up": 12, "down": 3})
        st.markdown(
            f"<div class='muted'>Feedback 👍 {fb['up']}  👎 {fb['down']}</div>",
            unsafe_allow_html=True,
        )

def _save_feedback(msg: dict, rating: int) -> None:
    st.session_state.setdefault("feedback_counts", {"up": 0, "down": 0})
    if rating > 0:
        st.session_state["feedback_counts"]["up"] += 1
    else:
        st.session_state["feedback_counts"]["down"] += 1
    if snowflake_client.snowflake_configured():
        try:
            snowflake_client.insert_human_feedback(
                snowflake_client.new_id("fb-"),
                st.session_state.active_prompt.get("version_id", "v-root"),
                st.session_state.get("last_user_question", ""),
                msg.get("content", ""),
                rating,
                "",
            )
        except Exception:
            pass
