"""Streamlit entry point — Ledger AI."""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

from ledger_ai.views.admin_view import render_admin_view
from ledger_ai.views.chat_view import render_chat_view

st.set_page_config(page_title="Ledger AI", layout="wide", initial_sidebar_state="expanded")

GLOBAL_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;600&family=Playfair+Display:wght@500;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg-primary: #0d0f14;
  --bg-secondary: #131620;
  --bg-card: #191d2a;
  --bg-card-hover: #1f2436;
  --accent-gold: #c9a84c;
  --accent-gold-muted: #8a6f32;
  --accent-gold-bright: #e8c96a;
  --accent-teal: #2dd4bf;
  --accent-red: #f87171;
  --text-primary: #f0efe8;
  --text-secondary: #9a9aaa;
  --text-muted: #555568;
  --border: #252840;
  --border-bright: #353858;
  --gold-line: #c9a84c33;
}
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg-primary) !important;
  color: var(--text-primary);
  font-family: 'Inter', system-ui, sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; }
.ledger-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-top: 1px solid rgba(201,168,76,0.3);
  border-radius: 10px;
  animation: fadein 300ms ease-out forwards;
  opacity: 0;
}
@keyframes fadein { to { opacity: 1; } }
.chat-bubble.user { background: var(--bg-card-hover); }
.chat-bubble.assistant { background: var(--bg-card-hover); }
.version-pill {
  display: inline-block;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  color: var(--accent-gold);
  border: 1px solid var(--accent-gold-muted);
  border-radius: 999px;
  padding: 0.1rem 0.45rem;
  margin-bottom: 0.35rem;
}
.score-track {
  height: 6px;
  border-radius: 999px;
  background: #252840;
  overflow: hidden;
  margin: 0.2rem 0 0.6rem 0;
}
.score-fill {
  height: 6px;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--accent-gold-muted), var(--accent-gold-bright));
  animation: growbar 700ms ease-out forwards;
  width: 0%;
}
@keyframes growbar { from { width: 0%; } }
.mono { font-family: 'IBM Plex Mono', monospace; }
.muted { color: var(--text-secondary); font-size: 0.85rem; }
hr.gold { border: none; border-top: 1px solid var(--gold-line); margin: 1rem 0; }
.wordmark span.gold-shimmer {
  background: linear-gradient(90deg, #c9a84c, #fff2b0, #c9a84c);
  background-size: 200% auto;
  animation: shimmer 2.2s ease-out 1;
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
@keyframes shimmer { 0% { background-position: 200% center; } 100% { background-position: 0 center; } }
.typing span {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--accent-gold);
  display: inline-block; margin: 0 2px;
  animation: pulse 1s infinite ease-in-out;
}
.typing span:nth-child(2) { animation-delay: 0.15s; }
.typing span:nth-child(3) { animation-delay: 0.3s; }
@keyframes pulse { 0%, 100% { opacity: 0.2; transform: translateY(0);} 50% { opacity: 1; transform: translateY(-2px);} }
.stButton button {
  border-radius: 8px !important;
  font-family: 'Playfair Display', serif !important;
}
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

if "nav" not in st.session_state:
    st.session_state.nav = "chat"
if "mock_llm" not in st.session_state:
    st.session_state.mock_llm = True
if "max_opt_iters" not in st.session_state:
    st.session_state.max_opt_iters = 3

with st.sidebar:
    st.markdown("### Navigation")
    st.session_state.nav = st.radio("View", ("Merchant Chat", "Admin Monitor"), horizontal=False)
    st.session_state.nav = "chat" if st.session_state.nav.startswith("Merchant") else "admin"
    st.session_state.mock_llm = st.toggle("Mock LLM (recommended for UI dev)", value=True)
    st.session_state.max_opt_iters = st.slider("Max optimizer iterations", 1, 10, 3)
    st.session_state.opt_task = st.text_area(
        "Optimizer task",
        value="Improve merchant spend insights chat assistant prompts.",
        height=100,
    )
    with st.expander("Tips: copy / shortcuts"):
        st.markdown(
            """
Streamlit listens for the **letter `c`** (with the app canvas focused, not inside a text box) as **Clear cache**.

- If **⌘C** seems to trigger cache actions, try **opening the app in Chrome or Safari** (not an embedded IDE browser), click **inside the chat input** once, then select text and copy.
- Or use **right‑click → Copy** on selected text.

There is **no supported config** to turn off that `c` shortcut in local dev; avoiding focus on the empty canvas helps.
"""
        )

if st.session_state.nav == "chat":
    render_chat_view()
else:
    render_admin_view()
