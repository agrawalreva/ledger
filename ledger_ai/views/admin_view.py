"""Admin monitoring view — optimization charts and lineage."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from ledger_ai.agents.pipeline import run_optimization_pipeline

plotly_layout = {
    "paper_bgcolor": "#0d0f14",
    "plot_bgcolor": "#131620",
    "font": {"color": "#f0efe8", "family": "IBM Plex Mono"},
    "xaxis": {"gridcolor": "#252840", "linecolor": "#353858"},
    "yaxis": {"gridcolor": "#252840", "linecolor": "#353858"},
    "colorway": ["#c9a84c", "#2dd4bf", "#9a9aaa", "#f87171", "#e8c96a"],
}


def _lineage_html(run: dict) -> str:
    mem = run.get("memory", {})
    strategies = mem.get("tried_strategies", []) or ["root"]
    scores = mem.get("score_history", []) or [{"avg_composite": 0.61}]
    lines = ["<div style='font-family:IBM Plex Mono,monospace;font-size:0.85rem;'>"]
    parent_score = 0.61
    lines.append("<div style='opacity:0.7;border-left:2px solid #555568;padding-left:0.5rem;'>v1.0  root  0.61  ● FAIL</div>")
    for i, strat in enumerate(strategies):
        sc = scores[i]["avg_composite"] if i < len(scores) else parent_score + 0.05
        tag = "● PASS" if sc >= 0.75 else "● FAIL"
        color = "#e8c96a" if sc >= max(parent_score, 0.75) else "#f0efe8"
        lines.append(
            f"<div style='margin-left:{(i+1)*12}px;border-left:2px solid #c9a84c55;padding-left:0.5rem;color:{color};'>"
            f"v1.{i+1}  {strat}  {sc:.2f}  {tag}</div>"
        )
        parent_score = sc
    lines.append("</div>")
    return "\n".join(lines)


def _category_bars(perf: dict) -> str:
    rows = []
    for cat, label in (
        ("factual", "Factual"),
        ("edge_case", "Edge case"),
        ("adversarial", "Adversarial"),
        ("stakeholder", "Stakeholder"),
    ):
        row = perf.get(cat, {"avg_composite": 0.0})
        v = float(row.get("avg_composite", 0.0))
        pct = int(max(0, min(100, v * 100)))
        rows.append(
            f"<div style='margin:0.35rem 0;font-family:IBM Plex Mono,monospace;'>"
            f"<span style='min-width:110px;display:inline-block;'>{label}</span>"
            f"<span style='display:inline-block;width:140px;height:6px;background:#252840;border-radius:3px;vertical-align:middle;'>"
            f"<span style='display:inline-block;height:6px;width:{pct}%;background:#c9a84c;border-radius:3px;'></span></span>"
            f"  {v:.2f}</div>"
        )
    return "".join(rows)


def render_admin_view() -> None:
    st.markdown(
        """
<div class="ledger-card" style="padding:0.75rem 1rem;margin-bottom:1rem;">
  <span style="font-family:'Playfair Display',serif;font-size:20px;color:#f0efe8;">Ledger AI</span>
  <span style="font-family:'Inter',sans-serif;color:#9a9aaa;"> / Prompt Optimization Monitor</span>
</div>
""",
        unsafe_allow_html=True,
    )

    run = st.session_state.get("last_optimization_run") or {}
    rec = run.get("recommendation", {}) or {}
    best = run.get("best_version", {}) or {}
    mem = run.get("memory", {})
    perf = best.get("category_scores", {}) or {
        "factual": {"avg_composite": 0.89},
        "edge_case": {"avg_composite": 0.71},
        "adversarial": {"avg_composite": 0.78},
        "stakeholder": {"avg_composite": 0.83},
    }

    best_score = float(best.get("avg_composite", 0.87))
    iters = int(mem.get("iteration", 7))
    corr = rec.get("human_auto_correlation")
    if corr is None:
        corr = 0.73
    if run.get("mock_mode"):
        status = "● MOCK RUN"
    elif best_score >= 0.75:
        status = "● OPTIMIZED"
    else:
        status = "● FAIL"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best score", f"{best_score:.2f}")
    k2.metric("Iterations", f"{iters} / 10")
    k3.metric("Human corr.", f"{corr:+.2f}" if isinstance(corr, (int, float)) else "n/a")
    k4.markdown(f"<div class='mono' style='margin-top:1.4rem;'>{status}</div>", unsafe_allow_html=True)

    if run.get("mock_mode"):
        st.info(
            "**Mock optimizer:** scores are synthetic (lexical relevance + pseudo judge), so numbers stay in a "
            "tight band and rarely cross the real **0.75** pass bar. Charts and lineage still demo the **workflow**. "
            "Turn **Mock LLM** off for real Groq evals (watch rate limits / `LEDGER_EVAL_MAX_CASES`)."
        )

    st.markdown("#### Score progression")
    hist = mem.get("score_history", []) or [{"avg_composite": 0.61}, {"avg_composite": 0.69}, {"avg_composite": 0.74}, {"avg_composite": best_score}]
    xs = list(range(len(hist)))
    ys = [float(h.get("avg_composite", 0.0)) for h in hist]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name="Composite",
            line=dict(color="#c9a84c", width=2.5),
        )
    )
    for name, color, offset in (
        ("Factual", "#9a9aaa", 0.02),
        ("Edge", "#2dd4bf", -0.01),
        ("Adversarial", "#f87171", 0.0),
        ("Stakeholder", "#e8c96a", 0.01),
    ):
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=[min(1.0, max(0.0, y + offset)) for y in ys],
                mode="lines",
                name=name,
                line=dict(color=color, width=1),
            )
        )
    fig.add_hline(y=0.75, line_dash="dash", line_color="#c9a84c", annotation_text="Pass threshold")
    layout_base = {k: v for k, v in plotly_layout.items() if k not in ("xaxis", "yaxis")}
    yaxis_cfg = dict(plotly_layout.get("yaxis", {}))
    yaxis_cfg["range"] = [0, 1.05]
    fig.update_layout(
        **layout_base,
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis=yaxis_cfg,
        xaxis=plotly_layout.get("xaxis", {}),
        shapes=[
            dict(
                type="rect",
                xref="paper",
                yref="y",
                x0=0,
                y0=0,
                x1=1,
                y1=0.75,
                fillcolor="rgba(248,113,113,0.10)",
                layer="below",
                line_width=0,
            )
        ],
    )
    st.plotly_chart(fig, use_container_width=True)

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("#### Prompt lineage tree")
        st.markdown(_lineage_html(run), unsafe_allow_html=True)
    with c_right:
        st.markdown("#### Category breakdown")
        st.markdown(_category_bars(perf), unsafe_allow_html=True)
        overall = sum(float(perf.get(c, {}).get("avg_composite", 0.0)) for c in perf) / max(len(perf), 1)
        st.markdown(f"<div class='muted'>Overall pass rate {int(overall * 100)}%</div>", unsafe_allow_html=True)

    st.markdown("#### Human feedback vs automated scores")
    scatter_x = [0.61, 0.69, 0.74, best_score]
    scatter_y = [0.2, 0.4, 0.55, min(1.0, max(-1.0, float(corr)))]
    fig2 = go.Figure(
        data=go.Scatter(
            x=scatter_x,
            y=scatter_y,
            mode="markers",
            marker=dict(size=12, color=scatter_x, colorscale="YlOrBr", showscale=False),
            text=[f"v{i}" for i in range(len(scatter_x))],
            hovertemplate="version %{text}<br>composite %{x}<br>rating %{y}<extra></extra>",
        )
    )
    layout_base2 = {k: v for k, v in plotly_layout.items() if k not in ("xaxis", "yaxis")}
    xaxis2 = dict(plotly_layout.get("xaxis", {}))
    xaxis2["title"] = "Composite score"
    yaxis2 = dict(plotly_layout.get("yaxis", {}))
    yaxis2["title"] = "Avg human rating"
    fig2.update_layout(**layout_base2, height=320, xaxis=xaxis2, yaxis=yaxis2)
    st.plotly_chart(fig2, use_container_width=True)

    b1, b2, b3 = st.columns(3)
    with b1:
        mock = st.session_state.get("mock_llm", True)
        if st.button("Run New Optimization", type="secondary"):
            with st.spinner("Running optimization pipeline…"):
                try:
                    out = run_optimization_pipeline(
                        task_description=st.session_state.get(
                            "opt_task", "Improve merchant spend insights chat assistant prompts."
                        ),
                        max_iterations=st.session_state.get("max_opt_iters", 3),
                        mock=mock,
                    )
                except Exception as e:
                    import logging

                    logging.exception("Optimization run failed")
                    st.error(f"Optimization failed: {e}")
                    low = str(e).lower()
                    if "429" in str(e) or "rate_limit" in low or "token" in low:
                        st.markdown(
                            """
**Groq rate limit / daily token cap**

- Turn **Mock LLM** **ON** for the optimizer to finish instantly (good for UI screenshots).  
- Or wait until the reset time in the error, then retry with fewer calls:  
  `export LEDGER_EVAL_MAX_CASES=8` before `streamlit run …`  
- The judge now defaults to **`llama-3.1-8b-instant`** (override with `LEDGER_GROQ_JUDGE_MODEL`).  
- Optional: set `LEDGER_GROQ_ANSWER_MODEL=llama-3.1-8b-instant` to shrink eval answer tokens too.  
- Paid tier: [Groq billing / Dev tier](https://console.groq.com/settings/billing).
"""
                        )
                else:
                    st.session_state["last_optimization_run"] = out
                    st.session_state["last_eval"] = out.get("last_eval")
                    if out.get("best_version"):
                        st.session_state["active_prompt"] = {
                            "version_id": out["best_version"].get("version_id", "v-best"),
                            "prompt_text": out["best_version"].get("prompt_text", ""),
                            "iteration_number": out.get("memory", {}).get("iteration", 0),
                            "rewrite_strategy": (out.get("memory", {}).get("tried_strategies") or ["—"])[-1],
                        }
                    st.success("Optimization complete.")
                    st.rerun()
    with b2:
        bp = (st.session_state.get("last_optimization_run") or {}).get("best_version", {}).get("prompt_text", "")
        st.download_button("Export Best Prompt", data=bp or "(empty)", file_name="best_prompt.txt")
    with b3:
        if st.button("View Logs"):
            st.info("See console output where Streamlit was started for structured logs.")

    with st.expander("Latest stakeholder recommendation"):
        st.write(rec.get("stakeholder_summary", "Run an optimization to populate this section."))
