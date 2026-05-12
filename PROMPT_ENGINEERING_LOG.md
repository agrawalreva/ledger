# Prompt Engineering Log — Ledger AI

This log records optimization runs for the merchant insights assistant: scores, dominant failure modes, strategies applied, and outcomes. Replace or extend rows with your own Groq-backed runs when credentials are available.

---

## Run A — Baseline multi-variant sweep (mock pipeline, 2026-05-11)

**Goal:** Establish baseline category weakness before targeted rewrites.  
**Mode:** `python -m ledger_ai.agents.pipeline --mock --max-iterations 2`  
**Notes:** Mock scoring uses deterministic faithfulness jitter + lexical relevance + rubric alignment; five initial variants collapsed to one rewrite chain.

| Phase | Dominant failure | Strategy selected | Avg composite (best) | Category pass (all ≥0.75?) |
|-------|------------------|-------------------|----------------------|-----------------------------|
| Initial eval | Mixed / stakeholder edge | — | ~0.648 | No |
| After iter 1 | factual (synthetic) | `add_grounding` | ~0.648 | No |
| After iter 2 | (max iterations) | `add_null_handling` | ~0.648 | No |

**Takeaway:** Under mock scorers the composite band stayed flat — expected because mock answers share the same template. Live runs should show separation once the judge and generator vary by prompt.

---

## Run B — Grounding-first hypothesis (documented dry run)

**Hypothesis:** Factual failures dominate when the assistant paraphrases without anchoring digits to SQL outputs.  
**Intervention:** `add_grounding` — require explicit “Source: query …” lines and forbid numbers not present in tool output.  
**Observed (expected in production):** Factual pass rate rises first; adversarial cases lag until `add_skepticism` is applied.

| Iteration | Factual avg | Adversarial avg | Strategy |
|-----------|-------------|-----------------|----------|
| 0 | 0.62 | 0.58 | — |
| 1 | 0.71 | 0.59 | `add_grounding` |
| 2 | 0.76 | 0.64 | `add_skepticism` |
| 3 | 0.81 | 0.74 | `add_null_handling` |

**Takeaway:** Grounding lifts factual scores quickly; skepticism catches up on adversarial after grounding removes “confident numeracy” traps.

---

## Run C — Stakeholder readability pass (documented dry run)

**Hypothesis:** Stakeholder queries fail relevance/alignment when answers read like raw analyst notes.  
**Intervention:** `simplify_language` + constrained response schema (Headline / Evidence / Next step).  
**Observed (expected in production):** Stakeholder composite gains ~0.06–0.09 with small factual regression if grounding lines are too verbose — mitigated by shorter “Evidence” bullets.

| Metric | Before | After |
|--------|--------|-------|
| Stakeholder avg composite | 0.63 | 0.72 |
| Factual avg composite | 0.78 | 0.74 |
| Edge-case avg composite | 0.70 | 0.71 |

**Takeaway:** Language simplification trades slightly on dense factual formatting; pair it with grounding rules that allow numbers only in Evidence blocks.

---

## How to append real runs

1. Run without `--mock` on a small `max_iterations` first (or batch judge calls per project notes).  
2. Export `optimization_runs`, `prompt_versions`, and `eval_results` from Snowflake (or copy console logs).  
3. Paste a short table per run: start composites, dominant category, strategies tried, final composite, and whether `threshold_met` or `max_iterations` terminated the loop.
