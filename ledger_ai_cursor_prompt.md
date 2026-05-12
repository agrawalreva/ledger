# Ledger AI — Full Build Cursor Prompt

## Project Overview

Build **Ledger AI**, a production-grade agentic GenAI platform with two purposes:
1. A **merchant-facing chat assistant** that answers natural language questions about payment and transaction data
2. A **self-optimizing prompt engine** — a 4-agent pipeline that autonomously evaluates, diagnoses, and rewrites the chat assistant's prompts until quality thresholds are met

This is a Streamlit application with two views (Merchant Chat and Admin Monitoring), a LangChain multi-agent backend, a multi-signal GenAI eval framework, and Snowflake as the data backend.

---

## Tech Stack

- **Frontend/App**: Streamlit with custom CSS (production-grade UI — see UI section)
- **Agent Framework**: LangChain (LCEL or AgentExecutor)
- **LLM**: Groq API (free tier) — use `langchain_groq` with `llama-3.3-70b-versatile` as the default model for all agents including the LLM-as-judge faithfulness scorer
- **Embeddings**: `sentence-transformers` (all-MiniLM-L6-v2) for cosine similarity scoring
- **Database**: Snowflake (`snowflake-connector-python`) — lightweight backend only
- **Data**: Synthetic merchant transaction data generated with `Faker` + a Kaggle payments CSV as base
- **Environment**: `.env` file with `GROQ_API_KEY`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`, `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA`, `SNOWFLAKE_WAREHOUSE`

---

## Project Structure

```
ledger_ai/
├── app.py                        # Streamlit entry point, view routing
├── .env                          # API keys and Snowflake config
├── requirements.txt
├── data/
│   └── generate_synthetic_data.py  # Generates synthetic merchant transactions
├── db/
│   └── snowflake_client.py       # Snowflake connection, read/write helpers
│   └── schema.sql                # CREATE TABLE statements
├── agents/
│   └── prompt_engineer.py        # Agent 1
│   └── evaluator.py              # Agent 2
│   └── diagnosis_rewrite.py      # Agent 3
│   └── recommendation.py         # Agent 4
│   └── pipeline.py               # Orchestrates the 4-agent loop
├── eval/
│   └── metrics.py                # Faithfulness, relevance, business alignment scorers
│   └── test_cases.py             # 60+ structured test cases across 4 categories
├── chat/
│   └── merchant_chat.py          # Merchant chat agent with Snowflake tool use
├── views/
│   └── chat_view.py              # Streamlit merchant chat UI
│   └── admin_view.py             # Streamlit admin monitoring UI
└── utils/
    └── prompt_templates.py       # Base prompt templates and versioning helpers
```

---

## Data Layer

### Synthetic Data Generation (`data/generate_synthetic_data.py`)

Generate a realistic merchant transaction dataset with `Faker`:

```python
# Generate ~5,000 transactions with these fields:
# transaction_id, merchant_id, merchant_name, merchant_category,
# transaction_date, amount, currency, payment_method,
# vendor_name, budget_category, is_flagged, flag_reason,
# region, quarter
```

Include at least:
- 20 unique merchants across 6 categories (retail, SaaS, logistics, food, travel, utilities)
- Realistic spending patterns with seasonal variance
- ~8% flagged transactions with flag reasons (duplicate, over-budget, high-risk-vendor)
- Budget allocations per merchant per category per quarter

Save as `data/transactions.csv`.

### Snowflake Schema (`db/schema.sql`)

```sql
-- Prompt versions with full lineage tree
CREATE TABLE prompt_versions (
    version_id VARCHAR PRIMARY KEY,
    parent_version_id VARCHAR,           -- NULL for root, else FK to self
    prompt_text TEXT,
    agent_task VARCHAR,                  -- e.g. 'merchant_insights'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    iteration_number INT,
    rewrite_strategy VARCHAR             -- e.g. 'add_grounding', 'inject_few_shot'
);

-- Evaluation results per prompt version per test case
CREATE TABLE eval_results (
    eval_id VARCHAR PRIMARY KEY,
    version_id VARCHAR REFERENCES prompt_versions(version_id),
    test_case_id VARCHAR,
    test_category VARCHAR,               -- factual|edge_case|adversarial|stakeholder
    faithfulness_score FLOAT,
    relevance_score FLOAT,
    business_alignment_score FLOAT,
    composite_score FLOAT,
    passed BOOLEAN,
    llm_judge_rationale TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Human feedback from the merchant chat view
CREATE TABLE human_feedback (
    feedback_id VARCHAR PRIMARY KEY,
    version_id VARCHAR REFERENCES prompt_versions(version_id),
    question TEXT,
    answer TEXT,
    rating INT,                          -- 1 (thumbs up) or -1 (thumbs down)
    feedback_note TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Optimization run metadata
CREATE TABLE optimization_runs (
    run_id VARCHAR PRIMARY KEY,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    total_iterations INT,
    final_version_id VARCHAR,
    threshold_met BOOLEAN,
    termination_reason VARCHAR           -- 'threshold_met' or 'max_iterations'
);
```

---

## Agent Architecture

### Agent 1 — Prompt Engineer (`agents/prompt_engineer.py`)

Given a task description and any prior iteration context, generate N=5 prompt variants.

Each variant must follow a structured format:
- **Role**: Who the assistant is
- **Context**: What data it has access to and how
- **Format**: How it should structure responses
- **Constraints**: What it must/must not do
- **Few-shot examples**: 1–2 worked examples embedded in the prompt

System prompt for the agent:
```python
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

system_prompt = """
You are an expert prompt engineer specializing in business intelligence assistants.
Generate {n} distinct prompt variants for a merchant insights chat assistant.
Each variant must differ meaningfully in: role framing, context injection strategy,
output format, and constraint specificity.
Return a JSON array of variant objects with keys: variant_id, strategy_name, prompt_text, rationale.
Return JSON only — no preamble, no markdown fences.
"""
```

### Agent 2 — Evaluator (`agents/evaluator.py`)

Run every prompt variant against the full 60+ test case suite. For each (variant, test_case) pair, compute three scores:

**Faithfulness (0.0–1.0) — LLM-as-judge:**
```python
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

judge_prompt = """
You are an evaluation judge. Given a question, a reference context, and a model answer,
score the answer's faithfulness to the context on a scale of 0.0 to 1.0.
Faithfulness means: every factual claim in the answer is supported by the context.
Return JSON only, no preamble: {"score": float, "rationale": str, "unsupported_claims": list}
"""
# Call the judge with the question, context, and answer as user message
# Parse the JSON response directly
```

**Relevance (0.0–1.0) — Cosine similarity:**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def relevance_score(question: str, answer: str) -> float:
    q_emb = model.encode(question)
    a_emb = model.encode(answer)
    return float(cosine_similarity([q_emb], [a_emb])[0][0])
```

**Business Alignment (0–2) — Rubric scorer:**
```python
# Check answer for:
# +1 if contains a specific metric or number
# +1 if contains an explicit recommendation or next step
# Normalize to 0.0–1.0
```

**Composite score**: `(faithfulness * 0.4) + (relevance * 0.3) + (business_alignment * 0.3)`

**Pass threshold**: composite >= 0.75 AND no single metric below 0.60

Aggregate by test category. Return a structured failure report:
```python
{
  "variant_id": str,
  "category_scores": {
    "factual": {"avg_composite": float, "pass_rate": float},
    "edge_case": {...},
    "adversarial": {...},
    "stakeholder": {...}
  },
  "dominant_failure_category": str,
  "overall_pass": bool
}
```

### Agent 3 — Diagnosis & Rewrite (`agents/diagnosis_rewrite.py`)

This is the core agentic loop. It receives the failure report and maintains memory across iterations.

**Memory structure:**
```python
rewrite_memory = {
    "iteration": int,
    "tried_strategies": list[str],       # strategies already attempted
    "score_history": list[dict],          # composite scores per iteration
    "worst_category_history": list[str]  # dominant failures over time
}
```

**Rewrite strategies (choose based on dominant failure):**
| Dominant failure | Strategy | What changes |
|---|---|---|
| factual | `add_grounding` | Add explicit instruction to cite data, add few-shot with numbers |
| edge_case | `add_null_handling` | Add instructions for missing/ambiguous data cases |
| adversarial | `add_skepticism` | Add instruction to flag uncertainty, never fabricate |
| stakeholder | `simplify_language` | Remove jargon, restructure output format for business users |

**Loop termination:**
- All category pass rates >= 0.75 → `threshold_met`
- Iteration count >= 10 → `max_iterations`
- Agent must not repeat a strategy already in `tried_strategies`

System prompt:
```python
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

system_prompt = """
You are an expert prompt optimizer with memory of past iterations.
Given a structured failure report and the history of strategies already tried,
diagnose the root cause of failures and select the most targeted rewrite strategy.
Do NOT repeat strategies in: {tried_strategies}.
Return JSON only, no preamble: {{"diagnosis": str, "selected_strategy": str, "rewritten_prompt": str, "rationale": str}}
"""
```

### Agent 4 — Recommendation (`agents/recommendation.py`)

After the loop terminates, synthesize the full optimization history into a final recommendation.

Output a structured report with:
- Best performing prompt version (full text)
- Performance summary across all 4 test categories
- Iteration history showing score improvement
- Plain-English rationale a business stakeholder can read
- Correlation between automated scores and human feedback ratings (from Snowflake)

---

## Eval Framework (`eval/`)

### Test Cases (`eval/test_cases.py`)

Build 60 test cases across 4 categories, each as a dict:
```python
{
    "test_case_id": str,
    "category": "factual" | "edge_case" | "adversarial" | "stakeholder",
    "question": str,
    "reference_context": str,   # excerpt from synthetic data
    "expected_elements": list[str],  # what a good answer must contain
    "adversarial": bool
}
```

**Category breakdown:**
- **Factual (15)**: "What was Merchant X's total spend in Q3?", "Which vendor received the most payments in the logistics category?"
- **Edge case (15)**: Questions about merchants with zero transactions, null budget fields, ambiguous merchant names shared across categories
- **Adversarial (15)**: Questions designed to induce hallucination — asking about data not in context, contradictory framing, leading questions with false premises
- **Stakeholder (15)**: Vague business-language queries — "How are we doing on costs?", "Should I be worried about anything?", "What's the biggest risk this quarter?"

---

## Merchant Chat Agent (`chat/merchant_chat.py`)

A separate LangChain agent powering the merchant chat view. It has one tool:

```python
@tool
def query_snowflake(sql: str) -> str:
    """Query the merchant transaction database. Use this to answer questions about
    spending, vendors, budgets, flags, and payment patterns."""
    # Execute SQL against Snowflake transactions table
    # Return results as a formatted string
```

The agent:
1. Receives the merchant's natural language question
2. Decides whether to call `query_snowflake` (it almost always should)
3. Synthesizes the query results into a clear, actionable answer
4. The active prompt version (latest from Snowflake) is injected as the system prompt

Every response is logged to Snowflake with the active `version_id` so human feedback can be correlated to prompt versions later.

---

## UI — Production Grade Streamlit

### Design Direction

**Aesthetic**: Refined financial intelligence platform. Think private banking meets modern data tooling — clean, authoritative, premium. The name "Ledger AI" should feel like it belongs in a JP Morgan or Goldman internal tool. Not a startup chatbot, not a Bloomberg terminal — the sophisticated middle ground.

**Design language**: Warm off-whites and deep navys with gold as the single accent color. Ledger books are cream and gold; this UI should echo that without being literal. Clean whitespace, precise typography, no clutter.

**Color palette (CSS variables via `st.markdown`):**
```css
--bg-primary: #0d0f14;          /* deep navy-black */
--bg-secondary: #131620;        /* slightly lighter navy */
--bg-card: #191d2a;             /* card surface */
--bg-card-hover: #1f2436;       /* card hover state */
--accent-gold: #c9a84c;         /* ledger gold — primary accent */
--accent-gold-muted: #8a6f32;   /* muted gold for secondary elements */
--accent-gold-bright: #e8c96a;  /* bright gold for highlights */
--accent-teal: #2dd4bf;         /* teal for positive signals / pass states */
--accent-red: #f87171;          /* red for failure / warning states */
--text-primary: #f0efe8;        /* warm off-white */
--text-secondary: #9a9aaa;      /* muted gray */
--text-muted: #555568;          /* subtle labels */
--border: #252840;              /* card borders */
--border-bright: #353858;       /* hover borders */
--gold-line: #c9a84c33;         /* gold at 20% opacity for dividers */
```

**Typography:**
- Display/headings: `Playfair Display` — serif, authoritative, premium financial feel
- Body/data: `IBM Plex Mono` — monospaced for numbers and data values, gives a precise ledger feel
- Labels/UI: `Inter` — clean sans for buttons, labels, navigation
- Import all three from Google Fonts in custom CSS

**Logo / wordmark**: Render the "Ledger AI" wordmark in the top left as:
```html
<span style="font-family: 'Playfair Display'; font-size: 22px; color: #f0efe8;">Ledger</span>
<span style="font-family: 'IBM Plex Mono'; font-size: 22px; color: #c9a84c;"> AI</span>
```

**Animations**:
- Subtle fade-in on card load (opacity 0 → 1, 300ms)
- Score bars animate left to right on render using CSS `@keyframes`
- Gold shimmer on the wordmark on initial load (one-shot, subtle)
- Typing indicator in chat: three gold dots pulsing

**Micro-details that make it feel premium**:
- A single 1px gold line (`--accent-gold` at 30% opacity) as a top border on every card
- Score numbers displayed in `IBM Plex Mono` so they feel precise and data-like
- Version badges styled as small pill tags: `[v3.2]` in gold monospace
- Pass/fail indicators: `● PASS` in teal, `● FAIL` in red, `● OPTIMIZED` in gold
- Section dividers: thin horizontal rule in `--gold-line` color

### Chat View (`views/chat_view.py`)

Layout:
```
┌─────────────────────────────────────────────────────────┐
│  Ledger AI   [merchant: ACME Corp ▼]    [Run Optimizer] │
├───────────────────────────────┬─────────────────────────┤
│                               │  INTELLIGENCE PANEL      │
│   CHAT WINDOW                 │  Active prompt v3.2      │
│                               │  ─────────────────────  │
│  ┌──────────────────────┐     │  Faithfulness  0.84      │
│  │ [merchant message]   │     │  ████████████░░  84%     │
│  └──────────────────────┘     │                          │
│  ┌──────────────────────────┐ │  Relevance     0.79      │
│  │ [v3.2] assistant reply   │ │  ███████████░░░  79%     │
│  │ 👍  👎                   │ │                          │
│  └──────────────────────────┘ │  Alignment     0.81      │
│                               │  ████████████░░  81%     │
│                               │  ─────────────────────  │
│                               │  Last optimized  2h ago  │
│                               │  Iterations used  7/10   │
│                               │  Final strategy          │
├───────────────────────────────┤  add_grounding           │
│  [input box]        [Send →]  │  ─────────────────────  │
│                               │  Feedback  👍 12  👎 3   │
└───────────────────────────────┴─────────────────────────┘
```

Key UI details:
- Chat bubbles: user messages right-aligned, warm off-white text on `--bg-card-hover` bg with a gold left accent border
- Assistant messages left-aligned, same card bg but with a teal left accent border to distinguish
- Version badge `[v3.2]` in gold monospace appears in the top-left corner of every assistant message
- Thumbs up/down styled as minimal icon buttons below each assistant message — they glow gold on hover
- "Run Optimizer" button in top right: gold border, transparent fill, `Playfair Display` label
- Merchant selector dropdown styled to match the dark theme
- Intelligence panel score bars: thin 6px height, rounded, gold fill on dark track

### Admin Monitoring View (`views/admin_view.py`)

Layout:
```
┌──────────────────────────────────────────────────────────────┐
│  Ledger AI  /  Prompt Optimization Monitor                   │
├───────────────┬───────────────┬───────────────┬─────────────┤
│  BEST SCORE   │  ITERATIONS   │  HUMAN CORR.  │  STATUS     │
│  0.87         │  7 / 10       │  +0.73        │  ● OPTIMIZED│
├───────────────┴───────────────┴───────────────┴─────────────┤
│  SCORE PROGRESSION                                           │
│  Line chart — composite + all 4 categories across iterations │
│  x: iteration number, y: score 0.0–1.0                      │
│  Composite line in gold, categories in muted colors         │
│  Threshold line at y=0.75 as a dashed gold horizontal rule  │
├──────────────────────────┬───────────────────────────────────┤
│  PROMPT LINEAGE TREE     │  CATEGORY BREAKDOWN               │
│                          │                                   │
│  v1.0  root       0.61   │  Factual      ████████░  0.89    │
│  └─ v1.1  +grd    0.69   │  Edge case    ██████░░░  0.71    │
│     └─ v1.2  +nul 0.74   │  Adversarial  ███████░░  0.78    │
│        └─ v2.0 ✓  0.87   │  Stakeholder  ████████░  0.83    │
│                          │                                   │
│                          │  Overall pass rate    82%         │
├──────────────────────────┴───────────────────────────────────┤
│  HUMAN FEEDBACK vs AUTOMATED SCORES                          │
│  Scatter plot: x = composite score, y = avg human rating    │
│  Each dot = one prompt version, colored by iteration number  │
├──────────────────────────────────────────────────────────────┤
│  [Run New Optimization]   [Export Best Prompt]   [View Logs] │
└──────────────────────────────────────────────────────────────┘
```

Chart styling (Plotly dark theme matching the palette):
```python
plotly_layout = {
    "paper_bgcolor": "#0d0f14",
    "plot_bgcolor": "#131620",
    "font": {"color": "#f0efe8", "family": "IBM Plex Mono"},
    "xaxis": {"gridcolor": "#252840", "linecolor": "#353858"},
    "yaxis": {"gridcolor": "#252840", "linecolor": "#353858"},
    "colorway": ["#c9a84c", "#2dd4bf", "#9a9aaa", "#f87171", "#e8c96a"]
}
```

The score progression line chart:
- Composite score: solid gold line, 2.5px width
- Category lines: 1px, muted colors
- Threshold at 0.75: dashed gold horizontal rule, labeled "Pass threshold"
- Shaded region below threshold in red at 10% opacity

The prompt lineage tree:
- Render as styled HTML inside `st.markdown` — indented rows, monospace font
- Each row: `[version_id]  [strategy]  [score]  [● PASS / ● FAIL]`
- Best-scoring node: gold text, gold left border
- Root node: muted, labeled "root"

The scatter plot:
- Each dot = one prompt version
- Color encodes iteration number (early = muted, late = bright gold)
- Hovering shows version ID, composite score, human rating, strategy used

The three action buttons at the bottom:
- "Run New Optimization": gold border, transparent, `Playfair Display`, full width of its column
- "Export Best Prompt": ghost button, teal border
- "View Logs": ghost button, muted border

---

## Pipeline Orchestration (`agents/pipeline.py`)

```python
def run_optimization_pipeline(task_description: str, max_iterations: int = 10) -> dict:
    run_id = generate_run_id()
    memory = init_memory()

    # Agent 1: Generate initial prompt variants
    variants = prompt_engineer_agent.run(task_description)
    save_variants_to_snowflake(variants, run_id, iteration=0)

    iteration = 0
    best_version = None

    while iteration < max_iterations:
        # Agent 2: Evaluate all current variants
        eval_results = evaluator_agent.run(variants)
        save_eval_results_to_snowflake(eval_results, run_id, iteration)

        # Check termination
        if all_pass(eval_results):
            best_version = get_best_variant(eval_results)
            log_run_completion(run_id, "threshold_met", iteration, best_version)
            break

        # Agent 3: Diagnose and rewrite
        rewrite_result = diagnosis_rewrite_agent.run(
            failure_report=eval_results,
            memory=memory
        )
        update_memory(memory, rewrite_result, eval_results)

        # Update variants with rewritten prompt
        variants = update_variants(variants, rewrite_result)
        save_variants_to_snowflake(variants, run_id, iteration + 1)

        iteration += 1

    else:
        best_version = get_best_variant(eval_results)
        log_run_completion(run_id, "max_iterations", iteration, best_version)

    # Agent 4: Final recommendation
    recommendation = recommendation_agent.run(
        run_id=run_id,
        history=memory,
        best_version=best_version,
        human_feedback=fetch_human_feedback_from_snowflake(run_id)
    )

    return {"run_id": run_id, "best_version": best_version, "recommendation": recommendation}
```

---

## Requirements (`requirements.txt`)

```
streamlit>=1.35.0
langchain>=0.2.0
langchain-groq>=0.1.6
groq>=0.9.0
sentence-transformers>=3.0.0
snowflake-connector-python>=3.10.0
plotly>=5.22.0
pandas>=2.2.0
numpy>=1.26.0
faker>=25.0.0
scikit-learn>=1.5.0
python-dotenv>=1.0.0
uuid
```

---

## Implementation Notes

1. **Start with data generation first** — run `generate_synthetic_data.py` and verify the CSV before touching agents
2. **Set up Snowflake next** — run `schema.sql`, verify connection with a simple SELECT before building agents
3. **Build and test the eval metrics in isolation** — unit test faithfulness, relevance, and business alignment scorers with hardcoded inputs before wiring to agents
4. **Build agents in order 2 → 1 → 3 → 4** — the evaluator is the core; build it first so you can test the others against it
5. **Build the chat agent separately** from the optimization pipeline — it's a clean standalone component
6. **Wire Streamlit views last** — both views are purely display layers on top of already-working backend code
7. **Use `st.session_state`** to persist chat history, active prompt version, and optimization run status across rerenders
8. **Never hardcode API keys** — always load from `.env` via `python-dotenv`
9. **Add a `--mock` flag** to the pipeline for development — returns fake eval scores without calling the LLM so you can test the pipeline logic and UI without hitting Groq at all
10. **Log everything** — every agent call, every eval result, every rewrite decision should print to console with iteration number and timestamp during development
11. **Groq rate limits**: the free tier allows ~30 requests/minute on `llama-3.3-70b-versatile`. The evaluator agent runs one LLM call per (variant × test_case) pair — that's up to 300 calls per iteration. Add a `time.sleep(2)` between calls in the evaluator loop and use `--mock` during UI development. For a full real run, batch test cases into groups of 10 and add a 60-second pause between batches to stay within limits.

---

## Resume-Ready Deliverables

When complete, the project should produce:
- A working Streamlit app demoed at localhost with both views fully functional
- A `README.md` with architecture diagram, setup instructions, and a section titled "Design Decisions" explaining why each eval metric was chosen
- A `PROMPT_ENGINEERING_LOG.md` documenting 3+ real optimization runs — showing how scores changed across iterations and which strategies worked
- Screenshots of both views for your portfolio

The `PROMPT_ENGINEERING_LOG.md` is critical — it demonstrates the prompt engineering iteration process the job description explicitly calls out, and doubles as proof of rigorous testing and iteration.
