-- Merchant transaction facts (loaded from data/transactions.csv)
CREATE TABLE IF NOT EXISTS merchant_transactions (
    transaction_id VARCHAR PRIMARY KEY,
    merchant_id VARCHAR,
    merchant_name VARCHAR,
    merchant_category VARCHAR,
    transaction_date DATE,
    amount FLOAT,
    currency VARCHAR,
    payment_method VARCHAR,
    vendor_name VARCHAR,
    budget_category VARCHAR,
    is_flagged BOOLEAN,
    flag_reason VARCHAR,
    region VARCHAR,
    quarter VARCHAR,
    budget_allocated FLOAT
);

-- Prompt versions with full lineage tree
CREATE TABLE IF NOT EXISTS prompt_versions (
    version_id VARCHAR PRIMARY KEY,
    parent_version_id VARCHAR,
    prompt_text TEXT,
    agent_task VARCHAR,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    iteration_number INT,
    rewrite_strategy VARCHAR
);

-- Evaluation results per prompt version per test case
CREATE TABLE IF NOT EXISTS eval_results (
    eval_id VARCHAR PRIMARY KEY,
    version_id VARCHAR REFERENCES prompt_versions(version_id),
    test_case_id VARCHAR,
    test_category VARCHAR,
    faithfulness_score FLOAT,
    relevance_score FLOAT,
    business_alignment_score FLOAT,
    composite_score FLOAT,
    passed BOOLEAN,
    llm_judge_rationale TEXT,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Human feedback from the merchant chat view
CREATE TABLE IF NOT EXISTS human_feedback (
    feedback_id VARCHAR PRIMARY KEY,
    version_id VARCHAR REFERENCES prompt_versions(version_id),
    question TEXT,
    answer TEXT,
    rating INT,
    feedback_note TEXT,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Optimization run metadata
CREATE TABLE IF NOT EXISTS optimization_runs (
    run_id VARCHAR PRIMARY KEY,
    started_at TIMESTAMP_NTZ,
    completed_at TIMESTAMP_NTZ,
    total_iterations INT,
    final_version_id VARCHAR,
    threshold_met BOOLEAN,
    termination_reason VARCHAR
);

-- Chat interaction logs (correlate answers to prompt versions)
CREATE TABLE IF NOT EXISTS chat_logs (
    log_id VARCHAR PRIMARY KEY,
    version_id VARCHAR,
    merchant_id VARCHAR,
    question TEXT,
    answer TEXT,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);
