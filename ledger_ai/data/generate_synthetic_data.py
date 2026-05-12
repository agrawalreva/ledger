"""
Generate ~5,000 synthetic merchant transactions (Faker) and save to data/transactions.csv.
Optionally merge summary stats from a Kaggle-style payments CSV if PAYMENTS_BASE_CSV is set.
"""

from __future__ import annotations

import json
import os
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

faker = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_CSV = SCRIPT_DIR / "transactions.csv"
CATALOG_JSON = SCRIPT_DIR / "merchant_catalog.json"

# 20 merchants across 6 categories (retail, SaaS, logistics, food, travel, utilities)
MERCHANTS: list[dict] = [
    {"merchant_id": "m-ret-001", "name": "Northwind Retail Co", "category": "retail"},
    {"merchant_id": "m-ret-002", "name": "Atlas Outfitters", "category": "retail"},
    {"merchant_id": "m-ret-003", "name": "Harbor Street Market", "category": "retail"},
    {"merchant_id": "m-saas-01", "name": "Vertex Cloud Systems", "category": "SaaS"},
    {"merchant_id": "m-saas-02", "name": "LedgerFlow Analytics", "category": "SaaS"},
    {"merchant_id": "m-saas-03", "name": "Northline CRM", "category": "SaaS"},
    {"merchant_id": "m-log-001", "name": "Continental Freight", "category": "logistics"},
    {"merchant_id": "m-log-002", "name": "Pacific Parcel Network", "category": "logistics"},
    {"merchant_id": "m-log-003", "name": "Ironhorse Trucking", "category": "logistics"},
    {"merchant_id": "m-food-01", "name": "Copperline Catering", "category": "food"},
    {"merchant_id": "m-food-02", "name": "Urban Grain Bistro", "category": "food"},
    {"merchant_id": "m-food-03", "name": "Summit Provisions", "category": "food"},
    {"merchant_id": "m-trv-001", "name": "Meridian Travel Group", "category": "travel"},
    {"merchant_id": "m-trv-002", "name": "Silverwing Airlines Desk", "category": "travel"},
    {"merchant_id": "m-util-01", "name": "Gridline Utilities", "category": "utilities"},
    {"merchant_id": "m-util-02", "name": "Clearwater Energy Co", "category": "utilities"},
    {"merchant_id": "m-ret-004", "name": "Sterling Home Goods", "category": "retail"},
    {"merchant_id": "m-saas-04", "name": "Obsidian Security Suite", "category": "SaaS"},
    {"merchant_id": "m-log-004", "name": "BlueAnchor Maritime", "category": "logistics"},
    {"merchant_id": "m-food-04", "name": "Riverside Commissary", "category": "food"},
]

BUDGET_CATEGORIES = ["Marketing", "Operations", "IT", "Travel", "Facilities", "Payroll-adjacent"]
REGIONS = ["US-West", "US-East", "US-Central", "EU-North", "EU-Central", "APAC"]
PAYMENT_METHODS = ["card", "ach", "wire", "virtual_card", "check"]
FLAG_REASONS = ["duplicate", "over-budget", "high-risk-vendor"]

QUARTER_SEASON_MULT = {"Q1": 0.92, "Q2": 1.0, "Q3": 1.08, "Q4": 1.15}


def _quarter_from_month(d: date) -> str:
    q = (d.month - 1) // 3 + 1
    return f"Q{q}"


def _budget_key(merchant_id: str, budget_cat: str, quarter: str) -> tuple:
    return (merchant_id, budget_cat, quarter)


def build_budget_allocations() -> dict[tuple, float]:
    """Per-merchant per budget category per quarter baseline spend cap."""
    out: dict[tuple, float] = {}
    for m in MERCHANTS:
        for bc in BUDGET_CATEGORIES:
            for q in ("Q1", "Q2", "Q3", "Q4"):
                base = random.uniform(15_000, 120_000)
                out[_budget_key(m["merchant_id"], bc, q)] = round(base, 2)
    return out


def generate_rows(n: int = 5000) -> pd.DataFrame:
    budgets = build_budget_allocations()
    rows: list[dict] = []
    start = date(2023, 1, 1)
    end = date(2025, 4, 30)

    for i in range(n):
        m = random.choice(MERCHANTS)
        d = start + timedelta(days=random.randint(0, (end - start).days))
        quarter = _quarter_from_month(d)
        budget_cat = random.choice(BUDGET_CATEGORIES)
        mult = QUARTER_SEASON_MULT[quarter]
        amount = round(max(20.0, np.random.lognormal(mean=6.5, sigma=1.1) * mult), 2)
        currency = "USD" if random.random() < 0.88 else random.choice(["EUR", "GBP"])
        vendor = faker.company()
        pm = random.choice(PAYMENT_METHODS)
        is_flagged = random.random() < 0.08
        flag_reason = random.choice(FLAG_REASONS) if is_flagged else ""
        if is_flagged and flag_reason == "over-budget":
            cap = budgets.get(_budget_key(m["merchant_id"], budget_cat, quarter), 50_000)
            amount = round(cap * random.uniform(1.05, 1.45), 2)
        alloc = budgets.get(_budget_key(m["merchant_id"], budget_cat, quarter), 50_000)

        rows.append(
            {
                "transaction_id": f"txn-{i+1:05d}",
                "merchant_id": m["merchant_id"],
                "merchant_name": m["name"],
                "merchant_category": m["category"],
                "transaction_date": d.isoformat(),
                "amount": amount,
                "currency": currency,
                "payment_method": pm,
                "vendor_name": vendor,
                "budget_category": budget_cat,
                "is_flagged": is_flagged,
                "flag_reason": flag_reason,
                "region": random.choice(REGIONS),
                "quarter": quarter,
                "budget_allocated": alloc,
            }
        )

    df = pd.DataFrame(rows)
    return df


def maybe_blend_kaggle_base(df: pd.DataFrame) -> pd.DataFrame:
    path = os.environ.get("PAYMENTS_BASE_CSV")
    if not path or not Path(path).is_file():
        return df
    base = pd.read_csv(path, nrows=2000)
    # Light touch: nudge amount distribution toward base column if present
    amt_col = None
    for c in base.columns:
        if "amount" in c.lower() or "payment" in c.lower():
            amt_col = c
            break
    if amt_col and amt_col in base.columns:
        sample = base[amt_col].dropna().astype(float)
        if len(sample) > 50:
            noise = np.random.choice(sample.values, size=len(df), replace=True)
            df["amount"] = (df["amount"] * 0.85 + pd.Series(noise).values * 0.15).round(2)
    return df


def main() -> None:
    df = generate_rows(5000)
    df = maybe_blend_kaggle_base(df)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    with open(CATALOG_JSON, "w", encoding="utf-8") as f:
        json.dump(MERCHANTS, f, indent=2)
    print(f"Wrote {len(df)} rows to {OUT_CSV}")
    print(f"Wrote merchant catalog to {CATALOG_JSON}")


if __name__ == "__main__":
    main()
