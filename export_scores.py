"""
Export customer effect scores for the dashboard (Mid Review "Data Load" check).

The React app cannot read a CSV file straight from backend/, so this script turns
backend/data/ite_scores.csv into a small JSON file the browser can fetch:
    frontend/public/ite_scores.json

Only the columns the dashboard needs are kept. The ground-truth column (true_ite)
is deliberately left out: a real company would never have it.

Run from the repo root:   python backend/causal/export_scores.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "backend" / "data" / "ite_scores.csv"
OUT = ROOT / "frontend" / "public" / "ite_scores.json"

KEEP = [
    "customer_id", "past_spend", "tenure_months", "visits_per_month",
    "price_sensitivity", "region", "ite_forest", "ite_linear",
]


def main() -> None:
    if not SCORES.exists():
        raise SystemExit("Scores not found. Run: python backend/causal/train_dml.py")
    df = pd.read_csv(SCORES)[KEEP].round(4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(OUT, orient="records")
    print(f"Saved {len(df)} customers -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
