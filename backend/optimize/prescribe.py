"""
Week 3 - Prescriptive optimization.

Turns the causal effect estimates into an action: WHO gets WHICH discount.

Steps
-----
1. Load the held-out customers (backend/data/ite_scores.csv).
2. Score each one with the exported model: effect of a full discount, and chance of
   buying anyway (backend/api/model_artifact.json, made by export_model.py).
3. Solve the budget problem with the engine (backend/optimize/engine.py).
4. Compare with two simple strategies:
     blanket rollout   same $20 discount for customers in ID order until money runs out
     likely buyers     $20 to the customers most likely to buy (ordinary ML)
5. Evaluate every strategy twice: with the model's ESTIMATED effects, and with the
   simulated TRUE effects (only possible because the data is synthetic).
6. Repeat for a range of budgets to show how the benefit grows with the budget.

Outputs
-------
  backend/data/prescription.csv           discount chosen for each customer
  docs/week3_metrics.json                 headline numbers
  docs/figures/budget_sweep.png           net revenue against budget for each strategy
  frontend/public/week3_prescription.json used by the dashboard "Allocation" tab

Run from the repo root:   python backend/optimize/prescribe.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from backend.api.model import CausalModel  # noqa: E402
from backend.optimize.engine import (TIERS, blanket_rollout, evaluate, gain_matrix,  # noqa: E402
                                     likely_buyer_rollout, solve, summary_sentence)

SCORES = ROOT / "backend" / "data" / "ite_scores.csv"
OUT_CSV = ROOT / "backend" / "data" / "prescription.csv"
OUT_METRICS = ROOT / "docs" / "week3_metrics.json"
OUT_FIG = ROOT / "docs" / "figures" / "budget_sweep.png"
OUT_WEB = ROOT / "frontend" / "public" / "week3_prescription.json"

BUDGET = 5000
SWEEP_BUDGETS = [500, 1000, 2500, 5000, 7500, 10000, 15000, 20000]


def strategies(effect, baseline, value, budget):
    gains = gain_matrix(effect, baseline, value, TIERS)
    return {
        "optimizer": solve(gains, TIERS, budget),
        "blanket": blanket_rollout(len(effect), budget),
        "likely_buyers": likely_buyer_rollout(baseline, budget),
    }


def main() -> None:
    if not SCORES.exists():
        raise SystemExit("Scores not found. Run: python backend/causal/train_dml.py")

    df = pd.read_csv(SCORES).sort_values("customer_id").reset_index(drop=True)
    model = CausalModel.load()
    effect, baseline = model.effect(df), model.baseline(df)
    value = df["past_spend"].to_numpy()          # revenue from one purchase
    true = df["true_ite"].to_numpy()

    chosen = strategies(effect, baseline, value, BUDGET)
    result = {}
    for name, d in chosen.items():
        result[name] = {"estimated": evaluate(d, effect, baseline, value), "true": evaluate(d, true, baseline, value)}

    # Per-customer prescription
    best = chosen["optimizer"]
    gains = gain_matrix(effect, baseline, value, TIERS)
    tier_index = {t: i for i, t in enumerate(TIERS)}
    expected_gain = np.array([gains[tier_index[int(d)], i] if d > 0 else 0.0 for i, d in enumerate(best)])
    table = pd.DataFrame({
        "customer_id": df["customer_id"], "discount": best.astype(int), "effect": effect.round(4),
        "baseline": baseline.round(4), "order_value": value.round(2), "expected_net_gain": expected_gain.round(2),
    })
    table.drop(columns=[]).to_csv(OUT_CSV, index=False)

    # Budget sweep (evaluated with the true effect)
    sweep = {"budgets": SWEEP_BUDGETS, "optimizer": [], "blanket": [], "likely_buyers": []}
    for b in SWEEP_BUDGETS:
        for name, d in strategies(effect, baseline, value, b).items():
            sweep[name].append(round(evaluate(d, true, baseline, value)["net_revenue"], 2))

    metrics = {"budget": BUDGET, "tiers": list(TIERS), "n_customers": int(len(df)), "results": result, "sweep": sweep}
    OUT_METRICS.write_text(json.dumps(metrics, indent=2))

    web = {
        "budget": BUDGET, "tiers": list(TIERS), "n_customers": int(len(df)),
        "strategies": {k: v["estimated"] for k, v in result.items()},
        "strategies_true": {k: v["true"] for k, v in result.items()},
        "sweep": sweep,
        "rows": table.to_dict(orient="records"),
    }
    OUT_WEB.write_text(json.dumps(web))

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 4.4))
    for name, color, label in [("optimizer", "#0F6B6B", "Optimizer"), ("blanket", "#8896A1", "Blanket $20 rollout"),
                               ("likely_buyers", "#E0A100", "Likely buyers")]:
        plt.plot(SWEEP_BUDGETS, sweep[name], marker="o", color=color, label=label, linewidth=2.2 if name == "optimizer" else 1.6)
    plt.axhline(0, color="#14212B", linewidth=0.8)
    plt.xlabel("discount budget ($)")
    plt.ylabel("net revenue gain ($, true effect)")
    plt.title("Net revenue gain against budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150)
    plt.close()

    print(f"Customers: {len(df)} | Budget: ${BUDGET} | Tiers: {TIERS}\n")
    print(f"{'Strategy':<16}{'Customers':>10}{'Spend':>9}{'Est. net revenue':>18}{'True net revenue':>18}{'True extra purchases':>22}")
    for name in ["optimizer", "blanket", "likely_buyers"]:
        e, t = result[name]["estimated"], result[name]["true"]
        print(f"{name:<16}{e['customers_discounted']:>10}{e['spend']:>9.0f}{e['net_revenue']:>18.0f}{t['net_revenue']:>18.0f}{t['extra_purchases']:>22.1f}")
    print("\n" + summary_sentence(result["optimizer"]["estimated"], result["blanket"]["estimated"], BUDGET))
    print("\nBudget sweep (true net revenue):")
    print(pd.DataFrame(sweep).to_string(index=False))
    print(f"\nSaved {OUT_CSV.relative_to(ROOT)}, {OUT_METRICS.relative_to(ROOT)}, {OUT_FIG.relative_to(ROOT)}, {OUT_WEB.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
