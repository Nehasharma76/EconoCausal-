"""
Budget optimizer (first version of the Week 3 "prescriptive" step).

Question: we have a fixed discount budget. Which customers should get which
discount so that we win the MOST EXTRA PURCHASES?

Inputs
------
  backend/data/ite_scores.csv   ite_forest = estimated effect of a full $20 discount
  BUDGET                        total money we can spend
  TIERS                         discount sizes a customer can receive, in dollars

Assumption (to be refined in Week 3)
------------------------------------
A smaller discount gives a smaller effect, but not proportionally smaller, because
returns diminish. RESPONSE says what share of the full-discount effect each tier gets:
    $10 gets 70% of the effect of $20.
So expected extra purchases from customer i with discount d =
    ite_forest[i] * RESPONSE[d]

Method
------
This is a "pick one option per customer" problem with a budget, solved exactly as
a mixed-integer linear program (scipy.optimize.milp):
  variable x[i, d] = 1 if customer i gets discount d, else 0
  each customer picks at most one discount:  sum over d of x[i, d] <= 1
  total spend stays within budget:           sum of d * x[i, d] <= BUDGET
  maximise:                                  sum of gain[i, d] * x[i, d]

Customers with a zero or negative effect are never worth a discount, so the
optimizer gives them nothing.

Outputs: backend/data/allocation.csv and a printed summary comparing the optimizer
with a blanket rollout (same budget, $20 to customers in ID order).

Run from the repo root:   python backend/optimize/allocate.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, identity

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "backend" / "data" / "ite_scores.csv"
OUT_CSV = ROOT / "backend" / "data" / "allocation.csv"

BUDGET = 5000
TIERS = (10, 20)
RESPONSE = {10: 0.7, 20: 1.0}  # share of the full-discount effect each tier achieves


def allocate(effects, budget=BUDGET, tiers=TIERS, response=None) -> np.ndarray:
    """Return the discount (in dollars, 0 = none) for each customer."""
    response = response or RESPONSE
    effects = np.asarray(effects, dtype=float)
    n, k = len(effects), len(tiers)

    # Variable order: all customers' first tier, then all customers' second tier, ...
    cost = np.repeat(np.array(tiers, dtype=float), n)
    gain = np.concatenate([effects * response[t] for t in tiers])

    one_option = hstack([identity(n, format="csr")] * k)          # one row per customer
    spend = csr_matrix(cost.reshape(1, -1))                        # a single budget row
    constraints = [
        LinearConstraint(one_option, ub=np.ones(n)),
        LinearConstraint(spend, ub=[budget]),
    ]
    result = milp(c=-gain, constraints=constraints, integrality=np.ones(n * k), bounds=Bounds(0, 1))
    if not result.success:
        raise RuntimeError(f"Optimizer failed: {result.message}")

    chosen = np.round(result.x).reshape(k, n)                      # row per tier, column per customer
    discounts = np.zeros(n)
    for row, tier in zip(chosen, tiers):
        discounts[row == 1] = tier
    return discounts


def expected_gain(discounts, effects, response=None) -> float:
    response = response or RESPONSE
    return float(sum(effects[i] * response[d] for i, d in enumerate(discounts) if d > 0))


def blanket_rollout(n, budget=BUDGET, dollars=20) -> np.ndarray:
    """Baseline: give the same discount to customers in ID order until the money runs out."""
    discounts = np.zeros(n)
    discounts[: int(budget // dollars)] = dollars
    return discounts


def main() -> None:
    if not SCORES.exists():
        raise SystemExit("Scores not found. Run: python backend/causal/train_dml.py")

    df = pd.read_csv(SCORES).sort_values("customer_id").reset_index(drop=True)
    est, true = df["ite_forest"].to_numpy(), df["true_ite"].to_numpy()

    best = allocate(est)
    blanket = blanket_rollout(len(df))

    out = pd.DataFrame({
        "customer_id": df["customer_id"],
        "discount": best.astype(int),
        "expected_gain": [est[i] * RESPONSE[int(d)] if d > 0 else 0.0 for i, d in enumerate(best)],
    }).round(4)
    out.to_csv(OUT_CSV, index=False)

    print(f"Customers: {len(df)} | Budget: ${BUDGET} | Tiers: {TIERS}\n")
    print(f"{'Strategy':<22}{'Customers':>10}{'Spend':>9}{'Est. extra purchases':>22}{'True extra purchases':>22}")
    for name, d in [("Optimizer", best), ("Blanket $20 rollout", blanket)]:
        print(f"{name:<22}{int((d > 0).sum()):>10}{d.sum():>9.0f}"
              f"{expected_gain(d, est):>22.1f}{expected_gain(d, true):>22.1f}")
    print("\n(True extra purchases uses the simulated ground truth, which a real company would not have.)")
    print(f"Saved {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
