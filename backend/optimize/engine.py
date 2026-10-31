"""
Budget optimization engine (Weeks 3 and 4).

Shared by the Week 3 script (backend/optimize/prescribe.py) and the Week 4 REST API.

Goal: spend a fixed discount budget so that expected NET REVENUE is as high as possible.

For one customer and one discount of d dollars:
    effect    = estimated rise in purchase probability from a FULL discount (the ITE)
    share     = how much of that effect a discount of d achieves (see response_for)
    baseline  = chance the customer buys anyway, with no discount
    value     = revenue from one purchase (we use the customer's past spend)

    extra revenue  = effect * share * (value - d)    customers persuaded to buy, paying value - d
    wasted cost    = baseline * d                    customers who would have bought anyway still get d off
    net gain(d)    = extra revenue - wasted cost

This is why the causal effect matters. A "Sure thing" has a high baseline, so a discount
mostly gives money away and the net gain is negative. A "Persuadable" has a high effect,
so the net gain is positive. The optimizer only discounts customers whose net gain is
positive, so it may spend LESS than the full budget.

Optimization: each customer receives at most one discount tier (or none), total face value
of discounts stays within the budget, and we maximise total net gain. This is a mixed-integer
linear program solved exactly with scipy.optimize.milp.

Assumptions to state in the viva:
  * A smaller discount gets a smaller share of the effect: $10 gets 70% of a $20 effect.
  * Baseline chance comes from a model fitted on customers who did not get a discount.
  * Revenue per purchase equals the customer's average spend.
  * The budget counts the face value of every discount issued.
"""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, identity

TIERS = (10, 20)
RESPONSE = {10: 0.7, 20: 1.0}
FULL_DISCOUNT = 20


def response_for(tier: float) -> float:
    """Share of the full-discount effect that a discount of `tier` dollars achieves."""
    return RESPONSE.get(tier, min(1.0, (tier / FULL_DISCOUNT) ** 0.5))


def gain_matrix(effect, baseline, order_value, tiers=TIERS) -> np.ndarray:
    """Expected net gain for every tier (rows) and customer (columns)."""
    effect, baseline, value = (np.asarray(a, dtype=float) for a in (effect, baseline, order_value))
    return np.vstack([effect * response_for(t) * (value - t) - baseline * t for t in tiers])


def solve(gains: np.ndarray, tiers=TIERS, budget: float = 5000, time_limit: float = 30) -> np.ndarray:
    """Pick at most one tier per customer to maximise total gain within the budget. Returns dollars per customer."""
    k, n = gains.shape
    cost = np.repeat(np.array(tiers, dtype=float), n)
    constraints = [
        LinearConstraint(hstack([identity(n, format="csr")] * k), ub=np.ones(n)),   # one option per customer
        LinearConstraint(csr_matrix(cost.reshape(1, -1)), ub=[float(budget)]),      # total spend within budget
    ]
    result = milp(c=-gains.reshape(-1), constraints=constraints, integrality=np.ones(n * k),
                  bounds=Bounds(0, 1), options={"time_limit": time_limit})
    if result.x is None:
        raise RuntimeError(f"Optimizer failed: {result.message}")
    chosen = np.round(result.x).reshape(k, n)
    discounts = np.zeros(n)
    for row, tier in zip(chosen, tiers):
        discounts[row == 1] = tier
    return discounts


def blanket_rollout(n: int, budget: float, dollars: float = FULL_DISCOUNT) -> np.ndarray:
    """Baseline 1: give everyone the same discount, in customer order, until the money runs out."""
    discounts = np.zeros(n)
    discounts[: int(budget // dollars)] = dollars
    return discounts


def likely_buyer_rollout(baseline, budget: float, dollars: float = FULL_DISCOUNT) -> np.ndarray:
    """Baseline 2: what ordinary ML does. Give the discount to the customers most likely to buy."""
    n = len(baseline)
    discounts = np.zeros(n)
    top = np.argsort(-np.asarray(baseline))[: min(n, int(budget // dollars))]
    discounts[top] = dollars
    return discounts


def evaluate(discounts, effect, baseline, order_value) -> dict:
    """Score a set of discounts. Pass the estimated effect, or the true effect to check against the truth."""
    d = np.asarray(discounts, dtype=float)
    effect, baseline, value = (np.asarray(a, dtype=float) for a in (effect, baseline, order_value))
    share = np.array([response_for(x) if x > 0 else 0.0 for x in d])
    extra_purchases = effect * share
    net = extra_purchases * (value - d) - baseline * d * (d > 0)
    return {
        "customers_discounted": int((d > 0).sum()),
        "spend": float(d.sum()),
        "extra_purchases": float(extra_purchases.sum()),
        "net_revenue": float(net.sum()),
    }


def summary_sentence(optimizer: dict, blanket: dict, budget: float) -> str:
    saved = optimizer["net_revenue"] - blanket["net_revenue"]
    return (f"With a ${budget:,.0f} budget, this strategy discounts {optimizer['customers_discounted']:,} customers, "
            f"spends ${optimizer['spend']:,.0f}, and is expected to add ${optimizer['net_revenue']:,.0f} in net revenue, "
            f"${saved:,.0f} more than a blanket $20 rollout.")
