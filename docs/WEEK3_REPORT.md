# Week 3 report: Prescriptive optimization and the allocation matrix

## Deliverables

- [x] Revenue-aware optimizer with a hard budget constraint (`backend/optimize/engine.py`)
- [x] Script that prescribes a discount for every customer and compares strategies (`backend/optimize/prescribe.py`)
- [x] Per-customer allocation file (`backend/data/prescription.csv`) and budget sweep (`docs/figures/budget_sweep.png`)
- [x] Dashboard "Allocation" tab: allocation matrix with filters, sorting and pages, plus the budget chart
- [x] Tests for the optimizer (`backend/tests/test_engine.py`)

## From effects to action

Week 2 produced a causal effect for each customer. Week 3 turns that into a decision: who gets which discount
($10, $20 or none) so that net revenue is highest without spending more than the budget.

For one customer and a discount of d dollars:

- extra revenue = effect x share(d) x (order value - d), the customers the discount persuades, who pay the reduced price
- wasted cost = baseline chance x d, customers who would have bought anyway still receive the discount
- net gain = extra revenue - wasted cost

The wasted cost is why the causal effect matters. A "Sure thing" has a high baseline chance, so a discount mostly
gives money away and the net gain is negative. The optimizer therefore skips them, and it may spend less than the
whole budget when no worthwhile customers are left.

The problem is solved exactly as a mixed-integer linear program (`scipy.optimize.milp`): each customer gets at most
one discount tier, total spend stays within budget, and total net gain is maximised.

## Assumptions

- A $10 discount achieves 70% of the effect of a $20 discount (diminishing returns).
- Baseline purchase chance comes from a logistic model fitted on customers who did not get a discount.
- Revenue per purchase equals the customer's average monthly spend.
- The budget counts the face value of every discount issued.

## Results (1,500 held-out customers, $5,000 budget)

| Strategy | Customers discounted | Estimated net revenue | True net revenue (simulation) | True extra purchases |
|---|---|---|---|---|
| Optimizer | 481 | $4,643 | $2,365 | 56.4 |
| Blanket $20 rollout | 250 | $288 | -$232 | 33.3 |
| Likely-buyers targeting | 250 | $26 | -$2,443 | 10.7 |

- Giving a flat discount, or discounting the customers most likely to buy, loses money. The optimizer earns a clear profit.
- The estimate ($4,643) is higher than the true value ($2,365). The optimizer selects customers the model rates highest, which includes some where the model was too optimistic. Always quote the true figure from the simulation as the reliable one.
- As the budget grows the optimizer's gain levels off around a $10,000 budget, because every worthwhile customer already has a discount. The two simple strategies keep losing more money.

## Limitations

- The response of a $10 discount is assumed, not measured.
- The optimizer uses estimated effects, so its self-reported gains are optimistic. The true-effect check is only possible on synthetic data.
- The benefit relies on the effect model being right for the customers chosen.
