"""
Synthetic retail campaign data for EconoCausal (Week 1).

Why synthetic? Real campaign data never tells you the TRUE effect of a discount
on one customer. Here we simulate it, so in Week 2 we can check whether the
Double ML model recovers the truth, and in the Mid Review we can prove it.

Causal story (this is the DAG we declare in causal/build_dag.py):

    past_spend, tenure_months, visits_per_month, price_sensitivity, region
        -> discount_given   (marketing targets certain customers: CONFOUNDING)
        -> purchased        (these same traits also drive buying)
    discount_given -> purchased   (the causal effect we want to estimate)

Columns
-------
customer_id         unique id
past_spend          average monthly spend (currency units)
tenure_months       months since first purchase
visits_per_month    site/app visits per month
price_sensitivity   0-1, higher = reacts more strongly to discounts
region              0-3 (categorical, encoded as int)
discount_given      treatment: 1 if the customer was shown the discount
purchased           outcome: 1 if the customer bought during the campaign
true_ite            TRUE change in purchase probability caused by the discount
                    (ground truth; the model must NOT use this as a feature)
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
FEATURES = ["past_spend", "tenure_months", "visits_per_month", "price_sensitivity", "region"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    past_spend = rng.gamma(shape=2.0, scale=40.0, size=n)            # ~ mean 80
    tenure_months = rng.integers(1, 61, size=n)
    visits_per_month = rng.poisson(lam=6, size=n)
    price_sensitivity = rng.beta(2, 3, size=n)
    region = rng.integers(0, 4, size=n)

    # Standardised versions used inside the formulas below
    z_spend = (past_spend - 80) / 55
    z_tenure = (tenure_months - 30) / 17
    z_visits = (visits_per_month - 6) / 2.5

    # --- Treatment assignment (NOT random -> confounding) -------------------
    # The marketing team historically targeted loyal, high-spending customers.
    logit_t = -0.3 + 0.9 * z_spend + 0.5 * z_tenure + 0.3 * z_visits + 0.2 * (region == 2)
    p_treat = sigmoid(logit_t)
    discount_given = rng.binomial(1, p_treat)

    # --- Baseline purchase probability (without any discount) ---------------
    logit_y0 = -1.0 + 0.8 * z_spend + 0.4 * z_tenure + 0.5 * z_visits
    p0 = sigmoid(logit_y0)

    # --- True treatment effect: heterogeneous --------------------------------
    # Price-sensitive customers with LOW loyalty respond most (the "Persuadables").
    # Very loyal customers would buy anyway, so the discount does little for them.
    true_ite = np.clip(0.02 + 0.35 * price_sensitivity - 0.10 * np.clip(z_spend, 0, None), -0.05, 0.5)
    # keep probabilities valid
    p1 = np.clip(p0 + true_ite, 0.0, 1.0)

    p_obs = np.where(discount_given == 1, p1, p0)
    purchased = rng.binomial(1, p_obs)

    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "past_spend": past_spend.round(2),
            "tenure_months": tenure_months,
            "visits_per_month": visits_per_month,
            "price_sensitivity": price_sensitivity.round(3),
            "region": region,
            "discount_given": discount_given,
            "purchased": purchased,
            "true_ite": (p1 - p0).round(4),
        }
    )
    return df


if __name__ == "__main__":
    df = generate()
    full_path = OUT_DIR / "retail_campaign.csv"
    df.to_csv(full_path, index=False)

    # Small upload-ready sample for the React dashboard (no ground-truth column)
    sample = df.drop(columns=["true_ite"]).head(300)
    sample_path = OUT_DIR.parents[1] / "frontend" / "public" / "sample_campaign.csv"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(sample_path, index=False)

    naive = df.groupby("discount_given")["purchased"].mean()
    print(f"Saved {len(df)} rows -> {full_path}")
    print(f"Saved sample -> {sample_path}")
    print(f"Treated share: {df['discount_given'].mean():.1%}")
    print(f"Naive purchase-rate difference (treated - control): {naive[1] - naive[0]:+.3f}")
    print(f"TRUE average treatment effect (ATE): {df['true_ite'].mean():+.3f}")
