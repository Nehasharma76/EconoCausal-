"""
Week 2 - Double Machine Learning (DML) training with EconML.

Builds on Week 1: same data (backend/data/retail_campaign.csv), same
treatment / outcome / confounders as the DAG in build_dag.py.

What Double ML does
-------------------
1. Fit a model that predicts the OUTCOME from the confounders (model_y).
2. Fit a model that predicts the TREATMENT from the confounders (model_t).
   These are the "nuisance" models; they use cross-fitting so a customer is never
   scored by a model that saw them.
3. Remove what the confounders explain from both treatment and outcome
   (residualise), then regress outcome residuals on treatment residuals.
   What is left is the causal effect, and it can vary by customer (the ITE).

We train two estimators and a baseline:
  * LinearDML        - effect varies linearly with the features
  * CausalForestDML  - effect varies non-linearly (the main model)
  * Predictive baseline - a standard "who is likely to buy" classifier, which is
                          what ordinary ML would use to pick who gets a discount

Run from the repo root (after Week 1 data exists):
    python backend/causal/train_dml.py
Output: backend/data/ite_scores.csv  (held-out test customers with all scores)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML, LinearDML
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend" / "data" / "retail_campaign.csv"
OUT = ROOT / "backend" / "data" / "ite_scores.csv"

TREATMENT = "discount_given"
OUTCOME = "purchased"
FEATURES = ["past_spend", "tenure_months", "visits_per_month", "price_sensitivity", "region"]
SEED = 42


def load_split(seed: int = SEED):
    """70/30 split, stratified on treatment so both parts keep the same discount share."""
    df = pd.read_csv(DATA)
    return train_test_split(df, test_size=0.3, random_state=seed, stratify=df[TREATMENT])


def nuisance_models():
    """Base learners for the nuisance steps. Swap in LightGBM here if you prefer."""
    model_y = RandomForestRegressor(n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=0)
    model_t = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=0)
    return model_y, model_t


def fit_linear_dml(train: pd.DataFrame) -> LinearDML:
    model_y, model_t = nuisance_models()
    est = LinearDML(model_y=model_y, model_t=model_t, discrete_treatment=True, cv=5, random_state=0)
    est.fit(train[OUTCOME], train[TREATMENT], X=train[FEATURES])
    return est


def fit_causal_forest(train: pd.DataFrame) -> CausalForestDML:
    model_y, model_t = nuisance_models()
    est = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        n_estimators=300,
        min_samples_leaf=20,
        cv=5,
        random_state=0,
    )
    est.fit(train[OUTCOME], train[TREATMENT], X=train[FEATURES])
    return est


def fit_predictive_baseline(train: pd.DataFrame) -> GradientBoostingClassifier:
    """Ordinary ML: predict P(purchase) from customer features. Ignores the discount."""
    clf = GradientBoostingClassifier(random_state=0)
    clf.fit(train[FEATURES], train[OUTCOME])
    return clf


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Data not found. Run: python backend/data/generate_data.py")

    train, test = load_split()
    print(f"Train: {len(train)} customers | Test: {len(test)} customers")

    print("Fitting LinearDML ...")
    linear = fit_linear_dml(train)
    print("Fitting CausalForestDML (this takes a minute) ...")
    forest = fit_causal_forest(train)
    print("Fitting predictive baseline ...")
    baseline = fit_predictive_baseline(train)

    X_test = test[FEATURES]
    scores = test[["customer_id", TREATMENT, OUTCOME, "true_ite"] + FEATURES].copy()
    scores["ite_linear"] = np.ravel(linear.effect(X_test))
    scores["ite_forest"] = np.ravel(forest.effect(X_test))
    scores["p_buy_baseline"] = baseline.predict_proba(X_test)[:, 1]
    scores.to_csv(OUT, index=False)

    lo, hi = linear.ate_interval(X_test, alpha=0.05)
    print("\n=== Average effect of the discount on purchase probability (test set) ===")
    print(f"TRUE ATE (simulation)        : {scores['true_ite'].mean():+.4f}")
    print(f"LinearDML ATE                : {scores['ite_linear'].mean():+.4f}  (95% CI {float(lo):+.3f} to {float(hi):+.3f})")
    print(f"CausalForestDML ATE          : {scores['ite_forest'].mean():+.4f}")

    print("\n=== How close are the per-customer effects to the truth? ===")
    for name in ["ite_linear", "ite_forest"]:
        corr = np.corrcoef(scores[name], scores["true_ite"])[0, 1]
        print(f"{name:<12} RMSE vs true ITE: {rmse(scores[name], scores['true_ite']):.4f} | correlation: {corr:.3f}")
    print(f"\nSaved {len(scores)} scored customers -> {OUT}")


if __name__ == "__main__":
    main()
