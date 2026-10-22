"""
Week 1 - Propensity score balancing.

Goal: make the treated and untreated groups comparable, as if we had run a
randomised A/B test. We:
  1. Fit a logistic regression for P(discount_given = 1 | confounders).
  2. Check overlap (do treated and untreated customers share propensity ranges?).
  3. Compute inverse-propensity weights (IPW).
  4. Compare standardised mean differences (SMD) BEFORE and AFTER weighting.
     A common rule: |SMD| < 0.1 means a covariate is well balanced.

Outputs saved to docs/figures/ : propensity_overlap.png, balance_love_plot.png
Run from the repo root:   python backend/causal/propensity.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend" / "data" / "retail_campaign.csv"
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TREATMENT = "discount_given"
COVARIATES = ["past_spend", "tenure_months", "visits_per_month", "price_sensitivity", "region"]


def fit_propensity(df: pd.DataFrame) -> np.ndarray:
    X = StandardScaler().fit_transform(df[COVARIATES])
    model = LogisticRegression(max_iter=1000).fit(X, df[TREATMENT])
    return np.clip(model.predict_proba(X)[:, 1], 0.01, 0.99)  # clip avoids huge weights


def ipw_weights(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.where(t == 1, 1.0 / p, 1.0 / (1.0 - p))


def smd(x: pd.Series, t: pd.Series, w=None) -> float:
    w = np.ones(len(x)) if w is None else w

    def wmean_var(mask):
        xv, wv = x[mask].to_numpy(), w[mask]
        m = np.average(xv, weights=wv)
        v = np.average((xv - m) ** 2, weights=wv)
        return m, v

    m1, v1 = wmean_var((t == 1).to_numpy())
    m0, v0 = wmean_var((t == 0).to_numpy())
    return (m1 - m0) / np.sqrt((v1 + v0) / 2)


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Data not found. Run: python backend/data/generate_data.py")

    df = pd.read_csv(DATA)
    p = fit_propensity(df)
    df["propensity"] = p
    w = ipw_weights(df[TREATMENT].to_numpy(), p)

    # Overlap plot
    plt.figure(figsize=(7, 4))
    plt.hist(p[df[TREATMENT] == 0], bins=40, alpha=0.6, label="not discounted", color="#9DB4C0")
    plt.hist(p[df[TREATMENT] == 1], bins=40, alpha=0.6, label="discounted", color="#0F6B6B")
    plt.xlabel("propensity score")
    plt.ylabel("customers")
    plt.title("Propensity overlap")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "propensity_overlap.png", dpi=150)
    plt.close()

    # Balance table + Love plot
    rows = []
    for c in COVARIATES:
        rows.append({"covariate": c, "smd_before": smd(df[c], df[TREATMENT]), "smd_after": smd(df[c], df[TREATMENT], w)})
    bal = pd.DataFrame(rows)
    print(bal.round(3).to_string(index=False))
    bal.to_csv(ROOT / "docs" / "balance_table.csv", index=False)

    y = np.arange(len(bal))
    plt.figure(figsize=(7, 4))
    plt.scatter(bal["smd_before"].abs(), y, label="before weighting", color="#E0A100")
    plt.scatter(bal["smd_after"].abs(), y, label="after IPW", color="#0F6B6B")
    plt.axvline(0.1, color="grey", linestyle="--", linewidth=1)
    plt.yticks(y, bal["covariate"])
    plt.xlabel("|standardised mean difference|")
    plt.title("Covariate balance (dashed line = 0.1 threshold)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "balance_love_plot.png", dpi=150)
    plt.close()

    # Weighted ATE (a simple, non-DML estimate to compare against in Week 2)
    t, yv = df[TREATMENT].to_numpy(), df["purchased"].to_numpy()
    ate_ipw = np.average(yv[t == 1], weights=w[t == 1]) - np.average(yv[t == 0], weights=w[t == 0])
    print(f"\nIPW-adjusted ATE: {ate_ipw:+.4f}")
    if "true_ite" in df.columns:
        print(f"TRUE ATE        : {df['true_ite'].mean():+.4f}")


if __name__ == "__main__":
    main()
