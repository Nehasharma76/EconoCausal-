"""
Compare base learners for Double ML.

Double ML needs two helper ("nuisance") models: one predicts the outcome, one
predicts the treatment. The project brief allows Random Forest or LightGBM. Here
we try three options with the same LinearDML setup and the same 70/30 split, so
the only thing that changes is the base learner:

  RandomForest       what Week 2 uses
  GradientBoosting   scikit-learn boosting
  LightGBM           fast gradient boosting (skipped if not installed)

For each learner we report, on the held-out customers:
  average effect       compare with the true average effect
  RMSE                 average error against the true per-customer effect (lower is better)
  correlation          how well the ranking matches the true effect (higher is better)
  fit seconds          training time

Outputs: docs/learner_comparison.csv  (and a printed table)
Run from the repo root:   python backend/causal/compare_learners.py   (a few minutes)
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from econml.dml import LinearDML
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)

sys.path.append(str(Path(__file__).resolve().parent))
from train_dml import FEATURES, OUTCOME, TREATMENT, load_split  # noqa: E402  (Week 2 module)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = ROOT / "docs" / "learner_comparison.csv"


def learners():
    """Return {name: (outcome model, treatment model)}."""
    options = {
        "RandomForest": (
            RandomForestRegressor(n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=0),
            RandomForestClassifier(n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=0),
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=0),
            GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0),
        ),
    }
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        options["LightGBM"] = (
            LGBMRegressor(n_estimators=200, learning_rate=0.05, min_child_samples=20, random_state=0, verbose=-1),
            LGBMClassifier(n_estimators=200, learning_rate=0.05, min_child_samples=20, random_state=0, verbose=-1),
        )
    except ImportError:
        print("LightGBM not installed, skipping it (pip install lightgbm).")
    return options


def main() -> None:
    train, test = load_split()
    true = test["true_ite"].to_numpy()
    rows = []
    for name, (model_y, model_t) in learners().items():
        print(f"Fitting LinearDML with {name} ...")
        start = time.time()
        est = LinearDML(model_y=model_y, model_t=model_t, discrete_treatment=True, cv=5, random_state=0)
        est.fit(train[OUTCOME], train[TREATMENT], X=train[FEATURES])
        seconds = time.time() - start
        effect = np.ravel(est.effect(test[FEATURES]))
        rows.append({
            "learner": name,
            "average_effect": effect.mean(),
            "true_average_effect": true.mean(),
            "rmse_vs_true": float(np.sqrt(np.mean((effect - true) ** 2))),
            "correlation_with_true": float(np.corrcoef(effect, true)[0, 1]),
            "fit_seconds": seconds,
        })

    table = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(OUT_CSV, index=False)
    print("\n" + table.round(4).to_string(index=False))

    best = table.sort_values("rmse_vs_true").iloc[0]
    print(f"\nLowest error: {best['learner']} (RMSE {best['rmse_vs_true']:.4f}).")
    print("Keep the learner with low error AND reasonable fit time. Differences this small are")
    print("within noise on 1,500 customers, so do not over-read them.")


if __name__ == "__main__":
    main()
