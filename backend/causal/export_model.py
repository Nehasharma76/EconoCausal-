"""
Week 4 - Package the causal model for the REST API.

Trains the same LinearDML model as Week 2 (same data split), plus a "baseline" model, and
saves everything as ONE small JSON file the API can load:
    backend/api/model_artifact.json

Why LinearDML for the API? On our data it ranked customers best (correlation 0.86 with
the true effect), and a linear effect model is just a list of numbers, so it can be saved
as plain JSON. The causal forest is still used offline for the Week 2 evaluation.

The baseline model answers "how likely is this customer to buy with NO discount?". We fit a
logistic regression on customers who did NOT receive a discount. Under the Week 1 causal
assumptions, that gives the no-discount purchase chance for everyone.

Run from the repo root (after Week 1 data exists):   python backend/causal/export_model.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(Path(__file__).resolve().parent))
from train_dml import FEATURES, OUTCOME, TREATMENT, fit_linear_dml, load_split  # noqa: E402  (Week 2 module)

from backend.monitoring.drift import build_reference  # noqa: E402

OUT = ROOT / "backend" / "api" / "model_artifact.json"


def main() -> None:
    train, test = load_split()

    print("Fitting LinearDML ...")
    dml = fit_linear_dml(train)
    coef = np.ravel(dml.coef_).astype(float)
    intercept = float(np.ravel(dml.intercept_)[0])

    print("Fitting baseline (no-discount) purchase model ...")
    control = train[train[TREATMENT] == 0]
    scaler = StandardScaler().fit(control[FEATURES])
    logit = LogisticRegression(max_iter=1000).fit(scaler.transform(control[FEATURES]), control[OUTCOME])

    effect = test[FEATURES].to_numpy() @ coef + intercept
    true = test["true_ite"].to_numpy()
    test_control = test[test[TREATMENT] == 0]
    auc = roc_auc_score(test_control[OUTCOME], logit.predict_proba(scaler.transform(test_control[FEATURES]))[:, 1])

    artifact = {
        "name": "econocausal-linear-dml",
        "version": "1.0",
        "features": FEATURES,
        "effect_model": {"type": "LinearDML", "coef": coef.round(6).tolist(), "intercept": round(intercept, 6)},
        "baseline_model": {
            "type": "logistic_regression",
            "feature_means": scaler.mean_.round(6).tolist(),
            "feature_scales": scaler.scale_.round(6).tolist(),
            "coef": logit.coef_[0].round(6).tolist(),
            "intercept": round(float(logit.intercept_[0]), 6),
        },
        "reference": build_reference(train, FEATURES),
        "metrics": {
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "test_average_effect": round(float(effect.mean()), 4),
            "test_true_average_effect": round(float(true.mean()), 4),
            "test_rmse_vs_true": round(float(np.sqrt(np.mean((effect - true) ** 2))), 4),
            "test_correlation_with_true": round(float(np.corrcoef(effect, true)[0, 1]), 4),
            "baseline_auc_on_untreated_test": round(float(auc), 4),
        },
    }
    OUT.write_text(json.dumps(artifact, indent=2))
    size_kb = OUT.stat().st_size / 1024
    print(f"Saved {OUT.relative_to(ROOT)} ({size_kb:.1f} KB)")
    print(json.dumps(artifact["metrics"], indent=2))


if __name__ == "__main__":
    main()
