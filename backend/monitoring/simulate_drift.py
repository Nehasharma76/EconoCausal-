"""
Week 4 - Drift detector demo.

Shows the detector on three batches of customers:
  1. No drift      the held-out test customers, which look like the training data
  2. Mild drift    customers spend 35% more than before
  3. Strong drift  customers spend 50% more, visit 2 more times a month, and the region mix shifts

Each batch is compared with the reference stored in the model artifact (the training data summary).
A good detector stays quiet on batch 1 and raises warnings or alerts on batches 2 and 3.

Outputs
-------
  docs/drift_report.json              full results
  docs/figures/drift_demo.png         PSI per feature for each batch
  frontend/public/week4_drift.json    used by the dashboard "Monitoring" tab (includes the reference)

Run from the repo root:   python backend/monitoring/simulate_drift.py
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
from backend.monitoring.drift import PSI_ALERT, PSI_WARN, detect_drift  # noqa: E402

SCORES = ROOT / "backend" / "data" / "ite_scores.csv"
OUT_JSON = ROOT / "docs" / "drift_report.json"
OUT_FIG = ROOT / "docs" / "figures" / "drift_demo.png"
OUT_WEB = ROOT / "frontend" / "public" / "week4_drift.json"


def scenarios(base: pd.DataFrame) -> dict:
    rng = np.random.default_rng(7)
    mild = base.copy()
    mild["past_spend"] = mild["past_spend"] * 1.35

    strong = base.copy()
    strong["past_spend"] = strong["past_spend"] * 1.5
    strong["visits_per_month"] = strong["visits_per_month"] + 2
    strong["region"] = np.where(rng.random(len(strong)) < 0.5, 3, strong["region"])  # half move to region 3
    return {"No drift": base, "Mild drift": mild, "Strong drift": strong}


def main() -> None:
    model = CausalModel.load()
    reference = model.artifact["reference"]
    base = pd.read_csv(SCORES)[model.features]

    results = {name: detect_drift(reference, batch) for name, batch in scenarios(base).items()}
    for name, rep in results.items():
        print(f"\n{name}: overall = {rep['overall'].upper()} ({rep['n_rows']} rows)")
        for f in rep["features"]:
            shift = "" if f["mean_shift_std"] is None else f"  mean shift {f['mean_shift_std']:+.2f} sd"
            print(f"  {f['feature']:<18} PSI {f['psi']:.3f}  {f['status']:<8}{shift}")

    OUT_JSON.write_text(json.dumps(results, indent=2))
    OUT_WEB.write_text(json.dumps({"thresholds": {"warn": PSI_WARN, "alert": PSI_ALERT},
                                   "features": model.features, "reference": reference, "scenarios": results}))

    colors = {"No drift": "#0F6B6B", "Mild drift": "#E0A100", "Strong drift": "#A4262C"}
    feats = model.features
    x = np.arange(len(feats))
    plt.figure(figsize=(8, 4.4))
    for i, (name, rep) in enumerate(results.items()):
        plt.bar(x + (i - 1) * 0.27, [f["psi"] for f in rep["features"]], 0.27, color=colors[name], label=name)
    plt.axhline(PSI_WARN, color="#8896A1", linestyle="--", linewidth=1)
    plt.axhline(PSI_ALERT, color="#14212B", linestyle="--", linewidth=1)
    plt.text(len(feats) - 0.5, PSI_WARN + 0.01, "warning", ha="right", fontsize=8, color="#5B6B77")
    plt.text(len(feats) - 0.5, PSI_ALERT + 0.01, "alert", ha="right", fontsize=8, color="#14212B")
    plt.xticks(x, feats, rotation=15)
    plt.ylabel("PSI (higher = more drift)")
    plt.title("Drift detector on three customer batches")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150)
    plt.close()
    print(f"\nSaved {OUT_JSON.relative_to(ROOT)}, {OUT_FIG.relative_to(ROOT)}, {OUT_WEB.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
