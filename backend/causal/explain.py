"""
Which customer features drive the discount effect?

The causal forest splits customers into groups where the discount works
differently. How often (and how usefully) it splits on a feature tells us how much
that feature explains differences in the effect. This is "effect heterogeneity".

Two views are produced:
  1. Forest feature importance   from the fitted CausalForestDML (Week 2 model)
  2. Correlation with the effect how strongly each feature moves with the
                                 estimated effect on the held-out customers

Because the data is simulated, we know the truth: the discount effect was built to
depend mostly on price_sensitivity (higher = bigger effect) and a little on
past_spend (higher = smaller effect). A good model should rank price_sensitivity
first. If it does not, the model is not finding the real pattern.

Outputs: docs/feature_importance.csv, docs/figures/effect_feature_importance.png
Run from the repo root:   python backend/causal/explain.py   (takes about a minute)
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from train_dml import FEATURES, fit_causal_forest, load_split  # noqa: E402  (Week 2 module)

ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = ROOT / "docs" / "feature_importance.csv"
OUT_FIG = ROOT / "docs" / "figures" / "effect_feature_importance.png"


def main() -> None:
    train, test = load_split()
    print("Fitting CausalForestDML ...")
    forest = fit_causal_forest(train)

    importance = pd.Series(forest.feature_importances_, index=FEATURES, name="forest_importance")
    effect = np.ravel(forest.effect(test[FEATURES]))
    corr = pd.Series({f: np.corrcoef(test[f], effect)[0, 1] for f in FEATURES}, name="correlation_with_effect")

    table = pd.concat([importance, corr], axis=1).sort_values("forest_importance", ascending=False)
    table.index.name = "feature"
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(OUT_CSV)
    print(table.round(3).to_string())

    top = table.index[0]
    print(f"\nMost important feature: {top}")
    print("Expected from the simulation: price_sensitivity" + (" (matches)" if top == "price_sensitivity" else " (does NOT match)"))

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    ordered = table.sort_values("forest_importance")
    plt.figure(figsize=(7, 4))
    plt.barh(ordered.index, ordered["forest_importance"], color="#0F6B6B")
    plt.xlabel("importance in the causal forest")
    plt.title("Which features drive the discount effect?")
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150)
    plt.close()
    print(f"Saved {OUT_CSV.relative_to(ROOT)} and {OUT_FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
