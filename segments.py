"""
Customer segmentation by treatment effect.

Marketing teams group customers into four classic "uplift" segments:

  Persuadables   buy ONLY because of the discount. The discount is worth paying for.
  Sure things    would buy anyway. A discount just gives away money.
  Lost causes    unlikely to buy either way. A discount is wasted.
  Sleeping dogs  the discount does the least (or may even put them off).

We assign segments from two model outputs in backend/data/ite_scores.csv:
  ite_forest      estimated effect of the discount on this customer (CausalForestDML)
  p_buy_baseline  how likely the customer is to buy anyway (ordinary ML model)

Rules (applied in this order) use the cut-offs below, which are easy to change:
  1. effect in the lowest 10%              -> Sleeping dogs
  2. effect in the highest 30%             -> Persuadables
  3. baseline purchase chance above median -> Sure things
  4. everyone else                         -> Lost causes

Note: causal forests shrink estimates towards the average, so almost none are
below zero. That is why "Sleeping dogs" is defined as the lowest-effect 10%
instead of "effect below zero".

Because the data is simulated we can CHECK the segments against the true effect
(true_ite): Persuadables should have the highest true effect.

Outputs: backend/data/segments.csv, docs/figures/segments.png
Run from the repo root:   python backend/causal/segments.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "backend" / "data" / "ite_scores.csv"
OUT_CSV = ROOT / "backend" / "data" / "segments.csv"
OUT_FIG = ROOT / "docs" / "figures" / "segments.png"

# ---- Cut-offs (change these to change the segments) ----
LOW_EFFECT_QUANTILE = 0.10    # lowest 10% of effects -> Sleeping dogs
HIGH_EFFECT_QUANTILE = 0.70   # top 30% of effects -> Persuadables
HIGH_BASELINE_QUANTILE = 0.50 # baseline above the median -> Sure things

ORDER = ["Persuadables", "Sure things", "Lost causes", "Sleeping dogs"]
COLORS = {"Persuadables": "#0F6B6B", "Sure things": "#E0A100", "Lost causes": "#8896A1", "Sleeping dogs": "#A4262C"}


def assign_segments(df: pd.DataFrame) -> pd.Series:
    """Return a segment label for every row, using ite_forest and p_buy_baseline."""
    low_effect = df["ite_forest"].quantile(LOW_EFFECT_QUANTILE)
    high_effect = df["ite_forest"].quantile(HIGH_EFFECT_QUANTILE)
    high_baseline = df["p_buy_baseline"].quantile(HIGH_BASELINE_QUANTILE)

    segment = np.full(len(df), "Lost causes", dtype=object)          # default, rule 4
    segment[(df["p_buy_baseline"] >= high_baseline).to_numpy()] = "Sure things"   # rule 3
    segment[(df["ite_forest"] >= high_effect).to_numpy()] = "Persuadables"        # rule 2
    segment[(df["ite_forest"] <= low_effect).to_numpy()] = "Sleeping dogs"        # rule 1 (wins ties)
    return pd.Series(segment, index=df.index, name="segment")


def main() -> None:
    if not SCORES.exists():
        raise SystemExit("Scores not found. Run: python backend/causal/train_dml.py")

    df = pd.read_csv(SCORES)
    df["segment"] = assign_segments(df)
    df[["customer_id", "segment"]].to_csv(OUT_CSV, index=False)

    summary = (
        df.groupby("segment")
        .agg(customers=("customer_id", "count"), avg_estimated_effect=("ite_forest", "mean"),
             avg_true_effect=("true_ite", "mean"), avg_baseline_buy=("p_buy_baseline", "mean"))
        .reindex(ORDER)
    )
    summary["share"] = summary["customers"] / len(df)
    print(summary.round(3).to_string())

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(summary.index, summary["customers"], color=[COLORS[s] for s in summary.index])
    axes[0].set_title("Customers per segment")
    axes[0].set_ylabel("customers")
    axes[1].bar(summary.index, summary["avg_true_effect"], color=[COLORS[s] for s in summary.index])
    axes[1].set_title("Average TRUE effect of the discount (simulation check)")
    axes[1].set_ylabel("change in purchase probability")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)
    print(f"\nSaved {OUT_CSV.relative_to(ROOT)} and {OUT_FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
