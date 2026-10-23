"""
Week 2 - Uplift and Qini evaluation.

Reads backend/data/ite_scores.csv (made by train_dml.py) and answers:
"If we give discounts to the customers the model ranks highest, how many EXTRA
purchases do we get compared with giving discounts at random?"

Curves
------
Qini curve   x = share of customers targeted (best-ranked first)
             y = cumulative extra purchases caused by the discount

Two versions are produced:
  ground truth  uses the simulated true effect of each customer. This is only
                possible because the data is synthetic, and it is the cleanest
                proof that the model ranks customers correctly.
  observed      uses only what a real company would see (who got a discount, who
                bought), with inverse propensity weights from Week 1 to correct
                for the biased targeting. It is noisier, especially on small data.

  random      straight line from 0 to the total effect (no skill)
  oracle      ranking by the TRUE effect (only possible on synthetic data)
  predictive  ranking by P(purchase) (what ordinary ML would do)
  linear / forest   the two Double ML models

Outputs
-------
  frontend/public/week2_results.json   used by the React "Results" tab
  docs/week2_metrics.json              headline numbers for the report
  docs/figures/qini_curves.png, docs/figures/decile_uplift.png

Run from the repo root:   python backend/causal/uplift.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from propensity import fit_propensity  # noqa: E402  (Week 1 module)

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "backend" / "data" / "ite_scores.csv"
FIG_DIR = ROOT / "docs" / "figures"
WEB_JSON = ROOT / "frontend" / "public" / "week2_results.json"
METRICS_JSON = ROOT / "docs" / "week2_metrics.json"

TREATMENT = "discount_given"
OUTCOME = "purchased"
POINTS = 101  # number of points kept on each curve


def ipw_terms(df: pd.DataFrame, e: np.ndarray) -> np.ndarray:
    """Per-customer contribution to the estimated causal effect (inverse propensity weighting)."""
    e = np.clip(e, 0.05, 0.95)
    t, y = df[TREATMENT].to_numpy(), df[OUTCOME].to_numpy()
    return t * y / e - (1 - t) * y / (1 - e)


def cumulative_gain(order_scores, per_customer_effect) -> np.ndarray:
    """Sort customers by score (high first) and accumulate their effect. Returns array of length n+1."""
    idx = np.argsort(-np.asarray(order_scores), kind="stable")
    return np.concatenate([[0.0], np.cumsum(np.asarray(per_customer_effect)[idx])])


def downsample(curve: np.ndarray, points: int = POINTS) -> np.ndarray:
    pos = np.linspace(0, len(curve) - 1, points).round().astype(int)
    return curve[pos]


def qini_area(model_curve: np.ndarray, random_curve: np.ndarray) -> float:
    """Area between a curve and the random line, with x scaled to 0-1 (higher is better)."""
    x = np.linspace(0, 1, len(model_curve))
    diff = model_curve - random_curve
    return float(np.sum((diff[1:] + diff[:-1]) / 2 * np.diff(x)))  # trapezoid rule (works on any NumPy)


def decile_table(df: pd.DataFrame, score_col: str, deciles: int = 10) -> pd.DataFrame:
    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    ranked["decile"] = np.floor(np.arange(len(ranked)) * deciles / len(ranked)).astype(int) + 1
    return ranked.groupby("decile").agg(estimated=(score_col, "mean"), true=("true_ite", "mean")).reset_index()


def build_curves(order: dict, per_customer_effect, total_for_random) -> dict:
    n = len(per_customer_effect)
    curves = {"random": np.linspace(0, total_for_random, n + 1)}
    for name, scores in order.items():
        curves[name] = cumulative_gain(scores, per_customer_effect)
    return curves


def main() -> None:
    if not SCORES.exists():
        raise SystemExit("Scores not found. Run: python backend/causal/train_dml.py")

    df = pd.read_csv(SCORES)
    n = len(df)
    k = int(0.3 * n)
    e = fit_propensity(df)
    obs_terms = ipw_terms(df, e)
    true_effect = df["true_ite"].to_numpy()

    rankings = {
        "forest": df["ite_forest"], "linear": df["ite_linear"],
        "predictive": df["p_buy_baseline"], "oracle": df["true_ite"],
    }
    truth = build_curves(rankings, true_effect, true_effect.sum())
    observed = build_curves({k_: v for k_, v in rankings.items() if k_ != "oracle"}, obs_terms, obs_terms.sum())

    def summarise(curves):
        return {
            "extra_purchases_top30": {name: round(float(c[k]), 1) for name, c in curves.items()},
            "qini_area": {name: round(qini_area(c, curves["random"]), 2) for name, c in curves.items() if name != "random"},
        }

    deciles = decile_table(df, "ite_forest")
    metrics = {
        "n_test": int(n),
        "true_ate": float(df["true_ite"].mean()),
        "linear_ate": float(df["ite_linear"].mean()),
        "forest_ate": float(df["ite_forest"].mean()),
        "forest_rmse": float(np.sqrt(np.mean((df["ite_forest"] - df["true_ite"]) ** 2))),
        "linear_rmse": float(np.sqrt(np.mean((df["ite_linear"] - df["true_ite"]) ** 2))),
        "forest_corr": float(np.corrcoef(df["ite_forest"], df["true_ite"])[0, 1]),
        "linear_corr": float(np.corrcoef(df["ite_linear"], df["true_ite"])[0, 1]),
        "predictive_corr": float(np.corrcoef(df["p_buy_baseline"], df["true_ite"])[0, 1]),
        "ground_truth": summarise(truth),
        "observed": summarise(observed),
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2))

    xs = downsample(np.linspace(0, 1, n + 1)).round(4).tolist()
    pack = lambda curves: {"x": xs, **{name: downsample(c).round(2).tolist() for name, c in curves.items()}}
    payload = {
        "metrics": metrics,
        "curves": {"ground_truth": pack(truth), "observed": pack(observed)},
        "deciles": {
            "decile": deciles["decile"].tolist(),
            "estimated": deciles["estimated"].round(4).tolist(),
            "true": deciles["true"].round(4).tolist(),
        },
    }
    WEB_JSON.parent.mkdir(parents=True, exist_ok=True)
    WEB_JSON.write_text(json.dumps(payload))

    # ---- Static figures for the report ----
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0, 1, n + 1)
    styles = {"forest": ("#0F6B6B", "Causal forest (DML)", 2.4), "linear": ("#4C9A9A", "Linear DML", 1.6),
              "predictive": ("#E0A100", "Predictive baseline", 1.8), "random": ("#8896A1", "Random rollout", 1.4),
              "oracle": ("#14212B", "Oracle (true effect)", 1.2)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for ax, curves, title in [(axes[0], truth, "Ground truth (simulation)"), (axes[1], observed, "Observed data (IPW-weighted)")]:
        for key, c in curves.items():
            color, label, lw = styles[key]
            ax.plot(x, c, color=color, label=label, linewidth=lw, linestyle="--" if key in ("random", "oracle") else "-")
        ax.set_title(title)
        ax.set_xlabel("share of customers given a discount (best-ranked first)")
    axes[0].set_ylabel("extra purchases caused by the discount")
    axes[0].legend()
    fig.suptitle("Qini curves on held-out customers")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "qini_curves.png", dpi=150)
    plt.close(fig)

    plt.figure(figsize=(7.5, 4.2))
    w = 0.38
    plt.bar(deciles["decile"] - w / 2, deciles["estimated"], w, color="#0F6B6B", label="model estimate")
    plt.bar(deciles["decile"] + w / 2, deciles["true"], w, color="#E0A100", label="true effect (simulation)")
    plt.xlabel("decile of customers (1 = highest predicted effect)")
    plt.ylabel("average effect on purchase probability")
    plt.title("Estimated vs true effect by decile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "decile_uplift.png", dpi=150)
    plt.close()

    print(f"Test customers: {n}")
    for label, summary in [("GROUND TRUTH", metrics["ground_truth"]), ("OBSERVED (IPW)", metrics["observed"])]:
        print(f"\n--- {label} ---")
        print(f"Extra purchases if we discount the top 30% ({k} customers):")
        for name, v in summary["extra_purchases_top30"].items():
            print(f"  {name:<11}: {v:7.1f}")
        print("Qini area vs random (higher is better):")
        for name, v in summary["qini_area"].items():
            print(f"  {name:<11}: {v:7.2f}")
    print(f"\nCorrelation with true effect -> forest {metrics['forest_corr']:.3f}, linear {metrics['linear_corr']:.3f}, predictive {metrics['predictive_corr']:.3f}")
    print(f"Saved {WEB_JSON.relative_to(ROOT)}, {METRICS_JSON.relative_to(ROOT)} and figures in docs/figures/")


if __name__ == "__main__":
    main()
