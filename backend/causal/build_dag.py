"""
Week 1 - Causal graphing with Microsoft DoWhy.

Steps:
  1. Load the campaign data.
  2. Declare the causal DAG (confounders -> treatment, confounders -> outcome,
     treatment -> outcome).
  3. Ask DoWhy to IDENTIFY the effect (backdoor adjustment set).
  4. Draw the DAG to docs/figures/causal_dag.png.
  5. Compare a naive estimate with a confounder-adjusted estimate.

Run from the repo root:   python backend/causal/build_dag.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from dowhy import CausalModel

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend" / "data" / "retail_campaign.csv"
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TREATMENT = "discount_given"
OUTCOME = "purchased"
CONFOUNDERS = ["past_spend", "tenure_months", "visits_per_month", "price_sensitivity", "region"]


def build_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    for c in CONFOUNDERS:
        g.add_edge(c, TREATMENT)
        g.add_edge(c, OUTCOME)
    g.add_edge(TREATMENT, OUTCOME)
    return g


def draw_graph(g: nx.DiGraph, path: Path) -> None:
    pos = {TREATMENT: (0, 0), OUTCOME: (2, 0)}
    n = len(CONFOUNDERS)
    for i, c in enumerate(CONFOUNDERS):
        pos[c] = (1, 2.0 - i * (4.0 / (n - 1)))
    colors = ["#E0A100" if v == TREATMENT else "#0F6B6B" if v == OUTCOME else "#9DB4C0" for v in g.nodes]
    plt.figure(figsize=(9, 6))
    nx.draw_networkx(g, pos, node_color=colors, node_size=2800, font_size=8, arrows=True, arrowsize=18)
    plt.title("EconoCausal causal DAG: confounders, treatment, outcome")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Data not found. Run: python backend/data/generate_data.py")

    df = pd.read_csv(DATA)
    g = build_graph()
    draw_graph(g, FIG_DIR / "causal_dag.png")

    # DoWhy accepts a networkx graph via its GML string
    gml = "\n".join(nx.generate_gml(g))
    model = CausalModel(data=df, treatment=TREATMENT, outcome=OUTCOME, graph=gml)

    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print(estimand)
    (ROOT / "docs" / "identified_estimand.txt").write_text(str(estimand))

    # Naive vs adjusted effect (sanity check, full DML comes in Week 2)
    naive = df.groupby(TREATMENT)[OUTCOME].mean()
    naive_ate = naive[1] - naive[0]

    est = model.estimate_effect(
        estimand,
        method_name="backdoor.linear_regression",
        target_units="ate",
    )
    print(f"\nNaive difference in means : {naive_ate:+.4f}")
    print(f"Adjusted (backdoor) ATE   : {est.value:+.4f}")
    if "true_ite" in df.columns:
        print(f"TRUE ATE (ground truth)   : {df['true_ite'].mean():+.4f}")


if __name__ == "__main__":
    main()
