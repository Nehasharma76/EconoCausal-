"""
Mid-Project Review "Causal Audit" - DoWhy refutation tests.

A refutation test tries to BREAK the causal estimate. If the estimate survives,
we have evidence (not proof) that it reflects a real causal effect.

Tests run on the Week 1 DAG and a backdoor linear-regression estimate:
  1. Placebo treatment      replace the discount with a random fake treatment.
                            Expected: the effect drops to about 0.
  2. Random common cause    add a random, irrelevant confounder.
                            Expected: the effect barely changes.
  3. Data subset            re-estimate on random 80% subsets of the data.
                            Expected: the effect stays stable.

Run from the repo root:   python backend/causal/refute.py
Saves the full report to docs/refutation_results.txt
"""

import sys
from pathlib import Path

import networkx as nx
import pandas as pd
from dowhy import CausalModel

sys.path.append(str(Path(__file__).resolve().parent))
from build_dag import DATA, OUTCOME, ROOT, TREATMENT, build_graph  # noqa: E402  (Week 1 module)

REPORT = ROOT / "docs" / "refutation_results.txt"


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Data not found. Run: python backend/data/generate_data.py")

    df = pd.read_csv(DATA).drop(columns=["true_ite"])  # never let ground truth into the model
    gml = "\n".join(nx.generate_gml(build_graph()))
    model = CausalModel(data=df, treatment=TREATMENT, outcome=OUTCOME, graph=gml)
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression", target_units="ate")
    print(f"Original estimated effect: {estimate.value:+.4f}\n")

    tests = [
        ("placebo_treatment_refuter", {"placebo_type": "permute", "num_simulations": 50}),
        ("random_common_cause", {"num_simulations": 50}),
        ("data_subset_refuter", {"subset_fraction": 0.8, "num_simulations": 50}),
    ]
    lines = [f"Original estimated effect: {estimate.value:+.4f}", ""]
    for name, kwargs in tests:
        result = model.refute_estimate(estimand, estimate, method_name=name, **kwargs)
        print(result, "\n")
        lines += [f"=== {name} ===", str(result), ""]
    REPORT.write_text("\n".join(lines))
    print(f"Saved {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
