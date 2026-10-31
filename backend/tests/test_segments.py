import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "causal"))
from segments import ORDER, assign_segments  # noqa: E402


def make_frame():
    # 20 customers: effect rises 0.00..0.19, baseline buy chance is independent of effect
    effect = np.linspace(0.0, 0.19, 20)
    baseline = np.tile([0.1, 0.9], 10)
    return pd.DataFrame({"ite_forest": effect, "p_buy_baseline": baseline})


def test_every_customer_gets_a_known_segment():
    labels = assign_segments(make_frame())
    assert len(labels) == 20
    assert set(labels.unique()) <= set(ORDER)


def test_highest_effect_customers_are_persuadables():
    df = make_frame()
    df["segment"] = assign_segments(df)
    top = df.sort_values("ite_forest").tail(6)
    assert (top["segment"] == "Persuadables").all()


def test_lowest_effect_customers_are_sleeping_dogs():
    df = make_frame()
    df["segment"] = assign_segments(df)
    bottom = df.sort_values("ite_forest").head(2)
    assert (bottom["segment"] == "Sleeping dogs").all()
