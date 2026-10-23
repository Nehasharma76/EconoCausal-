import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "causal"))
from uplift import cumulative_gain, decile_table, downsample, qini_area  # noqa: E402


def test_cumulative_gain_orders_best_first():
    scores = np.array([0.1, 0.9, 0.5])
    effect = np.array([1.0, 3.0, 2.0])
    gain = cumulative_gain(scores, effect)
    assert gain.tolist() == [0.0, 3.0, 5.0, 6.0]


def test_good_ranking_beats_random():
    rng = np.random.default_rng(0)
    effect = rng.uniform(0, 1, 1000)
    random_curve = np.linspace(0, effect.sum(), 1001)
    good = cumulative_gain(effect, effect)
    bad = cumulative_gain(-effect, effect)
    assert qini_area(good, random_curve) > 0 > qini_area(bad, random_curve)


def test_downsample_keeps_endpoints():
    curve = np.arange(1001, dtype=float)
    out = downsample(curve, 101)
    assert len(out) == 101 and out[0] == 0 and out[-1] == 1000


def test_decile_table_has_ten_groups():
    df = pd.DataFrame({"score": np.arange(100.0), "true_ite": np.arange(100.0)})
    table = decile_table(df, "score")
    assert len(table) == 10
    assert table.loc[0, "estimated"] > table.loc[9, "estimated"]
