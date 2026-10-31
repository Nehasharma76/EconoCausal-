import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "causal"))
from propensity import COVARIATES, TREATMENT, fit_propensity, ipw_weights, smd  # noqa: E402


def test_smd_is_near_zero_for_identical_groups():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=2000))
    t = pd.Series(np.tile([0, 1], 1000))  # alternating, so both groups look alike
    assert abs(smd(x, t)) < 0.1


def test_smd_is_large_for_shifted_groups():
    rng = np.random.default_rng(0)
    treated = rng.normal(loc=1.0, size=1000)
    control = rng.normal(loc=0.0, size=1000)
    x = pd.Series(np.concatenate([treated, control]))
    t = pd.Series([1] * 1000 + [0] * 1000)
    assert smd(x, t) > 0.8  # about one standard deviation apart


def test_smd_is_smaller_after_weighting_confounded_data():
    # Treatment depends on x, so groups differ. Weighting by 1/propensity should balance them.
    rng = np.random.default_rng(1)
    n = 5000
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-1.2 * x))
    t = rng.binomial(1, p)
    series, tseries = pd.Series(x), pd.Series(t)
    w = ipw_weights(t, p)
    assert abs(smd(series, tseries, w)) < abs(smd(series, tseries))


def test_weights_are_positive():
    t = np.array([1, 0, 1, 0])
    p = np.array([0.2, 0.4, 0.9, 0.5])
    w = ipw_weights(t, p)
    assert (w > 0).all()
    assert np.isclose(w[0], 5.0) and np.isclose(w[1], 1 / 0.6)


def test_propensity_stays_inside_clip_range():
    rng = np.random.default_rng(2)
    n = 1000
    df = pd.DataFrame({c: rng.normal(size=n) for c in COVARIATES})
    df[TREATMENT] = (df["past_spend"] > 0).astype(int)  # almost perfectly predictable treatment
    p = fit_propensity(df)
    assert p.min() >= 0.01 and p.max() <= 0.99
