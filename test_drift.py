import numpy as np
import pandas as pd

from backend.monitoring.drift import build_reference, detect_drift, psi


def make(n=3000, seed=0, shift=1.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "spend": rng.gamma(2.0, 40.0, n) * shift,
        "visits": rng.poisson(6, n),
        "region": rng.integers(0, 4, n),
    })


def test_psi_is_zero_for_identical_distributions():
    assert psi([0.25, 0.25, 0.5], [0.25, 0.25, 0.5]) == 0


def test_reference_marks_region_as_categorical():
    ref = build_reference(make(), ["spend", "visits", "region"])
    assert ref["region"]["type"] == "categorical"
    assert ref["spend"]["type"] == "numeric"


def test_same_distribution_is_not_flagged():
    train = make(seed=0)
    ref = build_reference(train, ["spend", "visits", "region"])
    report = detect_drift(ref, make(seed=1))
    assert report["overall"] == "ok"


def test_big_shift_triggers_alert():
    ref = build_reference(make(seed=0), ["spend", "visits", "region"])
    report = detect_drift(ref, make(seed=1, shift=1.6))
    spend = next(f for f in report["features"] if f["feature"] == "spend")
    assert spend["status"] == "alert"
    assert report["overall"] == "alert"


def test_unseen_region_is_detected():
    train = make(seed=0)
    ref = build_reference(train, ["spend", "visits", "region"])
    new = make(seed=1)
    new["region"] = 9                               # a region never seen in training
    report = detect_drift(ref, new)
    region = next(f for f in report["features"] if f["feature"] == "region")
    assert region["status"] == "alert"


def test_small_batch_adds_a_note():
    ref = build_reference(make(), ["spend", "visits", "region"])
    assert detect_drift(ref, make(n=50, seed=2))["notes"]
