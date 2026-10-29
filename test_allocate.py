import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "optimize"))
from allocate import allocate, blanket_rollout, expected_gain  # noqa: E402

RESPONSE = {10: 0.7, 20: 1.0}
EFFECTS = np.array([0.30, 0.20, 0.10, 0.05, -0.05])


def test_hand_worked_example_picks_two_small_discounts():
    # Budget 20. Option A: one $20 on the best customer = 0.30.
    # Option B: two $10 discounts on the best two = 0.7*0.30 + 0.7*0.20 = 0.35 (better).
    d = allocate(EFFECTS, budget=20, tiers=(10, 20), response=RESPONSE)
    assert d.tolist() == [10, 10, 0, 0, 0]
    assert np.isclose(expected_gain(d, EFFECTS, RESPONSE), 0.35)


def test_never_spends_more_than_budget():
    d = allocate(EFFECTS, budget=25, tiers=(10, 20), response=RESPONSE)
    assert d.sum() <= 25


def test_customers_with_negative_effect_get_nothing():
    d = allocate(EFFECTS, budget=1000, tiers=(10, 20), response=RESPONSE)
    assert d[4] == 0


def test_big_budget_gives_best_customers_the_full_discount():
    d = allocate(EFFECTS, budget=1000, tiers=(10, 20), response=RESPONSE)
    assert d[:4].tolist() == [20, 20, 20, 20]


def test_optimizer_beats_blanket_rollout():
    rng = np.random.default_rng(0)
    effects = rng.uniform(0, 0.3, 200)
    best = allocate(effects, budget=600, tiers=(10, 20), response=RESPONSE)
    blanket = blanket_rollout(200, budget=600, dollars=20)
    assert expected_gain(best, effects, RESPONSE) > expected_gain(blanket, effects, RESPONSE)
