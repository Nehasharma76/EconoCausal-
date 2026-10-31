import numpy as np

from backend.optimize.engine import (blanket_rollout, evaluate, gain_matrix, likely_buyer_rollout,
                                     response_for, solve)

TIERS = (10, 20)


def test_response_for_known_and_unknown_tiers():
    assert response_for(20) == 1.0 and response_for(10) == 0.7
    assert 0.4 < response_for(5) < 0.7          # unknown tier: smooth rule
    assert response_for(40) == 1.0              # capped at the full effect


def test_sure_things_have_negative_gain():
    # High baseline, tiny effect: the discount just gives money away
    g = gain_matrix([0.01], [0.9], [80.0], TIERS)
    assert (g < 0).all()


def test_persuadables_have_positive_gain():
    g = gain_matrix([0.30], [0.10], [80.0], TIERS)
    assert (g > 0).all()


def test_solver_skips_customers_with_negative_gain():
    effect = np.array([0.30, 0.01])
    baseline = np.array([0.10, 0.90])
    value = np.array([80.0, 80.0])
    d = solve(gain_matrix(effect, baseline, value, TIERS), TIERS, budget=1000)
    assert d[0] > 0 and d[1] == 0


def test_solver_respects_budget():
    rng = np.random.default_rng(0)
    n = 100
    d = solve(gain_matrix(rng.uniform(0.1, 0.4, n), rng.uniform(0.05, 0.3, n), np.full(n, 80.0), TIERS), TIERS, budget=300)
    assert d.sum() <= 300


def test_optimizer_beats_simple_strategies():
    rng = np.random.default_rng(1)
    n = 300
    effect, baseline, value = rng.uniform(-0.05, 0.4, n), rng.uniform(0.05, 0.6, n), rng.uniform(40, 120, n)
    budget = 1200
    best = evaluate(solve(gain_matrix(effect, baseline, value, TIERS), TIERS, budget), effect, baseline, value)
    blanket = evaluate(blanket_rollout(n, budget), effect, baseline, value)
    likely = evaluate(likely_buyer_rollout(baseline, budget), effect, baseline, value)
    assert best["net_revenue"] > blanket["net_revenue"]
    assert best["net_revenue"] > likely["net_revenue"]


def test_evaluate_counts_spend_and_customers():
    r = evaluate([20, 0, 10], [0.2, 0.2, 0.2], [0.1, 0.1, 0.1], [50, 50, 50])
    assert r["spend"] == 30 and r["customers_discounted"] == 2
