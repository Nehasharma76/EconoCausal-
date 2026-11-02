# Week 1 report: Causal graphing and dashboard scaffolding

## Deliverables

- [x] Mock retail dataset with confounders, treatment and outcome (`backend/data/generate_data.py`)
- [x] Causal DAG defined in DoWhy (`backend/causal/build_dag.py`, `docs/figures/causal_dag.png`)
- [x] Effect identification via backdoor adjustment (`docs/identified_estimand.txt`)
- [x] Propensity score balancing with overlap and balance plots (`backend/causal/propensity.py`)
- [x] React app with campaign upload and budget views (`frontend/`)

## Causal setup

- Treatment: `discount_given`
- Outcome: `purchased`
- Confounders: `past_spend`, `tenure_months`, `visits_per_month`, `price_sensitivity`, `region`

Marketing historically targeted loyal customers, so treated and untreated groups differ before any discount is
applied. Adjusting for the confounders removes that selection effect.

## Results

| Estimate | Value |
|---|---|
| Naive difference in purchase rate | +0.280 |
| Linear backdoor adjustment | +0.161 |
| Propensity (IPW) adjusted | +0.154 |
| True effect (known from simulation) | +0.125 |

After weighting, all covariates have |SMD| below 0.1 (see `docs/balance_table.csv` and `docs/figures/balance_love_plot.png`).

## Next week

Train EconML Double ML models (Random Forest or LightGBM base learners) to estimate per-customer effects, and
build Qini and uplift curves in the dashboard.
