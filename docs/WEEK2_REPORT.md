# Week 2 report: Double ML training and uplift curves

## Deliverables

- [x] Double ML training with EconML: `LinearDML` and `CausalForestDML` (`backend/causal/train_dml.py`)
- [x] Per-customer effect (ITE) for 1,500 held-out customers (`backend/data/ite_scores.csv`)
- [x] Qini curves and decile analysis (`backend/causal/uplift.py`, `docs/figures/qini_curves.png`, `docs/figures/decile_uplift.png`)
- [x] Predictive baseline to compare against ordinary ML
- [x] DoWhy refutation tests for the Mid Review Causal Audit (`backend/causal/refute.py`, `docs/refutation_results.txt`)
- [x] React "Uplift results" tab with Plotly Qini and decile charts (`frontend/src/components/UpliftView.jsx`)

## Method

Double ML fits one model for the outcome and one for the treatment from the same confounders as the Week 1 DAG,
using 5-fold cross-fitting with Random Forest base learners. It removes what the confounders explain from both,
then estimates the effect of the discount on what remains. The causal forest lets that effect vary non-linearly
across customers. Data is split 70/30, stratified by treatment, and all results are on the held-out 30%.

## Results

| Measure | Value |
|---|---|
| True average effect (simulation) | +0.127 |
| LinearDML average effect | +0.146 (95% CI +0.112 to +0.180) |
| CausalForestDML average effect | +0.150 |
| Correlation with true per-customer effect | linear 0.86, forest 0.80, predictive baseline -0.28 |
| RMSE vs true per-customer effect | linear 0.056, forest 0.057 |

Extra purchases from discounting the top 30% (ground truth): forest 93.6, linear 96.9, random 57.0, predictive 44.0,
oracle 101.8.

## Refutation tests (Causal Audit)

| Test | Original | New effect | Reading |
|---|---|---|---|
| Placebo treatment | +0.161 | -0.001 | Effect disappears with a fake treatment |
| Random common cause | +0.161 | +0.161 | Unchanged by an irrelevant confounder |
| Data subset (80%) | +0.161 | +0.160 | Stable across subsets |

## Honest limitations

- On this data the linear model matches or slightly beats the forest, since the simulated effect is close to
  linear. The forest should win on data with more complex effects.
- The observed-data (IPW) Qini curve is noisy at 1,500 customers. The forest does not clearly beat random on its
  full-curve area there, though it does in the top 30%. The ground-truth curve is the cleaner proof.
- The data is simulated. Results on real campaign data will be harder to read.

## Next week

SciPy optimizer: allocate a fixed budget across customers using these effects, and build the allocation table.
