# EconoCausal: Dynamic Pricing via Double Machine Learning

Standard ML predicts who is likely to buy. EconoCausal estimates **how much a discount changes each customer's
chance of buying** (the Individual Treatment Effect), then spends a fixed budget on the customers the discount
actually moves ("Persuadables").

Infotact Solutions Advanced Data Science & ML Engineering (Vol. II), Project 2.

## Project status

| Week | Backend (EconML, DoWhy, SciPy, FastAPI) | Frontend (React, Plotly) | Status |
|---|---|---|---|
| 1 | Causal DAG, synthetic data, propensity balancing | Upload and budget views | Done |
| 2 | Double ML training (ITE), uplift curves | Uplift results tab | Done |
| Mid review | DoWhy refutation tests | Customer scores table (filter and chart thousands of ITE scores) | Done |
| 3 | SciPy budget optimizer (revenue-aware) | Allocation matrix | Done |
| 4 | REST API, drift detector, Dockerfile | Monitoring tab, live budget, plain-language summaries | Done |

Extra analysis: customer segments, feature importance, base-learner comparison, viva notes.

## Repository layout

```
backend/
  data/generate_data.py       synthetic retail data with a known true effect
  causal/build_dag.py         Week 1: DoWhy DAG and effect identification
  causal/propensity.py        Week 1: propensity scores, balance plots, IPW
  causal/train_dml.py         Week 2: LinearDML and CausalForestDML
  causal/uplift.py            Week 2: Qini curves and deciles
  causal/refute.py            Mid review: DoWhy refutation tests
  causal/export_scores.py     dashboard data export
  causal/segments.py          Persuadables, Sure things, Lost causes, Sleeping dogs
  causal/explain.py           which features drive the effect
  causal/compare_learners.py  Random Forest vs Gradient Boosting vs LightGBM
  causal/export_model.py      Week 4: package the model for the API
  optimize/allocate.py        first optimizer prototype (purchase-based)
  optimize/engine.py          Week 3: revenue-aware optimizer (shared with the API)
  optimize/prescribe.py       Week 3: prescribe discounts, compare strategies, budget sweep
  api/main.py                 Week 4: FastAPI service
  api/model.py                loads backend/api/model_artifact.json
  monitoring/drift.py         Week 4: PSI drift detector
  monitoring/simulate_drift.py  drift demo on three batches
  tests/                      pytest (45 tests)
frontend/                     React (Vite) dashboard with 5 tabs
docs/                         reports, API guide, figures, viva notes
Dockerfile                    packages the API
```

## Run the backend

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt

# Weeks 1 and 2
python backend/data/generate_data.py
python backend/causal/build_dag.py
python backend/causal/propensity.py
python backend/causal/train_dml.py
python backend/causal/uplift.py
python backend/causal/refute.py

# Week 3
python backend/causal/export_model.py       # packages the model (also needed by Week 4)
python backend/optimize/prescribe.py        # optimizer, comparisons, budget sweep

# Week 4
python backend/monitoring/simulate_drift.py
uvicorn backend.api.main:app --port 8000    # API, docs at http://localhost:8000/docs

pytest backend/tests
```

All generated files are already included in the repo, so the dashboard works without re-running anything.

## Run the dashboard

```bash
cd frontend
npm install
npm run dev
```

Tabs: **Setup** (upload data, set a budget), **Uplift results**, **Customer scores**, **Allocation**, **Monitoring**.
If the API is running on port 8000, the Allocation tab can recalculate for any budget; otherwise it shows saved results.

## Key findings

On synthetic data where the true effect of a discount is known (true average +0.125):

- **Confounding is large.** A naive comparison gives +0.280 because marketing discounted loyal customers. Propensity weighting brings every covariate's standardised difference under 0.1.
- **Double ML recovers the effect.** LinearDML estimates +0.146 (95% CI +0.112 to +0.180) against +0.127 on the test set.
- **Causal targeting beats the alternatives.** Discounting the top 30% ranked by effect gives about 97 extra purchases, against 57 at random and 44 for an ordinary "likely buyer" model.
- **Refutation tests pass.** A placebo treatment gives about 0, and a random extra confounder or 80% data subsets leave the estimate unchanged.
- **The optimizer earns money where simple strategies lose it.** With a $5,000 budget the true net revenue is +$2,365 for the optimizer, -$232 for a blanket $20 rollout, and -$2,443 for likely-buyers targeting.
- **Drift is detected.** The PSI monitor is quiet on normal data, warns on a 35% spend increase and alerts on a large behaviour shift.

See `docs/` for the full weekly reports, `docs/API.md` for the API, and `docs/MID_REVIEW_NOTES.md` for viva preparation.

## Git workflow (required by Infotact)

- `main` holds merged code only; the Team Lead reviews and merges.
- Each member commits to their own branch (for example `member-a`, `member-b`).
- Commit meaningful changes at least daily. Empty or single-character commits do not count.
