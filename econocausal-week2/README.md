# EconoCausal: Dynamic Pricing via Double Machine Learning

Standard ML predicts who is likely to buy. EconoCausal estimates **how much a discount changes each customer's
chance of buying** (the Individual Treatment Effect), then spends a fixed budget on the customers the discount
actually moves ("Persuadables").

Infotact Solutions Advanced Data Science & ML Engineering (Vol. II), Project 2.

## Project status

| Week | Backend (EconML, DoWhy) | Frontend (React, Plotly) | Status |
|---|---|---|---|
| 1 | Causal DAG, synthetic data, propensity balancing | Upload and budget views | Done |
| 2 | Double ML training (ITE) | Qini and uplift curves | Done |
| Mid review | DoWhy refutation tests | Large ITE data load | Backend done, data load check next |
| 3 | SciPy budget optimizer | Allocation table | Planned |
| 4 | REST API, drift detector | Polish, plain-language summaries | Planned |

## Repository layout

```
backend/
  data/generate_data.py     synthetic retail data with a known true effect
  causal/build_dag.py       DoWhy DAG, effect identification, naive vs adjusted ATE
  causal/propensity.py      propensity scores, overlap and balance plots, IPW
  causal/train_dml.py       Week 2: LinearDML and CausalForestDML, per-customer effects
  causal/uplift.py          Week 2: Qini curves, decile checks, dashboard JSON
  causal/refute.py          Mid review: DoWhy refutation tests
  tests/                    pytest checks
frontend/                   React (Vite) app: CSV upload, budget settings
docs/                       Work Distribution Document, Week 1 report, figures
```

## Run the backend

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt

python backend/data/generate_data.py   # creates backend/data/retail_campaign.csv
python backend/causal/build_dag.py     # DAG figure + identified estimand
python backend/causal/propensity.py    # overlap and balance plots
pytest backend/tests

# Week 2
python backend/causal/train_dml.py     # trains DML models, writes backend/data/ite_scores.csv
python backend/causal/uplift.py        # Qini curves, deciles, frontend/public/week2_results.json
python backend/causal/refute.py        # DoWhy refutation tests -> docs/refutation_results.txt
```

## Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Open the local URL. On **Setup**, click **Use sample data** and set a budget. The **Uplift results** tab shows the
Week 2 charts (run the Week 2 backend scripts first so `frontend/public/week2_results.json` exists).

## Week 1 findings

On the synthetic data (5,000 customers, true average effect of the discount = +0.125):

- **Naive comparison** of buyers with and without a discount shows +0.280. It is inflated because marketing
  historically discounted loyal, high-spending customers (confounding).
- **Propensity weighting** brings every covariate's standardised mean difference under 0.1
  (`docs/balance_table.csv`) and gives +0.154.
- **Linear adjustment** gives +0.161. Neither is exact, because the true effect varies by customer and the
  relationships are not linear. That gap is the reason for Double ML in Week 2.

## Week 2 findings

Held-out test set of 1,500 customers. Extra purchases when the discount goes to the best-ranked 30% (ground truth):

| Targeting method | Extra purchases |
|---|---|
| Causal forest (DML) | 93.6 |
| Linear DML | 96.9 |
| Random rollout | 57.0 |
| Predictive baseline (likely buyers) | 44.0 |
| Oracle (true effect) | 101.8 |

- Double ML ranks customers close to the oracle, and far above random.
- Targeting likely buyers is worse than random, because they would have bought anyway.
- LinearDML's average effect is +0.146 (95% CI +0.112 to +0.180); the true value is +0.127.
- DoWhy refutation tests pass: a placebo treatment gives about 0, and a random extra confounder or 80% data
  subsets leave the estimate unchanged (`docs/refutation_results.txt`).

## Git workflow (required by Infotact)

- `main` holds merged code only; the Team Lead reviews and merges.
- Each member commits to their own branch (for example `member-a`, `member-b`).
- Commit meaningful changes at least daily. Empty or single-character commits do not count.
