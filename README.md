# EconoCausal: Dynamic Pricing via Double Machine Learning

Standard ML predicts who is likely to buy. EconoCausal estimates **how much a discount changes each customer's
chance of buying** (the Individual Treatment Effect), then spends a fixed budget on the customers the discount
actually moves ("Persuadables").

Infotact Solutions Advanced Data Science & ML Engineering (Vol. II), Project 2.

## Project status

| Week | Backend (EconML, DoWhy) | Frontend (React, Plotly) | Status |
|---|---|---|---|
| 1 | Causal DAG, synthetic data, propensity balancing | Upload and budget views | Done |
| 2 | Double ML training (ITE) | Qini and uplift curves | Planned |
| Mid review | DoWhy refutation tests | Large ITE data load | Planned |
| 3 | SciPy budget optimizer | Allocation table | Planned |
| 4 | REST API, drift detector | Polish, plain-language summaries | Planned |

## Repository layout

```
backend/
  data/generate_data.py     synthetic retail data with a known true effect
  causal/build_dag.py       DoWhy DAG, effect identification, naive vs adjusted ATE
  causal/propensity.py      propensity scores, overlap and balance plots, IPW
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
```

## Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Open the local URL, click **Use sample data**, and set a budget.

## Week 1 findings

On the synthetic data (5,000 customers, true average effect of the discount = +0.125):

- **Naive comparison** of buyers with and without a discount shows +0.280. It is inflated because marketing
  historically discounted loyal, high-spending customers (confounding).
- **Propensity weighting** brings every covariate's standardised mean difference under 0.1
  (`docs/balance_table.csv`) and gives +0.154.
- **Linear adjustment** gives +0.161. Neither is exact, because the true effect varies by customer and the
  relationships are not linear. That gap is the reason for Double ML in Week 2.

## Git workflow (required by Infotact)

- `main` holds merged code only; the Team Lead reviews and merges.
- Each member commits to their own branch (for example `member-a`, `member-b`).
- Commit meaningful changes at least daily. Empty or single-character commits do not count.
