# EconoCausal REST API

The API wraps the causal engine so other tools can use it over HTTP.

## Start it

```bash
pip install -r backend/api/requirements-api.txt     # or backend/requirements.txt for everything
uvicorn backend.api.main:app --reload --port 8000   # run from the repo root
```

Open http://localhost:8000/docs for an interactive page where every endpoint can be tried.

With Docker (not tested in the project environment):

```bash
docker build -t econocausal-api .
docker run -p 8000:8000 econocausal-api
```

## Model file

The API loads `backend/api/model_artifact.json` (about 3 KB), created by `python backend/causal/export_model.py`.
It holds the LinearDML effect coefficients, a baseline purchase model, a summary of the training data for drift
checks, and test-set metrics. Re-run the export script after retraining.

## Customer fields

Every request that takes customers expects these fields for each one:

| Field | Type | Rule |
|---|---|---|
| customer_id | integer | any |
| past_spend | number | 0 or more |
| tenure_months | number | 0 or more |
| visits_per_month | number | 0 or more |
| price_sensitivity | number | between 0 and 1 |
| region | integer | 0 or more |

A request can hold between 1 and 5,000 customers. Invalid input returns HTTP 422 with an explanation.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | /health | Is the service up, and which model is loaded |
| GET | /model | Features, test-set accuracy and drift thresholds |
| POST | /predict | Estimated discount effect and no-discount purchase chance per customer |
| POST | /allocate | Best discount for each customer within a budget |
| POST | /drift | Compare a batch of customers with the training data |

### POST /predict

```json
{"customers": [{"customer_id": 1, "past_spend": 120, "tenure_months": 12,
                "visits_per_month": 5, "price_sensitivity": 0.8, "region": 2}]}
```
Response:
```json
{"predictions": [{"customer_id": 1, "effect": 0.2925, "baseline_purchase_chance": 0.2422}]}
```
`effect` is the estimated rise in purchase probability from a full ($20) discount.

### POST /allocate

Request body is the same customer list plus `budget` (dollars) and optional `tiers` (discount sizes, default `[10, 20]`).
The response has the optimizer's result and two comparison strategies (`blanket`, `likely_buyers`), a plain-English
`summary`, and one row per customer with the chosen `discount` and `expected_net_gain`.

### POST /drift

Same customer list. Returns an `overall` status (`ok`, `warning` or `alert`) and a PSI score per feature.
PSI below 0.10 is stable, 0.10 to 0.25 is a warning, and above 0.25 is an alert. Batches under 500 customers
return a note because PSI is noisy on small samples.

## Example with curl

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"customers":[{"customer_id":1,"past_spend":120,"tenure_months":12,"visits_per_month":5,"price_sensitivity":0.8,"region":2}]}'
```
