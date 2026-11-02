# Week 4 report: REST API, drift detection and dashboard polish

## Deliverables

- [x] REST API wrapping the causal engine (`backend/api/main.py`) with docs at `/docs`
- [x] Model packaged as a small JSON file (`backend/causal/export_model.py` -> `backend/api/model_artifact.json`)
- [x] Automated data drift detector (`backend/monitoring/drift.py`) and demo (`backend/monitoring/simulate_drift.py`)
- [x] Dashboard "Monitoring" tab, live budget recalculation through the API, and plain-language summaries
- [x] Dockerfile, API guide (`docs/API.md`), and tests for the API, drift detector and optimizer (45 tests in total)

## REST API

| Endpoint | Purpose |
|---|---|
| GET /health | Service status and model version |
| GET /model | Features, test-set accuracy, drift thresholds |
| POST /predict | Effect and no-discount purchase chance per customer |
| POST /allocate | Optimal discounts within a budget, with comparison strategies and a summary sentence |
| POST /drift | Drift check of a new batch against the training data |

The model is saved as plain numbers (3.4 KB): the LinearDML coefficients, the baseline model, and a summary of the
training data. Using LinearDML for the API is a deliberate choice: it ranked customers best in Week 2 (correlation
0.86 with the true effect), and a linear model needs no heavy pickled files. The causal forest is still used for offline evaluation.

## Drift detector

The Population Stability Index compares each feature of a new batch with the training data. Below 0.10 is stable,
0.10 to 0.25 is a warning, and above 0.25 is an alert.

| Batch | Result | Detail |
|---|---|---|
| No drift (held-out customers) | OK | largest PSI 0.013 |
| Mild drift (35% higher spend) | WARNING | past spend PSI 0.134 |
| Strong drift (50% higher spend, 2 more visits, region mix shifted) | ALERT | visits 0.719, region 0.623, past spend 0.252 |

The detector stays quiet on normal data and escalates as the shift grows. The same calculation runs in the browser
(`frontend/src/psi.js`) so the Monitoring tab can check an uploaded CSV, and it matches the Python version exactly on the
sample data. Batches under 500 rows show a note because PSI is noisy on small samples.

## Dashboard polish

- **Allocation tab:** opens with a plain-English summary such as "With a $5,000 budget, this strategy gives discounts to 481 customers... and is expected to add $4,643 in net revenue." When the API is running, a budget box recalculates with any budget; otherwise it shows the saved $5,000 result.
- **Monitoring tab:** traffic-light banner, PSI chart with threshold lines, per-feature table, and a selector to check the uploaded data or the demo batches.
- Navigation, tables and charts share one visual style and work on narrow screens.

## Limitations

- Drift detection looks at the customer features, not at whether the discount effect itself changed. Checking effect stability needs fresh experiment data.
- The Dockerfile has not been built in this project environment.
- The dashboard was tested with automated component tests and a build, not in a real browser by hand.
- LinearDML is used by the API, so the allocation reflects a linear effect model.

## Possible extensions

Retrain automatically when drift is flagged, validate with a small randomized A/B test, and add authentication to the API.
