"""
EconoCausal REST API (Week 4).

Wraps the causal engine so other systems (the dashboard, a CRM, a notebook) can use it
over HTTP.

Start it from the repo root:
    uvicorn backend.api.main:app --reload --port 8000
Then open http://localhost:8000/docs for an interactive page where you can try every endpoint.

Endpoints
---------
GET  /health          is the service up, and which model is loaded
GET  /model           model details: features, test-set accuracy, drift thresholds
POST /predict         estimated discount effect and no-discount purchase chance per customer
POST /allocate        best discount for each customer within a budget (the optimizer)
POST /drift           compare a batch of customers with the training data and warn if it shifted
"""

import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from backend.api.model import CausalModel  # noqa: E402
from backend.monitoring.drift import PSI_ALERT, PSI_WARN, detect_drift  # noqa: E402
from backend.optimize.engine import (TIERS, blanket_rollout, evaluate, gain_matrix,  # noqa: E402
                                     likely_buyer_rollout, solve, summary_sentence)

MAX_CUSTOMERS = 5000

app = FastAPI(title="EconoCausal API", version="1.0",
              description="Estimate who a discount actually changes, then allocate a budget to them.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = CausalModel.load()


class Customer(BaseModel):
    customer_id: int
    past_spend: float = Field(ge=0)
    tenure_months: float = Field(ge=0)
    visits_per_month: float = Field(ge=0)
    price_sensitivity: float = Field(ge=0, le=1)
    region: int = Field(ge=0)


class Batch(BaseModel):
    customers: List[Customer] = Field(min_length=1, max_length=MAX_CUSTOMERS)


class AllocateRequest(Batch):
    budget: float = Field(gt=0, le=1_000_000)
    tiers: Optional[List[float]] = Field(default=None, description="Discount sizes in dollars, default [10, 20]")


def to_frame(batch: Batch) -> pd.DataFrame:
    return pd.DataFrame([c.model_dump() for c in batch.customers])


@app.get("/health")
def health():
    return {"status": "ok", "model": model.artifact["name"], "version": model.artifact["version"]}


@app.get("/model")
def model_info():
    return {
        "name": model.artifact["name"],
        "version": model.artifact["version"],
        "features": model.features,
        "metrics": model.artifact["metrics"],
        "drift_thresholds": {"warning": PSI_WARN, "alert": PSI_ALERT},
        "max_customers_per_request": MAX_CUSTOMERS,
    }


@app.post("/predict")
def predict(batch: Batch):
    df = to_frame(batch)
    effect, baseline = model.effect(df), model.baseline(df)
    return {"predictions": [
        {"customer_id": int(cid), "effect": round(float(e), 4), "baseline_purchase_chance": round(float(b), 4)}
        for cid, e, b in zip(df["customer_id"], effect, baseline)
    ]}


@app.post("/allocate")
def allocate(req: AllocateRequest):
    tiers = tuple(sorted(set(req.tiers))) if req.tiers else TIERS
    if any(t <= 0 or t > 100 for t in tiers):
        raise HTTPException(status_code=422, detail="Each discount tier must be between 0 and 100 dollars.")

    df = to_frame(req)
    effect, baseline = model.effect(df), model.baseline(df)
    value = df["past_spend"].to_numpy()

    best = solve(gain_matrix(effect, baseline, value, tiers), tiers, req.budget)
    scored = {
        "optimizer": evaluate(best, effect, baseline, value),
        "blanket": evaluate(blanket_rollout(len(df), req.budget), effect, baseline, value),
        "likely_buyers": evaluate(likely_buyer_rollout(baseline, req.budget), effect, baseline, value),
    }
    gains = gain_matrix(effect, baseline, value, tiers)
    index = {t: i for i, t in enumerate(tiers)}
    rows = [{
        "customer_id": int(df["customer_id"].iloc[i]), "discount": float(best[i]), "effect": round(float(effect[i]), 4),
        "baseline": round(float(baseline[i]), 4), "order_value": round(float(value[i]), 2),
        "expected_net_gain": round(float(gains[index[best[i]], i]) if best[i] > 0 else 0.0, 2),
    } for i in range(len(df))]
    return {
        "budget": req.budget, "tiers": list(tiers), "n_customers": len(df),
        "strategies": scored,
        "summary": summary_sentence(scored["optimizer"], scored["blanket"], req.budget),
        "rows": rows,
    }


@app.post("/drift")
def drift(batch: Batch):
    return detect_drift(model.artifact["reference"], to_frame(batch)[model.features])
