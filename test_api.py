import pandas as pd
from fastapi.testclient import TestClient

from backend.api.main import app

client = TestClient(app)


def customers(n=60, shift=1.0):
    df = pd.read_csv("backend/data/ite_scores.csv").head(n)
    cols = ["customer_id", "past_spend", "tenure_months", "visits_per_month", "price_sensitivity", "region"]
    out = df[cols].copy()
    out["past_spend"] = out["past_spend"] * shift
    return out.to_dict(orient="records")


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_model_info_lists_features_and_metrics():
    body = client.get("/model").json()
    assert "price_sensitivity" in body["features"]
    assert body["metrics"]["test_correlation_with_true"] > 0.8


def test_predict_matches_exported_scores():
    df = pd.read_csv("backend/data/ite_scores.csv").head(20)
    r = client.post("/predict", json={"customers": customers(20)})
    assert r.status_code == 200
    got = [p["effect"] for p in r.json()["predictions"]]
    assert max(abs(a - b) for a, b in zip(got, df["ite_linear"])) < 0.001


def test_allocate_respects_budget_and_returns_summary():
    r = client.post("/allocate", json={"customers": customers(200), "budget": 500})
    assert r.status_code == 200
    body = r.json()
    assert body["strategies"]["optimizer"]["spend"] <= 500
    assert len(body["rows"]) == 200
    assert "blanket" in body["summary"]
    assert body["strategies"]["optimizer"]["net_revenue"] >= body["strategies"]["blanket"]["net_revenue"]


def test_allocate_rejects_bad_tier():
    r = client.post("/allocate", json={"customers": customers(10), "budget": 100, "tiers": [-5]})
    assert r.status_code == 422


def test_allocate_rejects_missing_field():
    bad = customers(3)
    del bad[0]["region"]
    assert client.post("/allocate", json={"customers": bad, "budget": 100}).status_code == 422


def test_drift_endpoint_flags_shifted_batch():
    ok = client.post("/drift", json={"customers": customers(500)}).json()
    shifted = client.post("/drift", json={"customers": customers(500, shift=2.0)}).json()
    assert ok["overall"] == "ok"
    assert shifted["overall"] == "alert"
