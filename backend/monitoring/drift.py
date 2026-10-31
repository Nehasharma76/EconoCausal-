"""
Data drift detection (Week 4).

A causal model is only trusted while new customers look like the customers it was
trained on. If customer behaviour changes (for example people start spending more,
or the mix of regions shifts), the model's effect estimates can become wrong. This
module compares a NEW batch of customers with a stored reference of the training
data and raises a warning when they differ.

Method: Population Stability Index (PSI), a standard drift score in finance and
marketing analytics.
  1. Split each feature into bins using the TRAINING data (deciles for numbers,
     one bin per value for categories like region).
  2. Count what share of the training data (expected) and the new batch (actual)
     falls in each bin.
  3. PSI = sum over bins of (actual - expected) * ln(actual / expected)

Common reading of PSI for one feature:
  below 0.10        stable
  0.10 to 0.25      warning  (some shift, keep an eye on it)
  above 0.25        alert    (large shift, retrain or investigate)

We also report the shift in the feature's average, measured in training standard
deviations, because it is easy to explain ("past spend is up 0.6 standard deviations").

Very small batches give noisy PSI values, so a note is added below MIN_ROWS rows.
"""

import numpy as np
import pandas as pd

EPS = 1e-4          # avoids log(0) when a bin is empty
PSI_WARN = 0.10
PSI_ALERT = 0.25
MIN_ROWS = 500      # below this, PSI is noisy
CATEGORICAL_MAX_VALUES = 6


def build_reference(df: pd.DataFrame, features, bins: int = 10) -> dict:
    """Summarise the training data so later batches can be compared without the raw data."""
    reference = {}
    for f in features:
        x = df[f].to_numpy(dtype=float)
        values, counts = np.unique(x, return_counts=True)
        if len(values) <= CATEGORICAL_MAX_VALUES:
            reference[f] = {"type": "categorical", "values": values.tolist(), "proportions": (counts / len(x)).tolist()}
        else:
            edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))[1:-1]  # inner cut points
            reference[f] = {
                "type": "numeric",
                "edges": edges.tolist(),
                "proportions": (np.bincount(np.digitize(x, edges), minlength=len(edges) + 1) / len(x)).tolist(),
                "mean": float(x.mean()),
                "std": float(x.std()),
            }
    reference["_n"] = int(len(df))
    return reference


def psi(expected, actual) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status_for(value: float) -> str:
    if value >= PSI_ALERT:
        return "alert"
    if value >= PSI_WARN:
        return "warning"
    return "ok"


def feature_report(ref: dict, x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    if ref["type"] == "categorical":
        known = np.array(ref["values"])
        actual = [float(np.mean(x == v)) for v in known]
        actual.append(float(np.mean(~np.isin(x, known))))        # categories never seen in training
        expected = list(ref["proportions"]) + [0.0]
        shift = None
    else:
        edges = np.array(ref["edges"])
        actual = (np.bincount(np.digitize(x, edges), minlength=len(edges) + 1) / len(x)).tolist()
        expected = ref["proportions"]
        shift = float((x.mean() - ref["mean"]) / ref["std"]) if ref["std"] else 0.0
    value = psi(expected, actual)
    return {"psi": round(value, 4), "status": status_for(value),
            "mean_shift_std": None if shift is None else round(shift, 3)}


def detect_drift(reference: dict, df: pd.DataFrame) -> dict:
    """Compare a new batch with the reference. Returns per-feature scores and an overall status."""
    rows = []
    for f, ref in reference.items():
        if f.startswith("_"):
            continue
        if f not in df.columns:
            raise ValueError(f"New data is missing column: {f}")
        rows.append({"feature": f, **feature_report(ref, df[f].to_numpy())})
    order = {"ok": 0, "warning": 1, "alert": 2}
    overall = max((r["status"] for r in rows), key=order.get) if rows else "ok"
    report = {"n_rows": int(len(df)), "overall": overall, "features": rows, "notes": []}
    if len(df) < MIN_ROWS:
        report["notes"].append(f"Only {len(df)} rows: PSI is noisy below {MIN_ROWS} rows, treat results as a rough guide.")
    return report
