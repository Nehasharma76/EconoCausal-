"""
Loads the exported causal model (backend/api/model_artifact.json) and scores customers.

The artifact is a small JSON file made by backend/causal/export_model.py. It holds:
  * effect model    LinearDML: effect = intercept + sum(coef * feature)
  * baseline model  logistic regression: chance of buying with NO discount
  * reference       summary of the training data, used by the drift detector
  * metrics         how well the model did on held-out customers

Storing plain numbers instead of a pickled model keeps the file tiny, readable, and safe
to load on any machine.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "backend" / "api" / "model_artifact.json"


class CausalModel:
    def __init__(self, artifact: dict):
        self.artifact = artifact
        self.features = artifact["features"]
        self._coef = np.array(artifact["effect_model"]["coef"], dtype=float)
        self._intercept = float(artifact["effect_model"]["intercept"])
        base = artifact["baseline_model"]
        self._b_mean = np.array(base["feature_means"], dtype=float)
        self._b_scale = np.array(base["feature_scales"], dtype=float)
        self._b_coef = np.array(base["coef"], dtype=float)
        self._b_intercept = float(base["intercept"])

    @classmethod
    def load(cls, path: Path = ARTIFACT) -> "CausalModel":
        return cls(json.loads(Path(path).read_text()))

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {', '.join(missing)}")
        return df[self.features].to_numpy(dtype=float)

    def effect(self, df: pd.DataFrame) -> np.ndarray:
        """Estimated rise in purchase probability from a full discount, per customer (the ITE)."""
        return self._matrix(df) @ self._coef + self._intercept

    def baseline(self, df: pd.DataFrame) -> np.ndarray:
        """Estimated chance of buying with no discount, per customer."""
        z = ((self._matrix(df) - self._b_mean) / self._b_scale) @ self._b_coef + self._b_intercept
        return 1.0 / (1.0 + np.exp(-z))
