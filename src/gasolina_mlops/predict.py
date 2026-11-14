"""Carga del modelo e inferencia por lotes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from gasolina_mlops.features import FEATURES
from gasolina_mlops.train import METADATA_FILE, MODEL_FILE


def load_model(model_dir: Path) -> tuple[Any, dict[str, Any]]:
    model = joblib.load(model_dir / MODEL_FILE)
    metadata = json.loads((model_dir / METADATA_FILE).read_text(encoding="utf-8"))
    return model, metadata


def predict(model, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    proba = model.predict_proba(df[FEATURES])[:, 1]
    keep = [c for c in ("semana", "departamento", "combustible", "precio_t0") if c in df]
    out = df[keep].copy()
    out["probabilidad_sube"] = proba.round(4)
    out["prediccion"] = ["sube" if p >= threshold else "baja" for p in proba]
    return out
