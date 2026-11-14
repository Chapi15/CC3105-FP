"""Features y pipeline de scikit-learn (C2). Todas las transformaciones viven dentro del
Pipeline para que entrenamiento e inferencia apliquen exactamente lo mismo."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

LAGS = 4  # precio de la semana actual + 3 semanas anteriores
PRICE_COLS = [f"precio_t{i}" for i in range(LAGS)]
CATEGORICAL = ["departamento", "combustible"]
FEATURES = CATEGORICAL + PRICE_COLS
DERIVED = ["cambio_1", "cambio_2", "cambio_3", "desv_media_4", "rango_4"]
TARGET = "sube"


def add_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for i in range(1, LAGS):
        X[f"cambio_{i}"] = X[f"precio_t{i - 1}"] - X[f"precio_t{i}"]
    precios = X[PRICE_COLS]
    X["desv_media_4"] = X["precio_t0"] - precios.mean(axis=1)
    X["rango_4"] = precios.max(axis=1) - precios.min(axis=1)
    return X


def build_pipeline(algorithm: str, params: dict[str, Any], seed: int) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), ["precio_t0", *DERIVED]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    if algorithm == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=seed, **params)
    elif algorithm == "random_forest":
        model = RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    elif algorithm == "logistic_regression":
        model = LogisticRegression(max_iter=1000, **params)
    else:
        raise ValueError(f"Algoritmo no soportado: {algorithm}")
    return Pipeline(
        [
            ("derive", FunctionTransformer(add_derived_features)),
            ("preprocess", preprocess),
            ("model", model),
        ]
    )
