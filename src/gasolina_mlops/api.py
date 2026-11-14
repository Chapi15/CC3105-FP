"""API REST del modelo (C3, C6). Es lo que corre en DigitalOcean ("producción").

Local: uv run uvicorn gasolina_mlops.api:app --reload  ->  http://localhost:8000/docs
Variables: MODEL_DIR (default models), PREDICTIONS_LOG (default logs/predictions.jsonl)
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from gasolina_mlops.features import LAGS
from gasolina_mlops.layers import PRECIO_MAX, PRECIO_MIN
from gasolina_mlops.predict import load_model, predict
from gasolina_mlops.simulate import DEPARTAMENTOS

Departamento = Literal[tuple(DEPARTAMENTOS)]  # type: ignore[valid-type]
Precio = Annotated[float, Field(ge=PRECIO_MIN, le=PRECIO_MAX)]


class Serie(BaseModel):
    departamento: Departamento
    combustible: Literal["super", "regular", "diesel"]
    precios: list[Precio] = Field(
        min_length=LAGS,
        max_length=LAGS,
        description=f"Precio promedio (Q/galón) de las últimas {LAGS} semanas, "
        "de la más antigua a la más reciente.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "departamento": "Petén",
                "combustible": "diesel",
                "precios": [29.85, 30.02, 30.31, 30.64],
            }
        }
    }


class PredictRequest(BaseModel):
    series: list[Serie] = Field(min_length=1, max_length=500)


class Prediccion(BaseModel):
    departamento: str
    combustible: str
    precio_actual: float
    probabilidad_sube: float
    prediccion: Literal["sube", "baja"]


class PredictResponse(BaseModel):
    model_version: str
    predicciones: list[Prediccion]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.metadata = load_model(Path(os.getenv("MODEL_DIR", "models")))
    app.state.log_path = Path(os.getenv("PREDICTIONS_LOG", "logs/predictions.jsonl"))
    app.state.log_path.parent.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="¿Sube o baja la gasolina? — CC3105", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict:
    return app.state.metadata


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest) -> PredictResponse:
    start = time.perf_counter()
    rows = []
    for s in req.series:
        row = {"departamento": s.departamento, "combustible": s.combustible}
        # precios llega de antiguo a reciente; precio_t0 es el más reciente.
        row |= {f"precio_t{i}": p for i, p in enumerate(reversed(s.precios))}
        rows.append(row)
    df = pd.DataFrame(rows)
    meta = app.state.metadata
    out = predict(app.state.model, df, meta["threshold"])
    latency_ms = (time.perf_counter() - start) * 1000

    with open(app.state.log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": datetime.now(UTC).isoformat(),
            "model_version": meta["model_version"],
            "n": len(df),
            "latency_ms": round(latency_ms, 3),
            "probabilidades": out["probabilidad_sube"].tolist(),
            "inputs": rows,
        }, ensure_ascii=False) + "\n")

    preds = [
        Prediccion(
            departamento=r["departamento"],
            combustible=r["combustible"],
            precio_actual=r["precio_t0"],
            probabilidad_sube=r["probabilidad_sube"],
            prediccion=r["prediccion"],
        )
        for r in out.to_dict(orient="records")
    ]
    return PredictResponse(model_version=meta["model_version"], predicciones=preds)
