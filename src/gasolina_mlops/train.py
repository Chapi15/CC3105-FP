"""Entrenamiento con split temporal, baseline, quality gate y registro del modelo (C6, C7)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from gasolina_mlops import __version__
from gasolina_mlops.features import FEATURES, TARGET, build_pipeline

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
REFERENCE_FILE = "reference.parquet"


class QualityGateError(Exception):
    pass


def temporal_split(df: pd.DataFrame, test_weeks: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nunca barajar series de tiempo: se entrena con el pasado y se prueba con el futuro."""
    semanas = sorted(df["semana"].unique())
    corte = semanas[-test_weeks]
    return df[df["semana"] < corte], df[df["semana"] >= corte]


def evaluate(model, X: pd.DataFrame, y: pd.Series, threshold: float) -> dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "f1": round(float(f1_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred)), 4),
    }


def baseline_accuracy(X: pd.DataFrame, y: pd.Series) -> float:
    """Regla ingenua: 'si subió esta semana, sube la próxima'. El modelo debe superarla."""
    pred = (X["precio_t0"] > X["precio_t1"]).astype(int)
    return round(float(accuracy_score(y, pred)), 4)


def train(au_modelo: pd.DataFrame, cfg: dict[str, Any], model_dir: Path) -> dict[str, Any]:
    mcfg, seed = cfg["model"], cfg["seed"]
    df = au_modelo.copy()
    df["semana"] = pd.to_datetime(df["semana"])
    tr, te = temporal_split(df, mcfg["test_weeks"])

    pipeline = build_pipeline(mcfg["algorithm"], mcfg.get("params") or {}, seed)
    pipeline.fit(tr[FEATURES], tr[TARGET])
    metrics = evaluate(pipeline, te[FEATURES], te[TARGET], mcfg["threshold"])
    metrics["baseline_accuracy"] = baseline_accuracy(te, te[TARGET])

    if metrics["roc_auc"] < mcfg["min_roc_auc"]:
        raise QualityGateError(
            f"ROC AUC {metrics['roc_auc']} < mínimo requerido {mcfg['min_roc_auc']}"
        )
    if mcfg.get("must_beat_baseline") and metrics["accuracy"] <= metrics["baseline_accuracy"]:
        raise QualityGateError(
            f"accuracy {metrics['accuracy']} no supera al baseline {metrics['baseline_accuracy']}"
        )

    now = datetime.now(UTC)
    metadata = {
        "model_version": now.strftime("%Y%m%d%H%M%S"),
        "package_version": __version__,
        "trained_at": now.isoformat(),
        "algorithm": mcfg["algorithm"],
        "params": mcfg.get("params") or {},
        "threshold": mcfg["threshold"],
        "features": FEATURES,
        "train_weeks": [str(tr["semana"].min().date()), str(tr["semana"].max().date())],
        "test_weeks": [str(te["semana"].min().date()), str(te["semana"].max().date())],
        "n_train": len(tr),
        "n_test": len(te),
        "train_rate_sube": round(float(tr[TARGET].mean()), 4),
        "metrics": metrics,
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / MODEL_FILE)
    (model_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    tr[FEATURES].to_parquet(model_dir / REFERENCE_FILE, index=False)  # referencia para drift
    return metadata


def write_model_card(meta: dict[str, Any], path: Path) -> Path:
    m = meta["metrics"]
    lines = [
        f"# Model card — gasolina {meta['model_version']}",
        "",
        "## Uso previsto",
        "Estimar si el precio promedio semanal de un combustible en un departamento sube la",
        "próxima semana. Es un ejemplo con datos simulados: no usar para decidir nada real.",
        "",
        "## Entrenamiento",
        f"- Algoritmo: `{meta['algorithm']}` · Hiperparámetros: `{json.dumps(meta['params'])}`",
        f"- Semanas de entrenamiento: {meta['train_weeks'][0]} a {meta['train_weeks'][1]}"
        f" ({meta['n_train']} filas)",
        f"- Semanas de prueba: {meta['test_weeks'][0]} a {meta['test_weeks'][1]}"
        f" ({meta['n_test']} filas)",
        f"- Proporción de semanas que suben (entrenamiento): {meta['train_rate_sube']:.1%}",
        f"- Features de entrada: {', '.join(f'`{f}`' for f in meta['features'])}",
        "",
        "## Métricas (prueba, split temporal)",
        "| Métrica | Valor |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in m.items()],
        "",
        "`baseline_accuracy` es la regla ingenua “si subió esta semana, sube la próxima”.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
