"""Monitoreo (C8): drift de datos con PSI y resumen operativo del log de la API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gasolina_mlops.features import CATEGORICAL, add_derived_features

MONITORED_NUMERIC = ["precio_t0", "cambio_1", "rango_4"]
# El nivel de precio es una caminata aleatoria: siempre "deriva". Se reporta, pero no alerta
# (si sube mucho, es señal para re-entrenar, no un error de datos).
INFORMATIVE = ["precio_t0"]
EPS = 1e-6


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref, cur = reference.dropna().to_numpy(), current.dropna().to_numpy()
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, edges)[0] / max(len(ref), 1) + EPS
    cur_pct = np.histogram(cur, edges)[0] / max(len(cur), 1) + EPS
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    cats = sorted(set(reference.dropna()) | set(current.dropna()))
    ref_pct = reference.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy() + EPS
    cur_pct = current.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy() + EPS
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def nivel(psi: float, aviso: float, alerta: float) -> str:
    return "alerta" if psi > alerta else "aviso" if psi > aviso else "ok"


def drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, aviso: float, alerta: float
) -> dict:
    ref, cur = add_derived_features(reference), add_derived_features(current)
    psi = {c: psi_numeric(ref[c], cur[c]) for c in MONITORED_NUMERIC}
    psi |= {c: psi_categorical(ref[c], cur[c]) for c in CATEGORICAL}
    psi = {k: round(v, 4) for k, v in psi.items()}
    niveles = {k: nivel(v, aviso, alerta) for k, v in psi.items()}
    evaluadas = [k for k in psi if k not in INFORMATIVE]
    alertas = [k for k in evaluadas if niveles[k] == "alerta"]
    avisos = [k for k in evaluadas if niveles[k] == "aviso"]
    return {
        "n_reference": len(ref),
        "n_current": len(cur),
        "psi_aviso": aviso,
        "psi_alerta": alerta,
        "psi": psi,
        "nivel": niveles,
        "informativas": INFORMATIVE,
        "estado": "alerta" if alertas else "aviso" if avisos else "ok",
        "drifted_features": alertas,
        "drift_detected": bool(alertas),
    }


def predictions_summary(log_path: Path) -> dict[str, Any] | None:
    if not log_path.exists():
        return None
    rows = [json.loads(x) for x in log_path.read_text(encoding="utf-8").splitlines() if x]
    if not rows:
        return None
    lat = np.array([r["latency_ms"] for r in rows])
    probas = np.concatenate([r["probabilidades"] for r in rows])
    return {
        "requests": len(rows),
        "predicciones": int(len(probas)),
        "latencia_ms_p50": round(float(np.percentile(lat, 50)), 2),
        "latencia_ms_p95": round(float(np.percentile(lat, 95)), 2),
        "probabilidad_sube_promedio": round(float(probas.mean()), 4),
    }


def render_markdown(report: dict, ops: dict | None) -> str:
    estado = {
        "ok": "✅ Sin drift significativo",
        "aviso": "🟡 Aviso: distribución distinta, revisar",
        "alerta": "🔴 ALERTA: drift fuerte, investigar y re-entrenar",
    }[report["estado"]]
    lines = [
        "# Reporte de monitoreo",
        "",
        f"**Estado:** {estado}",
        "",
        f"Umbrales PSI: aviso > {report['psi_aviso']} · alerta > {report['psi_alerta']}. "
        f"Referencia: {report['n_reference']} filas · Actual: {report['n_current']} filas",
        "",
        "| Feature | PSI | Nivel |",
        "|---|---|---|",
        *[
            f"| {k}{' (informativa)' if k in report['informativas'] else ''} | {v} "
            f"| {report['nivel'][k]} |"
            for k, v in report["psi"].items()
        ],
    ]
    if ops:
        lines += ["", "## Servicio (API)", "", "| Métrica | Valor |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in ops.items()]
    return "\n".join(lines) + "\n"
