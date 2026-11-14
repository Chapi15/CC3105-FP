"""Arquitectura medallón (C2, C4): raw -> cusn (bronce) -> ag (plata) -> au (oro).

La lógica es la misma que en los notebooks de Databricks (carpeta databricks/), escrita en
pandas para poder probarla localmente y en CI.
"""

from __future__ import annotations

import json
import unicodedata
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from gasolina_mlops.features import LAGS, TARGET
from gasolina_mlops.simulate import COMBUSTIBLES, DEPARTAMENTOS
from gasolina_mlops.storage import list_files, read_text

PRECIO_MIN, PRECIO_MAX = 15.0, 60.0


def _sin_tildes(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


_DEPTO_CANON = {_sin_tildes(d).lower(): d for d in DEPARTAMENTOS}


# ---------------------------------------------------------------- cusn (bronce)
def to_cusn(raw_uri: str) -> pd.DataFrame:
    """Aplana cada JSON (una fila por lectura) sin transformar valores."""
    rows = []
    ingested = datetime.now(UTC).isoformat()
    for path in list_files(raw_uri, "fecha=*/*.json"):
        foto = json.loads(read_text(path))
        for lectura in foto.get("lecturas", []):
            rows.append(
                {
                    **{k: v for k, v in foto.items() if k != "lecturas"},
                    "combustible": lectura.get("combustible"),
                    "precio_texto": lectura.get("precio"),
                    "_source_file": path,
                    "_ingested_at": ingested,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- ag (plata)
def parse_precio(texto: str | None) -> float:
    if not texto:
        return np.nan
    limpio = str(texto).upper().replace("Q", "").replace(" ", "").replace(",", ".")
    try:
        return float(limpio)
    except ValueError:
        return np.nan


def normalizar_combustible(texto: str | None) -> str | None:
    if texto is None:
        return None
    t = _sin_tildes(str(texto)).strip().lower()
    return t if t in COMBUSTIBLES else None


def normalizar_departamento(texto: str | None) -> str | None:
    if texto is None:
        return None
    return _DEPTO_CANON.get(_sin_tildes(str(texto)).strip().lower())


def to_ag(cusn: pd.DataFrame) -> pd.DataFrame:
    df = cusn.drop(columns=[c for c in cusn.columns if c.startswith("_")]).copy()
    df["combustible"] = df["combustible"].map(normalizar_combustible)
    df["departamento"] = df["departamento"].map(normalizar_departamento)
    df["precio"] = df["precio_texto"].map(parse_precio)
    df["fecha_captura"] = pd.to_datetime(df["fecha_captura"])
    df["semana"] = df["fecha_captura"].dt.to_period("W-SUN").dt.start_time
    df = df.drop_duplicates(subset=["foto_id", "combustible"])
    df = df.dropna(subset=["combustible", "departamento", "precio"])
    df = df[df["precio"].between(PRECIO_MIN, PRECIO_MAX)]
    cols = ["foto_id", "archivo", "estacion_id", "marca", "departamento", "fecha_captura",
            "semana", "combustible", "precio"]
    return df[cols].sort_values(["fecha_captura", "estacion_id"]).reset_index(drop=True)


# ---------------------------------------------------------------- au (oro)
def precios_semanales(ag: pd.DataFrame) -> pd.DataFrame:
    return (
        ag.groupby(["semana", "departamento", "combustible"])
        .agg(precio_promedio=("precio", "mean"), n_fotos=("foto_id", "nunique"))
        .round({"precio_promedio": 3})
        .reset_index()
    )


def build_series_features(semanal: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (semana, departamento, combustible) con los precios de las últimas
    semanas (precio_t0 = semana actual) y el objetivo: ¿la próxima semana sube?"""
    frames = []
    semanas = pd.date_range(semanal["semana"].min(), semanal["semana"].max(), freq="W-MON")
    for (depto, comb), g in semanal.groupby(["departamento", "combustible"]):
        s = g.set_index("semana")["precio_promedio"].reindex(semanas)
        s = s.interpolate(limit=2, limit_area="inside")  # rellena huecos cortos sin fotos
        f = pd.DataFrame({"semana": semanas, "departamento": depto, "combustible": comb})
        for lag in range(LAGS):
            f[f"precio_t{lag}"] = s.shift(lag).to_numpy()
        siguiente = s.shift(-1).to_numpy()
        f["precio_siguiente"] = siguiente
        f[TARGET] = np.where(np.isnan(siguiente), np.nan, (siguiente > f["precio_t0"]))
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=[f"precio_t{i}" for i in range(LAGS)])
    return out.reset_index(drop=True)


def to_au(ag: pd.DataFrame) -> dict[str, pd.DataFrame]:
    semanal = precios_semanales(ag)
    series = build_series_features(semanal)
    modelo = series.dropna(subset=[TARGET]).astype({TARGET: int})
    # La última semana no tiene objetivo todavía: es lo que se predice en producción.
    ultima = series[series["semana"] == series["semana"].max()].drop(columns=[TARGET])
    nacional = (
        ag.groupby(["semana", "combustible"])["precio"].mean().round(3).rename("precio_promedio")
        .reset_index()
    )
    return {
        "precios_semanales": semanal,
        "modelo": modelo,
        "ultima_semana": ultima,
        "precio_nacional": nacional,
    }
