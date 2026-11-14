"""Validaciones de datos entre capas (C7), al estilo Great Expectations."""

from __future__ import annotations

import pandas as pd

from gasolina_mlops.features import PRICE_COLS
from gasolina_mlops.layers import PRECIO_MAX, PRECIO_MIN
from gasolina_mlops.simulate import COMBUSTIBLES, DEPARTAMENTOS


class DataValidationError(Exception):
    pass


def validate_ag(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    required = ["foto_id", "estacion_id", "departamento", "fecha_captura", "semana",
                "combustible", "precio"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return [f"Columnas faltantes: {missing}"]
    if df.empty:
        return ["La capa ag está vacía"]
    for col in required:
        if (n := int(df[col].isna().sum())):
            errors.append(f"{col} tiene {n} nulos")
    if df.duplicated(["foto_id", "combustible"]).any():
        errors.append("Hay lecturas duplicadas (foto_id, combustible)")
    if (bad := int((~df["precio"].between(PRECIO_MIN, PRECIO_MAX)).sum())):
        errors.append(f"{bad} precios fuera de [{PRECIO_MIN}, {PRECIO_MAX}]")
    if (unk := set(df["combustible"]) - set(COMBUSTIBLES)):
        errors.append(f"Combustibles desconocidos: {sorted(unk)}")
    if (unk := set(df["departamento"]) - set(DEPARTAMENTOS)):
        errors.append(f"Departamentos desconocidos: {sorted(unk)}")
    return errors


def validate_au_modelo(df: pd.DataFrame, min_rows: int = 50) -> list[str]:
    errors: list[str] = []
    if len(df) < min_rows:
        errors.append(f"Muy pocas filas para entrenar: {len(df)} < {min_rows}")
    if df[PRICE_COLS].isna().any().any():
        errors.append("Hay precios nulos en las features")
    if "sube" in df and not set(df["sube"].unique()) <= {0, 1}:
        errors.append("'sube' debe ser 0/1")
    if "sube" in df and df["sube"].nunique() < 2:
        errors.append("El objetivo tiene una sola clase")
    return errors


def assert_valid(errors: list[str]) -> None:
    if errors:
        raise DataValidationError("; ".join(errors))
