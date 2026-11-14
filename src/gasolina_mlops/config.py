"""Carga de configuración. Variables de entorno sobrescriben valores del YAML."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("configs/config.yaml")


def load_config(path: str | Path | None = None, base_uri: str | None = None) -> dict[str, Any]:
    path = Path(path or os.getenv("GASOLINA_CONFIG", DEFAULT_CONFIG))
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    override = base_uri or os.getenv("GASOLINA_BASE_URI")
    if override:
        cfg["storage"]["base_uri"] = override
    return cfg


def layer_uri(cfg: dict[str, Any], layer: str, *parts: str) -> str:
    """Ruta de una capa medallón: <base_uri>/<capa>/<parts...> (local o s3://)."""
    base = cfg["storage"]["base_uri"].rstrip("/")
    return "/".join([base, cfg["storage"]["layers"][layer], *parts])
