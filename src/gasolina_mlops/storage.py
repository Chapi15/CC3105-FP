"""Acceso a archivos local o en S3 con la misma interfaz.

Para S3 se usa fsspec/s3fs (`uv sync --extra s3`); las credenciales se leen del entorno
(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY), nunca del código.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def is_remote(uri: str) -> bool:
    return "://" in uri


def _fs(uri: str):
    import fsspec  # se instala con el extra s3

    return fsspec.core.url_to_fs(uri)[0]


def list_files(uri: str, pattern: str) -> list[str]:
    if is_remote(uri):
        protocol = uri.split("://", 1)[0]
        return sorted(f"{protocol}://{p}" for p in _fs(uri).glob(f"{uri.rstrip('/')}/{pattern}"))
    return sorted(str(p) for p in Path(uri).glob(pattern))


def read_text(path: str) -> str:
    if is_remote(path):
        with _fs(path).open(path, "r", encoding="utf-8") as f:
            return f.read()
    return Path(path).read_text(encoding="utf-8")


def write_text_atomic(path: str, text: str) -> None:
    """Escribe en .tmp y renombra, para que un sync nunca suba un archivo a medias."""
    if is_remote(path):
        with _fs(path).open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def write_parquet(df: pd.DataFrame, path: str) -> str:
    if not is_remote(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
