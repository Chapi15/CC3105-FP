"""Simulador de las fotos de rótulos de precios.

En el proyecto real, cada foto de un rótulo de gasolinera se convierte en un JSON con las
lecturas de precio (a mano, con OCR o con un modelo de visión). Este módulo genera JSON
equivalentes con errores típicos de captura para poder probar el pipeline sin fotos reales.

Formato de cada archivo (raw/fotos/fecha=AAAA-MM-DD/<foto_id>.json):
{
  "foto_id": "...", "archivo": "imagenes/fecha=.../<foto_id>.jpg",
  "estacion_id": "GUA-01", "marca": "Volcán", "departamento": "Guatemala",
  "fecha_captura": "2025-01-07T09:15:00",
  "lecturas": [{"combustible": "Súper", "precio": "Q 32.49"}, ...]
}
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from gasolina_mlops.storage import write_text_atomic

DEPARTAMENTOS = {
    # departamento: (código, recargo promedio por transporte en Q/galón)
    "Guatemala": ("GUA", 0.0),
    "Sacatepéquez": ("SAC", 0.1),
    "Escuintla": ("ESC", -0.2),
    "Quetzaltenango": ("QUE", 0.4),
    "Huehuetenango": ("HUE", 0.8),
    "Alta Verapaz": ("AVE", 0.7),
    "Izabal": ("IZA", 0.5),
    "Petén": ("PET", 1.5),
}
COMBUSTIBLES = {"super": 32.0, "regular": 30.5, "diesel": 28.5}
MARCAS = ["Volcán", "Quetzal", "Ceiba", "Maya"]  # marcas ficticias

_ETIQUETAS = {
    "super": ["Súper", "SUPER", "super ", "Super"],
    "regular": ["Regular", "REGULAR", "regular"],
    "diesel": ["Diésel", "DIESEL", "diesel", "Diesel"],
}


def simulate_national_prices(
    start: str, weeks: int, rng: np.random.Generator, shock: bool = False
) -> pd.DataFrame:
    """Precio nacional semanal por combustible: cambios con inercia + choques comunes."""
    semanas = pd.date_range(start, periods=weeks, freq="W-MON")
    rows = []
    common = rng.normal(0, 0.25, weeks)  # choque internacional del petróleo, compartido
    if shock:
        # Últimas 12 semanas: alza fuerte y mercado mucho más volátil.
        common[-12:] = common[-12:] * 3 + 0.4
    for comb, base in COMBUSTIBLES.items():
        price, change = base, 0.0
        for t, semana in enumerate(semanas):
            change = 0.6 * change + common[t] + rng.normal(0, 0.2) - 0.03 * (price - base)
            price = price + change
            rows.append({"semana": semana, "combustible": comb, "precio_nacional": price})
    return pd.DataFrame(rows)


def _texto_precio(precio: float, rng: np.random.Generator) -> str:
    r = rng.random()
    if r < 0.02:
        return ""  # lectura ilegible
    if r < 0.03:
        return f"Q {precio / 10:.3f}"  # punto decimal mal ubicado
    texto = str(rng.choice(["Q {:.2f}", "Q{:.2f}", "{:.2f}"])).format(precio)
    return texto.replace(".", ",") if rng.random() < 0.2 else texto  # coma decimal


def _foto_id(estacion: str, fecha: pd.Timestamp) -> str:
    return hashlib.sha1(f"{estacion}{fecha.isoformat()}".encode()).hexdigest()[:12]


def generate_photos(cfg: dict, seed: int, shock: bool = False) -> list[dict]:
    sim = cfg["simulation"]
    rng = np.random.default_rng(seed)
    nacional = simulate_national_prices(sim["start_date"], sim["weeks"], rng, shock)
    precio = nacional.set_index(["semana", "combustible"])["precio_nacional"]

    estaciones = []
    for depto, (codigo, recargo) in DEPARTAMENTOS.items():
        for i in range(1, sim["stations_per_department"] + 1):
            estaciones.append(
                {
                    "estacion_id": f"{codigo}-{i:02d}",
                    "departamento": depto,
                    "marca": MARCAS[(i - 1) % len(MARCAS)],
                    "offset": recargo + rng.normal(0, 0.15),
                }
            )

    fotos = []
    for semana in nacional["semana"].unique():
        for est in estaciones:
            if rng.random() > sim["photo_probability"]:
                continue
            fecha = pd.Timestamp(semana) + pd.Timedelta(
                days=int(rng.integers(0, 7)), hours=int(rng.integers(6, 20))
            )
            lecturas = []
            for comb in COMBUSTIBLES:
                if rng.random() < 0.05:
                    continue  # la foto no muestra este combustible
                valor = precio[(semana, comb)] + est["offset"] + rng.normal(0, 0.05)
                etiqueta = rng.choice(_ETIQUETAS[comb])
                lecturas.append({"combustible": etiqueta, "precio": _texto_precio(valor, rng)})
            depto = est["departamento"]
            if rng.random() < 0.05:  # captura inconsistente del departamento
                depto = depto.lower().replace("é", "e").replace("á", "a")
            fid = _foto_id(est["estacion_id"], fecha)
            fotos.append(
                {
                    "foto_id": fid,
                    "archivo": f"imagenes/fecha={fecha:%Y-%m-%d}/{fid}.jpg",
                    "estacion_id": est["estacion_id"],
                    "marca": est["marca"],
                    "departamento": depto,
                    "fecha_captura": fecha.isoformat(),
                    "lecturas": lecturas,
                }
            )
    # Algunas fotos se suben dos veces.
    dupes = rng.choice(len(fotos), size=max(1, len(fotos) // 100), replace=False)
    return fotos + [fotos[i] for i in dupes]


def write_photos(fotos: list[dict], raw_uri: str) -> int:
    for n, foto in enumerate(fotos):
        fecha = foto["fecha_captura"][:10]
        # Sufijo con el índice para que los duplicados queden como archivos distintos.
        path = f"{raw_uri}/fecha={fecha}/{foto['foto_id']}-{n:05d}.json"
        write_text_atomic(path, json.dumps(foto, ensure_ascii=False))
    return len(fotos)
