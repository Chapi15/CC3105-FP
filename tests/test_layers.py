import numpy as np
import pytest

from gasolina_mlops import layers, simulate
from gasolina_mlops.validation import validate_ag, validate_au_modelo


def test_simulacion_reproducible(cfg):
    a = simulate.generate_photos(cfg, seed=1)
    b = simulate.generate_photos(cfg, seed=1)
    assert a == b


def test_cusn_conserva_todo_y_agrega_metadatos(cusn, fotos):
    assert len(cusn) == sum(len(f["lecturas"]) for f in fotos)
    assert {"_source_file", "_ingested_at", "precio_texto"} <= set(cusn.columns)


@pytest.mark.parametrize(
    "texto,esperado",
    [("Q 32.49", 32.49), ("Q32,49", 32.49), ("30.10", 30.10), ("", np.nan), (None, np.nan),
     ("ilegible", np.nan)],
)
def test_parse_precio(texto, esperado):
    valor = layers.parse_precio(texto)
    assert (np.isnan(valor) and np.isnan(esperado)) or valor == pytest.approx(esperado)


@pytest.mark.parametrize(
    "texto,esperado", [("Súper", "super"), ("SUPER", "super"), ("Diésel", "diesel"), ("gas", None)]
)
def test_normalizar_combustible(texto, esperado):
    assert layers.normalizar_combustible(texto) == esperado


def test_normalizar_departamento():
    assert layers.normalizar_departamento("peten ") == "Petén"
    assert layers.normalizar_departamento("Narnia") is None


def test_ag_limpia_y_valida(ag, cusn):
    assert validate_ag(ag) == []
    assert len(ag) < len(cusn)  # se descartaron duplicados, ilegibles y fuera de rango


def test_validacion_ag_detecta_errores(ag):
    malo = ag.copy()
    malo.loc[0, "precio"] = 3.2
    malo.loc[1, "departamento"] = "Narnia"
    errores = validate_ag(malo)
    assert any("fuera de" in e for e in errores)
    assert any("Narnia" in e for e in errores)


def test_au_objetivo_bien_construido(au):
    modelo = au["modelo"]
    assert validate_au_modelo(modelo) == []
    # El objetivo compara el precio de la semana siguiente contra la actual.
    esperado = (modelo["precio_siguiente"] > modelo["precio_t0"]).astype(int)
    assert (modelo["sube"] == esperado).all()


def test_au_ultima_semana_sin_objetivo(au):
    ultima = au["ultima_semana"]
    assert "sube" not in ultima.columns
    assert ultima["semana"].nunique() == 1
    assert ultima["semana"].max() > au["modelo"]["semana"].max()
