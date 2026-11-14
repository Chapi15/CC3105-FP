import pandas as pd
import pytest

from gasolina_mlops.features import FEATURES, add_derived_features, build_pipeline
from gasolina_mlops.predict import load_model, predict
from gasolina_mlops.train import QualityGateError, temporal_split, train


def test_split_temporal_sin_fuga(au, cfg):
    tr, te = temporal_split(au["modelo"], cfg["model"]["test_weeks"])
    assert tr["semana"].max() < te["semana"].min()
    assert te["semana"].nunique() == cfg["model"]["test_weeks"]


def test_features_derivadas():
    X = pd.DataFrame([{"precio_t0": 31.0, "precio_t1": 30.0, "precio_t2": 30.5, "precio_t3": 30.5}])
    out = add_derived_features(X)
    assert out.loc[0, "cambio_1"] == pytest.approx(1.0)
    assert out.loc[0, "cambio_2"] == pytest.approx(-0.5)
    assert out.loc[0, "rango_4"] == pytest.approx(1.0)


def test_algoritmo_invalido():
    with pytest.raises(ValueError):
        build_pipeline("bola_de_cristal", {}, 0)


def test_modelo_supera_gates(trained, cfg):
    m = trained[1]["metrics"]
    assert m["roc_auc"] >= cfg["model"]["min_roc_auc"]
    assert m["accuracy"] > m["baseline_accuracy"]


def test_quality_gate_bloquea(au, cfg, tmp_path):
    estricto = {**cfg, "model": {**cfg["model"], "min_roc_auc": 0.99}}
    with pytest.raises(QualityGateError):
        train(au["modelo"], estricto, tmp_path)


def test_prediccion(trained, au):
    model, meta = load_model(trained[0])
    out = predict(model, au["ultima_semana"], meta["threshold"])
    assert out["probabilidad_sube"].between(0, 1).all()
    assert set(out["prediccion"]) <= {"sube", "baja"}


def test_comportamiento_direccional(trained, au):
    """Con inercia en los precios, una racha de alzas debe dar más probabilidad de subir
    que una racha de bajas."""
    model, _ = load_model(trained[0])
    base = au["modelo"][FEATURES].sample(200, random_state=0)
    p0 = base["precio_t0"]
    alza = base.assign(precio_t1=p0 - 0.3, precio_t2=p0 - 0.6, precio_t3=p0 - 0.9)
    baja = base.assign(precio_t1=p0 + 0.3, precio_t2=p0 + 0.6, precio_t3=p0 + 0.9)
    assert model.predict_proba(alza)[:, 1].mean() > model.predict_proba(baja)[:, 1].mean()
