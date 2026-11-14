import pytest
from fastapi.testclient import TestClient

SERIE = {"departamento": "Petén", "combustible": "diesel", "precios": [29.8, 30.0, 30.3, 30.6]}


@pytest.fixture()
def client(trained, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(trained[0]))
    monkeypatch.setenv("PREDICTIONS_LOG", str(tmp_path / "predictions.jsonl"))
    from gasolina_mlops.api import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_model_info(client):
    assert "metrics" in client.get("/model-info").json()


def test_predict(client, tmp_path):
    r = client.post("/predict", json={"series": [SERIE, {**SERIE, "combustible": "super"}]})
    assert r.status_code == 200
    preds = r.json()["predicciones"]
    assert len(preds) == 2
    assert preds[0]["precio_actual"] == 30.6  # el último de la lista es la semana actual
    assert preds[0]["prediccion"] in {"sube", "baja"}
    assert (tmp_path / "predictions.jsonl").exists()


@pytest.mark.parametrize(
    "cambio",
    [
        {"precios": [30.0, 30.1, 30.2]},          # faltan semanas
        {"precios": [30.0, 30.1, 30.2, 300.0]},   # precio imposible
        {"combustible": "gas"},
        {"departamento": "Narnia"},
    ],
)
def test_predict_rechaza_entradas_invalidas(client, cambio):
    assert client.post("/predict", json={"series": [{**SERIE, **cambio}]}).status_code == 422


def test_predict_lista_vacia(client):
    assert client.post("/predict", json={"series": []}).status_code == 422
