import copy

from gasolina_mlops import layers, monitoring, simulate
from gasolina_mlops.config import layer_uri


def test_sin_drift_en_la_misma_distribucion(au):
    m = au["modelo"]
    semanas = sorted(m["semana"].unique())
    ref, cur = m[m["semana"].isin(semanas[:40])], m[m["semana"].isin(semanas[40:])]
    assert not monitoring.drift_report(ref, cur, 0.25, 1.5)["drift_detected"]


def test_detecta_drift_con_choque(au, cfg, tmp_path):
    c = copy.deepcopy(cfg)
    c["storage"]["base_uri"] = str(tmp_path)
    simulate.write_photos(simulate.generate_photos(c, 7, shock=True), layer_uri(c, "raw"))
    shock = layers.to_au(layers.to_ag(layers.to_cusn(layer_uri(c, "raw"))))["modelo"]
    ultimas = sorted(shock["semana"].unique())[-12:]
    report = monitoring.drift_report(au["modelo"], shock[shock["semana"].isin(ultimas)], 0.25, 1.5)
    assert report["drift_detected"]
    assert report["estado"] == "alerta"


def test_niveles():
    assert monitoring.nivel(0.1, 0.25, 1.5) == "ok"
    assert monitoring.nivel(0.5, 0.25, 1.5) == "aviso"
    assert monitoring.nivel(2.0, 0.25, 1.5) == "alerta"
