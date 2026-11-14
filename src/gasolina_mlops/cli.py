"""CLI: cada etapa del ciclo de vida es un comando (C2, C5).

uv run gasolina <comando>     (o: uv run python -m gasolina_mlops <comando>)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from gasolina_mlops import layers, monitoring, simulate, storage
from gasolina_mlops.config import layer_uri, load_config
from gasolina_mlops.predict import load_model, predict
from gasolina_mlops.train import REFERENCE_FILE, QualityGateError, train, write_model_card
from gasolina_mlops.validation import (
    DataValidationError,
    assert_valid,
    validate_ag,
    validate_au_modelo,
)


def cmd_simulate(cfg, args):
    fotos = simulate.generate_photos(cfg, cfg["seed"] + (1 if args.shock else 0), shock=args.shock)
    n = simulate.write_photos(fotos, layer_uri(cfg, "raw"))
    print(f"raw -> {layer_uri(cfg, 'raw')} ({n} fotos)")


def cmd_cusn(cfg, args):
    df = layers.to_cusn(layer_uri(cfg, "raw"))
    print(f"cusn -> {storage.write_parquet(df, layer_uri(cfg, 'bronze', 'lecturas.parquet'))}"
          f" ({len(df)} lecturas)")


def cmd_ag(cfg, args):
    df = layers.to_ag(storage.read_parquet(layer_uri(cfg, "bronze", "lecturas.parquet")))
    assert_valid(validate_ag(df))
    print(f"ag -> {storage.write_parquet(df, layer_uri(cfg, 'silver', 'precios.parquet'))}"
          f" ({len(df)} precios, validado)")


def cmd_au(cfg, args):
    tables = layers.to_au(storage.read_parquet(layer_uri(cfg, "silver", "precios.parquet")))
    assert_valid(validate_au_modelo(tables["modelo"]))
    for name, df in tables.items():
        path = storage.write_parquet(df, layer_uri(cfg, "gold", f"{name}.parquet"))
        print(f"au -> {path} ({len(df)} filas)")


def cmd_train(cfg, args):
    modelo = storage.read_parquet(layer_uri(cfg, "gold", "modelo.parquet"))
    meta = train(modelo, cfg, Path(cfg["model"]["dir"]))
    write_model_card(meta, Path(cfg["monitoring"]["reports_dir"]) / "model_card.md")
    print(f"modelo {meta['model_version']} -> {cfg['model']['dir']}")
    print(json.dumps(meta["metrics"], indent=2))


def cmd_predict(cfg, args):
    model, meta = load_model(Path(cfg["model"]["dir"]))
    ultima = storage.read_parquet(layer_uri(cfg, "gold", "ultima_semana.parquet"))
    out = predict(model, ultima, meta["threshold"])
    path = storage.write_parquet(out, layer_uri(cfg, "gold", "predicciones.parquet"))
    print(f"predicciones -> {path}")
    print(out.to_string(index=False))


def cmd_monitor(cfg, args):
    mon = cfg["monitoring"]
    reference = pd.read_parquet(Path(cfg["model"]["dir"]) / REFERENCE_FILE)
    current = storage.read_parquet(args.current or layer_uri(cfg, "gold", "modelo.parquet"))
    if args.last_weeks:
        semanas = sorted(current["semana"].unique())[-args.last_weeks:]
        current = current[current["semana"].isin(semanas)]
    report = monitoring.drift_report(reference, current, mon["psi_aviso"], mon["psi_alerta"])
    ops = monitoring.predictions_summary(Path(mon["predictions_log"]))
    if ops:
        report["servicio"] = ops
    reports = Path(mon["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "drift_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (reports / "drift_report.md").write_text(monitoring.render_markdown(report, ops), "utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.fail_on_drift and report["drift_detected"]:
        sys.exit(2)


def cmd_pipeline(cfg, args):
    for step in (cmd_cusn, cmd_ag, cmd_au, cmd_train, cmd_predict):
        step(cfg, args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gasolina")
    parser.add_argument("--config", default=None, help="YAML de configuración")
    parser.add_argument("--base-uri", default=None, help="Sobrescribe storage.base_uri")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("simulate", help="Genera JSON de fotos simuladas en raw/")
    s.add_argument("--shock", action="store_true", help="Escenario con alza fuerte (drift)")
    s.set_defaults(func=cmd_simulate)
    sub.add_parser("cusn", help="raw -> cusn (bronce)").set_defaults(func=cmd_cusn)
    sub.add_parser("ag", help="cusn -> ag (plata) + validación").set_defaults(func=cmd_ag)
    sub.add_parser("au", help="ag -> au (oro) + validación").set_defaults(func=cmd_au)
    sub.add_parser("train", help="Entrena, evalúa y registra el modelo").set_defaults(
        func=cmd_train
    )
    sub.add_parser("predict", help="Predice la próxima semana desde au").set_defaults(
        func=cmd_predict
    )
    sub.add_parser("pipeline", help="cusn -> ag -> au -> train -> predict").set_defaults(
        func=cmd_pipeline
    )
    m = sub.add_parser("monitor", help="Reporte de drift contra los datos de entrenamiento")
    m.add_argument("--current", default=None, help="Parquet au/modelo a comparar")
    m.add_argument("--last-weeks", type=int, default=12, help="Ventana reciente a comparar")
    m.add_argument("--fail-on-drift", action="store_true")
    m.set_defaults(func=cmd_monitor)

    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.base_uri)
    try:
        args.func(cfg, args)
    except (DataValidationError, QualityGateError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
