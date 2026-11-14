# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Entrenamiento desde `au.modelo` (desarrollo)
# MAGIC
# MAGIC Databricks es el ambiente de desarrollo: aquí se experimenta con el **mismo paquete** que
# MAGIC corre en CI y en DigitalOcean, para no reescribir la lógica del modelo en notebooks (C2).
# MAGIC
# MAGIC > **Sin probar en Databricks.** Si el repo es privado, instalar desde un wheel subido a un
# MAGIC > volume en lugar de `git+https`.

# COMMAND ----------

# MAGIC %pip install "git+https://github.com/prof-aro-uvg/CC3105-FP.git"

# COMMAND ----------

dbutils.widgets.text("catalogo", "cc3105")
catalogo = dbutils.widgets.get("catalogo")

# COMMAND ----------

import json
import tempfile
from pathlib import Path

import yaml

from gasolina_mlops.train import train

cfg = yaml.safe_load("""
seed: 42
model:
  algorithm: logistic_regression
  params: {C: 1.0}
  test_weeks: 18
  threshold: 0.5
  min_roc_auc: 0.65
  must_beat_baseline: true
""")

modelo = spark.table(f"{catalogo}.au.modelo").toPandas()
out = Path(tempfile.mkdtemp())
meta = train(modelo, cfg, out)
print(json.dumps(meta["metrics"], indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC Experimento sugerido: cambiar `algorithm` a `gradient_boosting` y explicar por qué empeora
# MAGIC con un split temporal (los árboles no extrapolan cuando el precio sale del rango visto).
