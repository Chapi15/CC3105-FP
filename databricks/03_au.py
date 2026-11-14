# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · ag → au (oro)
# MAGIC
# MAGIC Tablas de consumo:
# MAGIC - `au.precios_semanales`: promedio semanal por departamento y combustible (dashboard).
# MAGIC - `au.precio_nacional`: promedio semanal nacional por combustible (dashboard).
# MAGIC - `au.modelo`: precios de las últimas 4 semanas + objetivo `sube` (entrenamiento).
# MAGIC
# MAGIC Equivale a `gasolina_mlops.layers.to_au`. Diferencia: aquí no se interpolan semanas sin
# MAGIC fotos, se descartan (ejercicio: implementarlo con una tabla calendario).
# MAGIC
# MAGIC > **Sin probar en Databricks.**

# COMMAND ----------

dbutils.widgets.text("catalogo", "cc3105")
catalogo = dbutils.widgets.get("catalogo")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalogo}.au")

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

ag = spark.table(f"{catalogo}.ag.precios")

semanal = ag.groupBy("semana", "departamento", "combustible").agg(
    F.round(F.avg("precio"), 3).alias("precio_promedio"),
    F.countDistinct("foto_id").alias("n_fotos"),
)
semanal.write.mode("overwrite").saveAsTable(f"{catalogo}.au.precios_semanales")

nacional = ag.groupBy("semana", "combustible").agg(
    F.round(F.avg("precio"), 3).alias("precio_promedio")
)
nacional.write.mode("overwrite").saveAsTable(f"{catalogo}.au.precio_nacional")

# COMMAND ----------

w = Window.partitionBy("departamento", "combustible").orderBy("semana")
series = semanal.select(
    "semana",
    "departamento",
    "combustible",
    F.col("precio_promedio").alias("precio_t0"),
    *[F.lag("precio_promedio", i).over(w).alias(f"precio_t{i}") for i in (1, 2, 3)],
    F.lead("precio_promedio", 1).over(w).alias("precio_siguiente"),
).dropna(subset=["precio_t1", "precio_t2", "precio_t3"])

modelo = series.where(F.col("precio_siguiente").isNotNull()).withColumn(
    "sube", (F.col("precio_siguiente") > F.col("precio_t0")).cast("int")
)
modelo.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalogo}.au.modelo"
)
display(modelo.orderBy(F.desc("semana")).limit(20))
