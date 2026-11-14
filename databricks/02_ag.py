# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · cusn → ag (plata)
# MAGIC
# MAGIC Limpieza: normaliza combustibles y departamentos, convierte el texto del precio a número,
# MAGIC elimina duplicados, ilegibles y precios fuera de rango, y valida el resultado.
# MAGIC
# MAGIC Equivale a `gasolina_mlops.layers.to_ag` + `validation.validate_ag`.
# MAGIC
# MAGIC > **Sin probar en Databricks.**

# COMMAND ----------

dbutils.widgets.text("catalogo", "cc3105")
catalogo = dbutils.widgets.get("catalogo")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalogo}.ag")

PRECIO_MIN, PRECIO_MAX = 15.0, 60.0
DEPARTAMENTOS = ["Guatemala", "Sacatepéquez", "Escuintla", "Quetzaltenango", "Huehuetenango",
                 "Alta Verapaz", "Izabal", "Petén"]
COMBUSTIBLES = ["super", "regular", "diesel"]

# COMMAND ----------

from pyspark.sql import functions as F


def sin_tildes(col):
    return F.translate(F.lower(F.trim(col)), "áéíóúü", "aeiouu")


deptos = spark.createDataFrame(
    [(d, d.lower().translate(str.maketrans("áéíóúü", "aeiouu"))) for d in DEPARTAMENTOS],
    "departamento_canon string, clave string",
)

cusn = spark.table(f"{catalogo}.cusn.lecturas")

ag = (
    cusn.withColumn("combustible", sin_tildes(F.col("combustible")))
    .withColumn("clave", sin_tildes(F.col("departamento")))
    .join(F.broadcast(deptos), "clave", "left")
    .withColumn("departamento", F.col("departamento_canon"))
    .withColumn(
        "precio",
        F.regexp_replace(F.regexp_replace(F.upper("precio_texto"), "[Q ]", ""), ",", ".")
        .try_cast("double"),
    )
    .withColumn("fecha_captura", F.to_timestamp("fecha_captura"))
    .withColumn("semana", F.date_trunc("week", "fecha_captura").cast("date"))  # lunes
    .dropDuplicates(["foto_id", "combustible"])
    .where(F.col("combustible").isin(COMBUSTIBLES))
    .where(F.col("departamento").isNotNull())
    .where(F.col("precio").between(PRECIO_MIN, PRECIO_MAX))
    .select("foto_id", "archivo", "estacion_id", "marca", "departamento", "fecha_captura",
            "semana", "combustible", "precio")
)

# COMMAND ----------

# Validaciones (C7): si alguna falla, el job se detiene y no se publica la capa.
errores = []
if ag.count() == 0:
    errores.append("ag vacía")
if ag.groupBy("foto_id", "combustible").count().where("count > 1").count():
    errores.append("duplicados (foto_id, combustible)")
if ag.where(F.col("fecha_captura").isNull()).count():
    errores.append("fecha_captura nula")
assert not errores, errores

(ag.write.mode("overwrite").option("overwriteSchema", "true")
   .saveAsTable(f"{catalogo}.ag.precios"))
display(spark.table(f"{catalogo}.ag.precios").limit(20))
