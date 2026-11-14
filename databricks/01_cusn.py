# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · raw → cusn (bronce)
# MAGIC
# MAGIC Lee los JSON de lecturas de fotos desde el external location de S3 y los guarda **sin
# MAGIC transformar valores**, una fila por lectura, con metadatos de ingesta.
# MAGIC
# MAGIC Equivale a `gasolina_mlops.layers.to_cusn` (la versión en pandas que se prueba en CI).
# MAGIC
# MAGIC > **Sin probar en Databricks.** Ajustar `catalogo` y `raw_path` a cada workspace.

# COMMAND ----------

dbutils.widgets.text("catalogo", "cc3105")
dbutils.widgets.text("raw_path", "s3://<bucket>/<equipo>/raw/fotos/")

catalogo = dbutils.widgets.get("catalogo")
raw_path = dbutils.widgets.get("raw_path")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalogo}.cusn")

# COMMAND ----------

from pyspark.sql import functions as F

# La carpeta fecha=AAAA-MM-DD se descubre como columna de partición.
raw = spark.read.option("recursiveFileLookup", "false").json(raw_path)

cusn = (
    raw.select("*", F.explode_outer("lecturas").alias("lectura"))
    .select(
        "foto_id",
        "archivo",
        "estacion_id",
        "marca",
        "departamento",
        "fecha_captura",
        F.col("lectura.combustible").alias("combustible"),
        F.col("lectura.precio").alias("precio_texto"),
        F.col("_metadata.file_path").alias("_source_file"),
        F.current_timestamp().alias("_ingested_at"),
    )
)

(cusn.write.mode("overwrite").option("overwriteSchema", "true")
     .saveAsTable(f"{catalogo}.cusn.lecturas"))
display(spark.table(f"{catalogo}.cusn.lecturas").limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC Alternativa en SQL (útil para explicar en clase):
# MAGIC ```sql
# MAGIC SELECT f.*, l.combustible, l.precio AS precio_texto
# MAGIC FROM read_files('s3://<bucket>/<equipo>/raw/fotos/', format => 'json') f
# MAGIC LATERAL VIEW explode(f.lecturas) AS l
# MAGIC ```
