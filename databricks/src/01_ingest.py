# Databricks notebook source
# MAGIC %md
# MAGIC # 01 Ingest: NASA C-MAPSS files into a bronze Delta table
# MAGIC
# MAGIC Downloads the archive into a Unity Catalog volume (skipped if the files are already there),
# MAGIC then loads every `train_FD00x.txt` into `raw_train`. Mirrors `01_fetch_data.py` + `03_load_raw_table.py`.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "cmapss")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.raw_files")
volume_dir = f"/Volumes/{catalog}/{schema}/raw_files"

# COMMAND ----------

import io
import os
import urllib.request
import zipfile

if not any(f.startswith("train_FD") for f in os.listdir(volume_dir)):
    # The original ti.arc.nasa.gov link now serves an HTML page; data.nasa.gov hosts the same archive
    url = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    for name in archive.namelist():
        if name.endswith(".txt"):
            with open(os.path.join(volume_dir, os.path.basename(name)), "wb") as out:
                out.write(archive.read(name))

print(sorted(os.listdir(volume_dir)))

# COMMAND ----------

# The files are space-delimited with trailing spaces, so split on whitespace instead of using a CSV reader.
# Everything stays a STRING in the raw layer; casting happens in staging.
from pyspark.sql import functions as F

cols = ["id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"sensor_{i}" for i in range(1, 22)]

def load_raw(pattern, table):
    lines = (
        spark.read.text(f"{volume_dir}/{pattern}")
        .select(F.split(F.trim("value"), r"\s+").alias("parts"), F.col("_metadata.file_name").alias("source_file"))
    )
    raw = lines.select(*[F.col("parts")[i].alias(c) for i, c in enumerate(cols)], "source_file")
    raw.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.{table}")
    print(f"Total rows loaded into {table}: {spark.table(f'{catalog}.{schema}.{table}').count():,}")

# train_FD00x: run-to-failure histories (model training)
# test_FD00x: histories cut off before failure (batch scoring); RUL_FD00x holds their true remaining life
load_raw("train_FD00*.txt", "raw_train")
load_raw("test_FD00*.txt", "raw_test")

# COMMAND ----------

# RUL_FD00x has one value per line, in engine order, so line number = engine number
import pandas as pd

truth = []
for dataset in ["FD001", "FD002", "FD003", "FD004"]:
    with open(f"{volume_dir}/RUL_{dataset}.txt") as f:
        values = [int(line) for line in f if line.strip()]
    truth += [(f"{dataset}_{n:03d}", dataset, rul) for n, rul in enumerate(values, start=1)]

(spark.createDataFrame(pd.DataFrame(truth, columns=["engine_id", "dataset", "true_rul"]))
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.test_rul_truth"))
print(f"Total test engines with true RUL: {len(truth)}")
