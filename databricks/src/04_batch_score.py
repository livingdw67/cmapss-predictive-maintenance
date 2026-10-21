# Databricks notebook source
# MAGIC %md
# MAGIC # 04 Batch scoring: fleet risk from the champion models
# MAGIC
# MAGIC Scores the latest cycle of every NASA test engine (histories cut off before failure) with the
# MAGIC `champion` versions of both classifiers, applies the tandem decision matrix, and writes
# MAGIC `engine_risk_scores`. The NASA true RUL for those engines is joined in, so the alerts can be checked
# MAGIC against what actually happened.
# MAGIC
# MAGIC | Early Warning | Critical Action | Action |
# MAGIC |---|---|---|
# MAGIC | 0 | 0 | Normal operations |
# MAGIC | 1 | 0 | Schedule maintenance |
# MAGIC | 1 | 1 | Immediate action required |
# MAGIC | 0 | 1 | Anomaly - investigate |

# COMMAND ----------

# MAGIC %pip install xgboost==2.1.4 --quiet
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "cmapss")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

latest = spark.sql(f"""
    SELECT f.*
    FROM {catalog}.{schema}.fs_test_features f
    QUALIFY cycle = MAX(cycle) OVER (PARTITION BY engine_id)
""").toPandas()

scores = latest[["engine_id", "cycle"]].copy()
versions = {}
for model_name, prefix in [("turbofan_early_warning", "early_warning"),
                           ("turbofan_critical_action", "critical_action")]:
    uri = f"models:/{catalog}.{schema}.{model_name}@champion"
    preds = mlflow.pyfunc.load_model(uri).predict(latest)
    scores[f"{prefix}_probability"] = preds["probability"].values
    scores[f"{prefix}_alert"] = preds["alert"].values
    versions[prefix] = client.get_model_version_by_alias(f"{catalog}.{schema}.{model_name}", "champion").version

actions = {(0, 0): "Normal operations", (1, 0): "Schedule maintenance",
           (1, 1): "Immediate action required", (0, 1): "Anomaly - investigate"}
scores["action"] = [actions[(a, b)] for a, b in zip(scores["early_warning_alert"], scores["critical_action_alert"])]
scores["early_warning_version"] = versions["early_warning"]
scores["critical_action_version"] = versions["critical_action"]

(spark.createDataFrame(scores)
    .join(spark.table(f"{catalog}.{schema}.test_rul_truth"), "engine_id")
    .withColumnRenamed("cycle", "last_cycle")
    .selectExpr("*", "current_timestamp() AS scored_at")
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{catalog}.{schema}.engine_risk_scores"))

# COMMAND ----------

# How the alerts line up with the true remaining life of each test engine
display(spark.sql(f"""
    SELECT action,
           COUNT(*) AS engines,
           ROUND(AVG(true_rul), 1) AS avg_true_rul,
           SUM(IF(true_rul <= 50, 1, 0)) AS truly_within_50,
           SUM(IF(true_rul <= 15, 1, 0)) AS truly_within_15
    FROM {catalog}.{schema}.engine_risk_scores
    GROUP BY action
    ORDER BY avg_true_rul
"""))

# COMMAND ----------

# Recall and precision at the engine level (one prediction per test engine)
display(spark.sql(f"""
    SELECT 'early_warning (<=50)' AS model,
           ROUND(SUM(IF(early_warning_alert = 1 AND true_rul <= 50, 1, 0)) / SUM(IF(true_rul <= 50, 1, 0)), 3) AS recall,
           ROUND(SUM(IF(early_warning_alert = 1 AND true_rul <= 50, 1, 0)) / SUM(early_warning_alert), 3) AS precision
    FROM {catalog}.{schema}.engine_risk_scores
    UNION ALL
    SELECT 'critical_action (<=15)',
           ROUND(SUM(IF(critical_action_alert = 1 AND true_rul <= 15, 1, 0)) / SUM(IF(true_rul <= 15, 1, 0)), 3),
           ROUND(SUM(IF(critical_action_alert = 1 AND true_rul <= 15, 1, 0)) / SUM(critical_action_alert), 3)
    FROM {catalog}.{schema}.engine_risk_scores
"""))
