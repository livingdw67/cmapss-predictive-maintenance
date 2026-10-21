# Databricks notebook source
# MAGIC %md
# MAGIC # 03 Train and register the tandem classifiers
# MAGIC
# MAGIC Port of `08_build_register_classifiers.ipynb` (v2.0) with MLflow tracking and the Unity Catalog model registry:
# MAGIC - **Model A (Early Warning):** failure within 50 cycles
# MAGIC - **Model B (Critical Action):** failure within 15 cycles
# MAGIC
# MAGIC Engines are split 60/20/20 (stratified by lifespan, no engine in two sets). For each model, three
# MAGIC imbalance strategies are compared (class weighting, SMOTE, 6x undersampling), each with its F2 threshold
# MAGIC tuned on the validation engines. The best validation F2 is registered and gets the `champion` alias.
# MAGIC The registered model applies its own threshold, so scoring code never hard-codes one.

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
import numpy as np
import pandas as pd
import sklearn
import xgboost
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import fbeta_score, precision_score, recall_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier

mlflow.set_registry_uri("databricks-uc")
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/cmapss-tandem-classifiers")

FEATURES = ["hpc_outlet_temp", "physical_core_speed", "bypass_ratio",
            "hpc_temp_ma_5", "core_speed_ma_5", "bypass_ratio_ma_5",
            "hpc_temp_std_5", "core_speed_std_5"]
TARGETS = {
    "turbofan_early_warning": ("early_warning", 50),
    "turbofan_critical_action": ("critical_action", 15),
}

df = spark.table(f"{catalog}.{schema}.fs_turbofan_predictive_features").toPandas()
for _, (label, horizon) in TARGETS.items():
    df[label] = (df["target_rul"] <= horizon).astype(int)
print(f"{len(df):,} records, {df['engine_id'].nunique()} engines")

# COMMAND ----------

# 60/20/20 engine-level split, stratified by lifespan bucket
engine_stats = df.groupby("engine_id").agg(max_cycle=("cycle", "max")).reset_index()
engine_stats["lifespan_bucket"] = pd.cut(engine_stats["max_cycle"], bins=[0, 200, 300, 400, 600],
                                         labels=["short", "medium", "long", "very_long"])
train_stats, rest_stats = train_test_split(engine_stats, test_size=0.4,
                                           stratify=engine_stats["lifespan_bucket"], random_state=42)
val_stats, test_stats = train_test_split(rest_stats, test_size=0.5,
                                         stratify=rest_stats["lifespan_bucket"], random_state=42)

train_df = df[df["engine_id"].isin(train_stats["engine_id"])].reset_index(drop=True)
val_df = df[df["engine_id"].isin(val_stats["engine_id"])].reset_index(drop=True)
test_df = df[df["engine_id"].isin(test_stats["engine_id"])].reset_index(drop=True)
for name, part in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
    print(f"{name:<11} {len(part):>7,} records | {part['engine_id'].nunique()} engines")

X_train, X_val, X_test = train_df[FEATURES], val_df[FEATURES], test_df[FEATURES]

# COMMAND ----------

def manual_smote(X, y, random_state=42, k_neighbors=5):
    rng = np.random.default_rng(random_state)
    X_min = X[y == 1].values
    n_to_generate = int((y == 0).sum() - (y == 1).sum())
    neighbors = NearestNeighbors(n_neighbors=k_neighbors).fit(X_min).kneighbors(X_min, return_distance=False)
    base = rng.integers(0, len(X_min), n_to_generate)
    nbr = neighbors[base, rng.integers(0, k_neighbors, n_to_generate)]
    synthetic = X_min[base] + rng.random((n_to_generate, 1)) * (X_min[nbr] - X_min[base])
    X_res = pd.DataFrame(np.vstack([X.values, synthetic]), columns=X.columns)
    y_res = pd.Series(np.concatenate([y.values, np.ones(n_to_generate, dtype=int)]))
    order = rng.permutation(len(X_res))
    return X_res.iloc[order].reset_index(drop=True), y_res.iloc[order].reset_index(drop=True)


def undersample_6x(X, y, random_state=42):
    rng = np.random.default_rng(random_state)
    minority = np.where(y == 1)[0]
    majority = np.where(y == 0)[0]
    # Cap at the available majority rows (Early Warning positives are ~23%, so 6x can exceed them)
    sampled = rng.choice(majority, size=min(6 * len(minority), len(majority)), replace=False)
    selected = rng.permutation(np.concatenate([minority, sampled]))
    return X.iloc[selected].reset_index(drop=True), y.iloc[selected].reset_index(drop=True)


def best_f2_threshold(y_true, proba):
    scores = {t: fbeta_score(y_true, (proba >= t).astype(int), beta=2) for t in np.arange(0.05, 0.95, 0.01)}
    t = max(scores, key=scores.get)
    return float(t), float(scores[t])


class ThresholdedClassifier(mlflow.pyfunc.PythonModel):
    """XGBoost classifier plus the F2 threshold chosen on validation engines."""

    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def predict(self, context, model_input):
        proba = self.model.predict_proba(model_input[FEATURES])[:, 1]
        return pd.DataFrame({"probability": proba, "alert": (proba >= self.threshold).astype(int)})

# COMMAND ----------

client = MlflowClient()
summary = []

for model_name, (label, horizon) in TARGETS.items():
    y_train, y_val, y_test = train_df[label], val_df[label], test_df[label]
    strategies = {
        "baseline": (X_train, y_train, float((y_train == 0).sum() / (y_train == 1).sum())),
        "smote": (*manual_smote(X_train, y_train), 1.0),
        "undersample": (*undersample_6x(X_train, y_train), 1.0),
    }

    with mlflow.start_run(run_name=f"{model_name}") as parent:
        mlflow.log_params({"horizon_cycles": horizon, "features": ",".join(FEATURES),
                           "train_engines": len(train_stats), "val_engines": len(val_stats),
                           "test_engines": len(test_stats)})
        results = {}
        for strategy, (X_tr, y_tr, spw) in strategies.items():
            with mlflow.start_run(run_name=strategy, nested=True):
                params = dict(n_estimators=300, max_depth=7, learning_rate=0.08, scale_pos_weight=spw,
                              eval_metric="aucpr", random_state=42, n_jobs=-1)
                model = XGBClassifier(**params).fit(X_tr, y_tr)
                threshold, val_f2 = best_f2_threshold(y_val, model.predict_proba(X_val)[:, 1])
                test_proba = model.predict_proba(X_test)[:, 1]
                test_pred = (test_proba >= threshold).astype(int)
                metrics = {
                    "val_f2": val_f2,
                    "test_f2": fbeta_score(y_test, test_pred, beta=2),
                    "test_recall": recall_score(y_test, test_pred),
                    "test_precision": precision_score(y_test, test_pred),
                    "test_pr_auc": average_precision_score(y_test, test_proba),
                    "threshold": threshold,
                }
                mlflow.log_params({"strategy": strategy, "train_rows": len(y_tr),
                                   **{k: v for k, v in params.items() if k != "n_jobs"}})
                mlflow.log_metrics(metrics)
                results[strategy] = {"model": model, **metrics}

        # Select on validation F2 only; test metrics are reported, never used for selection
        best = max(results, key=lambda k: results[k]["val_f2"])
        r = results[best]
        mlflow.log_param("selected_strategy", best)
        mlflow.log_metrics({k: v for k, v in r.items() if k != "model"})

        wrapped = ThresholdedClassifier(r["model"], r["threshold"])
        sample = X_test.head(100)
        info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=wrapped,
            input_example=sample,
            signature=infer_signature(sample, wrapped.predict(None, sample)),
            pip_requirements=[f"xgboost=={xgboost.__version__}", f"scikit-learn=={sklearn.__version__}",
                              f"pandas=={pd.__version__}"],
            registered_model_name=f"{catalog}.{schema}.{model_name}",
        )
        version = info.registered_model_version
        client.set_registered_model_alias(f"{catalog}.{schema}.{model_name}", "champion", version)
        client.update_model_version(
            f"{catalog}.{schema}.{model_name}", version,
            description=f"{best} selected on validation F2; threshold {r['threshold']:.2f} tuned on validation engines.",
        )
        summary.append({"model": model_name, "version": version, "strategy": best,
                        **{k: round(v, 4) for k, v in r.items() if k != "model"}})

display(pd.DataFrame(summary))
