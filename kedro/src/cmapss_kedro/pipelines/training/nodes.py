"""Engine-level split and the per-horizon classifier: the logic of `databricks/src/03_train_register.py`.

For each horizon, three imbalance strategies are compared (class weighting, SMOTE, 6x undersampling),
each with its F2 threshold tuned on validation engines. The best validation F2 wins; test engines are
reported, never used for selection.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier

from cmapss_kedro.pipelines.data_processing.nodes import FEATURES

log = logging.getLogger(__name__)


class ThresholdedClassifier:
    """An XGBoost classifier plus the F2 threshold chosen on validation engines, so scoring never
    hard-codes a threshold."""

    def __init__(self, model: XGBClassifier, threshold: float, strategy: str):
        self.model, self.threshold, self.strategy = model, threshold, strategy

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        proba = self.model.predict_proba(features[FEATURES])[:, 1]
        return pd.DataFrame({"probability": proba, "alert": (proba >= self.threshold).astype(int)},
                            index=features.index)


def split_engines(model_input: pd.DataFrame, split: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """60/20/20 split by engine, stratified by lifespan, so no engine's cycles land in two sets."""
    stats = model_input.groupby("engine_id").agg(max_cycle=("cycle", "max")).reset_index()
    stats["bucket"] = pd.cut(stats["max_cycle"], bins=split["lifespan_bins"])
    held_out = split["validation_share"] + split["test_share"]
    train_ids, rest = train_test_split(stats, test_size=held_out, stratify=stats["bucket"],
                                       random_state=split["seed"])
    val_ids, test_ids = train_test_split(rest, test_size=split["test_share"] / held_out,
                                         stratify=rest["bucket"], random_state=split["seed"])
    parts = [model_input[model_input["engine_id"].isin(ids["engine_id"])].reset_index(drop=True)
             for ids in (train_ids, val_ids, test_ids)]
    for name, part in zip(("train", "validation", "test"), parts):
        log.info("%-10s %7d rows, %d engines", name, len(part), part["engine_id"].nunique())
    return tuple(parts)


def _smote(X, y, seed, k_neighbors=5):
    """Synthetic failures on lines between each failure row and one of its 5 nearest failure rows,
    until the classes are balanced."""
    rng = np.random.default_rng(seed)
    X_min = X[y == 1].to_numpy()
    n_new = int((y == 0).sum() - (y == 1).sum())
    neighbors = NearestNeighbors(n_neighbors=k_neighbors).fit(X_min).kneighbors(X_min, return_distance=False)
    base = rng.integers(0, len(X_min), n_new)
    other = neighbors[base, rng.integers(0, k_neighbors, n_new)]
    synthetic = X_min[base] + rng.random((n_new, 1)) * (X_min[other] - X_min[base])
    X_res = pd.DataFrame(np.vstack([X.to_numpy(), synthetic]), columns=X.columns)
    y_res = pd.Series(np.concatenate([y.to_numpy(), np.ones(n_new, dtype=int)]))
    order = rng.permutation(len(X_res))
    return X_res.iloc[order].reset_index(drop=True), y_res.iloc[order].reset_index(drop=True)


def _undersample(X, y, seed, ratio=6):
    """All failure rows plus `ratio` times as many healthy rows (capped at the healthy rows there are)."""
    rng = np.random.default_rng(seed)
    minority, majority = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    sampled = rng.choice(majority, size=min(ratio * len(minority), len(majority)), replace=False)
    keep = rng.permutation(np.concatenate([minority, sampled]))
    return X.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True)


def _best_f2_threshold(y_true, proba, grid):
    """F2 weights recall 4x as much as precision: a missed failure costs more than an early inspection."""
    scores = {t: fbeta_score(y_true, (proba >= t).astype(int), beta=2)
              for t in np.arange(grid["start"], grid["stop"], grid["step"])}
    best = max(scores, key=scores.get)
    return round(float(best), 2), float(scores[best])


def train_classifier(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, horizon: int,
                     xgboost: dict, threshold_grid: dict) -> tuple[ThresholdedClassifier, pd.DataFrame, dict]:
    """Label rows as failing within `horizon` cycles, fit one XGBoost model per imbalance strategy, tune
    each threshold on validation engines and keep the best validation F2."""
    y_train, y_val, y_test = (part["target_rul"].le(horizon).astype(int) for part in (train, validation, test))
    X_train = train[FEATURES]
    seed = xgboost["random_state"]
    strategies = {
        "class_weight": (X_train, y_train, float((y_train == 0).sum() / (y_train == 1).sum())),
        "smote": (*_smote(X_train, y_train, seed), 1.0),
        "undersample": (*_undersample(X_train, y_train, seed), 1.0),
    }
    rows, models = [], {}
    for strategy, (X, y, weight) in strategies.items():
        model = XGBClassifier(**xgboost, scale_pos_weight=weight).fit(X, y)
        threshold, val_f2 = _best_f2_threshold(y_val, model.predict_proba(validation[FEATURES])[:, 1], threshold_grid)
        test_proba = model.predict_proba(test[FEATURES])[:, 1]
        test_alert = (test_proba >= threshold).astype(int)
        rows.append({"strategy": strategy, "train_rows": len(y), "threshold": threshold, "val_f2": val_f2,
                     "test_f2": fbeta_score(y_test, test_alert, beta=2),
                     "test_recall": recall_score(y_test, test_alert),
                     "test_precision": precision_score(y_test, test_alert),
                     "test_pr_auc": average_precision_score(y_test, test_proba)})
        models[strategy] = ThresholdedClassifier(model, threshold, strategy)
    comparison = pd.DataFrame(rows)
    best = comparison.loc[comparison["val_f2"].idxmax()]
    comparison = comparison.round(4)
    log.info("horizon %d: %s selected (val F2 %.3f, threshold %.2f)\n%s", horizon, best["strategy"],
             best["val_f2"], best["threshold"], comparison.to_string(index=False))
    metrics = {k: {"value": float(best[k]), "step": 0}
               for k in ("threshold", "val_f2", "test_f2", "test_recall", "test_precision", "test_pr_auc")}
    return models[best["strategy"]], comparison, metrics
