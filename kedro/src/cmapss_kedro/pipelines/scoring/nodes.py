"""Fleet scoring and its check against NASA's true remaining life: `databricks/src/04_batch_score.py`."""

import math

import pandas as pd

from cmapss_kedro.pipelines.training.nodes import ThresholdedClassifier

ACTIONS = {(0, 0): "Normal operations", (1, 0): "Schedule maintenance",
           (1, 1): "Immediate action required", (0, 1): "Anomaly - investigate"}


def score_fleet(features: pd.DataFrame, early_warning_model: ThresholdedClassifier,
                critical_action_model: ThresholdedClassifier) -> pd.DataFrame:
    """Both alerts for each engine's latest cycle, combined into a maintenance action."""
    scores = features[["engine_id", "cycle"]].rename(columns={"cycle": "last_cycle"})
    for prefix, model in (("early_warning", early_warning_model), ("critical_action", critical_action_model)):
        predicted = model.predict(features)
        scores[f"{prefix}_probability"] = predicted["probability"]
        scores[f"{prefix}_alert"] = predicted["alert"]
    scores["action"] = [ACTIONS[pair] for pair in zip(scores["early_warning_alert"], scores["critical_action_alert"])]
    return scores


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion. Better than p ± 1.96·se when p is near 0 or 1 or n is
    small, which is the case for recall on the ~100 engines that are truly close to failure."""
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def evaluate_alerts(risk_scores: pd.DataFrame, truth: pd.DataFrame, early_warning_horizon: int,
                    critical_action_horizon: int) -> tuple[dict, pd.DataFrame]:
    """Engine-level recall and precision of each alert against the true remaining life (one prediction per
    test engine), with Wilson intervals, plus how the actions line up with what actually happened."""
    scored = risk_scores.merge(truth, on="engine_id", validate="one_to_one")
    metrics = {}
    for prefix, horizon in (("early_warning", early_warning_horizon), ("critical_action", critical_action_horizon)):
        failing = scored["true_rul"] <= horizon
        alert = scored[f"{prefix}_alert"] == 1
        for name, hits, n in (("recall", (alert & failing).sum(), failing.sum()),
                              ("precision", (alert & failing).sum(), alert.sum())):
            low, high = wilson_interval(int(hits), int(n))
            metrics[f"{prefix}_engine_{name}"] = {"value": hits / n if n else math.nan, "step": 0}
            metrics[f"{prefix}_engine_{name}_ci_low"] = {"value": low, "step": 0}
            metrics[f"{prefix}_engine_{name}_ci_high"] = {"value": high, "step": 0}
    summary = (scored.groupby("action")
               .agg(engines=("engine_id", "size"), avg_true_rul=("true_rul", "mean"),
                    truly_within_early_warning=("true_rul", lambda s: int((s <= early_warning_horizon).sum())),
                    truly_within_critical_action=("true_rul", lambda s: int((s <= critical_action_horizon).sum())))
               .sort_values("avg_true_rul").round(1).reset_index())
    return metrics, summary
