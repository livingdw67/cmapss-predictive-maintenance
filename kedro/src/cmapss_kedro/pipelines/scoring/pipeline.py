"""Inference (what gets packaged as the MLflow model) and its evaluation against true RUL."""

from kedro.pipeline import Node, Pipeline

from .nodes import evaluate_alerts, score_fleet


def create_inference_pipeline() -> Pipeline:
    return Pipeline([
        Node(score_fleet, ["latest_test_features", "early_warning.model", "critical_action.model"], "risk_scores",
             name="score_fleet"),
    ])


def create_evaluation_pipeline() -> Pipeline:
    return Pipeline([
        Node(evaluate_alerts, ["risk_scores", "test_rul_truth", "params:early_warning.horizon",
                               "params:critical_action.horizon"], ["fleet_metrics", "action_summary"],
             name="evaluate_alerts"),
    ])


def create_pipeline(**kwargs) -> Pipeline:
    return create_inference_pipeline() + create_evaluation_pipeline()
