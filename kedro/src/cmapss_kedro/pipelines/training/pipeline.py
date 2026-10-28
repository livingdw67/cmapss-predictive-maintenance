"""Engine split, then one classifier per alert horizon.

The classifier pipeline is written once and instantiated under two namespaces, `early_warning` and
`critical_action`. Kedro prefixes its outputs and its `params:horizon` with the namespace, so each copy
reads its own horizon from parameters_training.yml and writes its own model and metrics.
"""

from kedro.pipeline import Node, Pipeline

from .nodes import split_engines, train_classifier

TARGETS = ["early_warning", "critical_action"]


def _classifier(target: str) -> Pipeline:
    template = Pipeline([
        Node(train_classifier,
             ["train", "validation", "test", "params:horizon", "params:xgboost", "params:threshold_grid"],
             ["model", "strategy_comparison", "metrics"], name="train_classifier"),
    ])
    return Pipeline(template, namespace=target, inputs={"train", "validation", "test"},
                    parameters={"params:xgboost", "params:threshold_grid"})


def create_pipeline(**kwargs) -> Pipeline:
    split = Pipeline([
        Node(split_engines, ["model_input", "params:split"], ["train", "validation", "test"], name="split_engines"),
    ])
    return split + sum(_classifier(t) for t in TARGETS)
