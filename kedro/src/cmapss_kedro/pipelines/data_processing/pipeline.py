"""Raw text files -> staged telemetry -> training and scoring feature tables."""

from kedro.pipeline import Node, Pipeline

from .nodes import build_scoring_features, build_training_features, stack_rul_truth, stage_telemetry

DATASETS = ["FD001", "FD002", "FD003", "FD004"]


def create_pipeline(**kwargs) -> Pipeline:
    # Each raw file comes from the "{split}_{dataset}" catalog factory; the dict keys become the
    # node's keyword arguments, so the dataset name travels with its frame.
    return Pipeline([
        Node(stage_telemetry, {d: f"train_{d}" for d in DATASETS}, "train_telemetry", name="stage_train_telemetry"),
        Node(stage_telemetry, {d: f"test_{d}" for d in DATASETS}, "test_telemetry", name="stage_test_telemetry"),
        Node(stack_rul_truth, {d: f"RUL_{d}" for d in DATASETS}, "test_rul_truth", name="stack_rul_truth"),
        Node(build_training_features, ["train_telemetry", "params:rul_cap"], "model_input",
             name="build_training_features"),
        Node(build_scoring_features, "test_telemetry", "latest_test_features", name="build_scoring_features"),
    ])
