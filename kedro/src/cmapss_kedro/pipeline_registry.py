"""Project pipelines.

`__default__` runs everything in one MLflow run. It is wrapped with kedro-mlflow's `pipeline_ml_factory`,
so at the end of the run the inference pipeline (score_fleet), with both trained classifiers as artifacts,
is logged and registered as a single MLflow model that takes `latest_test_features` and returns
`risk_scores`.
"""

from kedro.pipeline import Pipeline
from kedro_mlflow.pipeline import pipeline_ml_factory

from cmapss_kedro.pipelines import data_processing, scoring, training


def register_pipelines() -> dict[str, Pipeline]:
    inference = scoring.pipeline.create_inference_pipeline()
    full = (data_processing.create_pipeline() + training.create_pipeline() + inference
            + scoring.pipeline.create_evaluation_pipeline())
    return {
        "__default__": pipeline_ml_factory(
            training=full, inference=inference, input_name="latest_test_features",
            log_model_kwargs={"name": "fleet_scorer", "signature": "auto",
                              "registered_model_name": "cmapss_fleet_scorer"}),
        "data_processing": data_processing.create_pipeline(),
        "training": training.create_pipeline(),
        "scoring": scoring.create_pipeline(),
    }
