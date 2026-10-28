import math

import numpy as np
import pandas as pd
import pytest

from cmapss_kedro.pipeline_registry import register_pipelines
from cmapss_kedro.pipelines.data_processing.nodes import (
    FEATURES, RAW_COLUMNS, build_scoring_features, build_training_features, stack_rul_truth, stage_telemetry)
from cmapss_kedro.pipelines.scoring.nodes import ACTIONS, score_fleet, wilson_interval
from cmapss_kedro.pipelines.training.nodes import split_engines


def raw_file(engines: dict[int, int]) -> pd.DataFrame:
    """A raw C-MAPSS frame (unnamed columns, as loaded) with `cycles` rows per engine number."""
    rows = [[engine, cycle] + [0.0] * 3 + [float(cycle)] * 21
            for engine, cycles in engines.items() for cycle in range(1, cycles + 1)]
    return pd.DataFrame(rows, columns=range(len(RAW_COLUMNS)))


def test_staging_keys_engines_by_dataset():
    staged = stage_telemetry(FD001=raw_file({1: 3}), FD002=raw_file({1: 2}))
    assert sorted(staged["engine_id"].unique()) == ["FD001_001", "FD002_001"]
    assert {"hpc_outlet_temp", "physical_core_speed", "bypass_ratio"} <= set(staged.columns)


def test_training_features_cap_rul_and_roll_within_engine():
    staged = stage_telemetry(FD001=raw_file({1: 200, 2: 6}))
    features = build_training_features(staged, rul_cap=130)
    first = features[features["engine_id"] == "FD001_001"]
    assert first["target_rul"].iloc[0] == 130 and first["target_rul"].iloc[-1] == 0
    second = features[features["engine_id"] == "FD001_002"].reset_index(drop=True)
    # Sensors equal the cycle number, so the 5-cycle mean at cycle 6 is mean(2..6) = 4, and the
    # window must not reach back into engine 1's cycles
    assert second.loc[5, "hpc_temp_ma_5"] == 4.0
    assert second.loc[0, "hpc_temp_ma_5"] == 1.0
    assert math.isnan(second.loc[0, "hpc_temp_std_5"])


def test_scoring_features_keep_last_cycle_per_engine():
    latest = build_scoring_features(stage_telemetry(FD001=raw_file({1: 7, 2: 3})))
    assert latest.set_index("engine_id")["cycle"].to_dict() == {"FD001_001": 7, "FD001_002": 3}
    assert list(latest.columns) == ["engine_id", "cycle", *FEATURES]


def test_rul_truth_numbers_engines_in_file_order():
    truth = stack_rul_truth(FD003=pd.DataFrame({0: [112, 98]}))
    assert truth.to_dict("list") == {"engine_id": ["FD003_001", "FD003_002"], "dataset": ["FD003"] * 2,
                                     "true_rul": [112, 98]}


def test_split_never_puts_an_engine_in_two_sets():
    rng = np.random.default_rng(0)
    rows = [{"engine_id": f"E{e:03d}", "cycle": c} for e in range(100) for c in range(1, rng.integers(130, 400))]
    split = {"validation_share": 0.2, "test_share": 0.2, "lifespan_bins": [0, 200, 300, 400, 600], "seed": 42}
    parts = split_engines(pd.DataFrame(rows), split)
    ids = [set(p["engine_id"]) for p in parts]
    assert [len(i) for i in ids] == [60, 20, 20]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


class FixedAlert:
    def __init__(self, alerts):
        self.alerts = alerts

    def predict(self, features):
        return pd.DataFrame({"probability": [float(a) for a in self.alerts], "alert": self.alerts}, index=features.index)


def test_score_fleet_applies_the_decision_matrix():
    features = pd.DataFrame({"engine_id": list("abcd"), "cycle": [1, 2, 3, 4]})
    scores = score_fleet(features, FixedAlert([0, 1, 1, 0]), FixedAlert([0, 0, 1, 1]))
    assert list(scores["action"]) == [ACTIONS[(0, 0)], ACTIONS[(1, 0)], ACTIONS[(1, 1)], ACTIONS[(0, 1)]]


@pytest.mark.parametrize("hits, n, low, high", [(9, 10, 0.596, 0.982), (10, 10, 0.722, 1.0), (0, 10, 0.0, 0.278)])
def test_wilson_interval_matches_published_values(hits, n, low, high):
    assert wilson_interval(hits, n) == pytest.approx((low, high), abs=1e-3)


def test_default_pipeline_trains_both_horizons_and_packages_inference():
    pipelines = register_pipelines()
    names = {n.name for n in pipelines["__default__"].nodes}
    assert {"early_warning.train_classifier", "critical_action.train_classifier", "score_fleet"} <= names
    assert pipelines["__default__"].inference.inputs() == {
        "latest_test_features", "early_warning.model", "critical_action.model"}
