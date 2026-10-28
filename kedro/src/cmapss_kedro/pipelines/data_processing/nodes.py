"""Raw NASA files to model-ready features: the staging, mart and feature-store steps of
`databricks/src/02_transform.py`, in pandas."""

import pandas as pd

RAW_COLUMNS = ["id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"sensor_{i}" for i in range(1, 22)]
SENSOR_NAMES = [
    "fan_inlet_temp", "lpc_outlet_temp", "hpc_outlet_temp", "lpt_outlet_temp", "fan_inlet_pressure",
    "bypass_duct_pressure", "hpc_outlet_pressure", "physical_fan_speed", "physical_core_speed",
    "engine_pressure_ratio", "static_pressure_hpc_outlet", "ratio_fuel_flow_ps30", "corrected_fan_speed",
    "corrected_core_speed", "bypass_ratio", "burner_fuel_air_ratio", "bleed_enthalpy", "demanded_fan_speed",
    "demanded_corrected_fan_speed", "hpt_coolant_bleed", "lpt_coolant_bleed",
]
WINDOW = 5
# Rolling means (trend) and standard deviations (volatility rises as parts degrade) over the last 5 cycles
ROLLING = {
    "hpc_temp_ma_5": ("hpc_outlet_temp", "mean"),
    "core_speed_ma_5": ("physical_core_speed", "mean"),
    "bypass_ratio_ma_5": ("bypass_ratio", "mean"),
    "hpc_temp_std_5": ("hpc_outlet_temp", "std"),
    "core_speed_std_5": ("physical_core_speed", "std"),
}
FEATURES = ["hpc_outlet_temp", "physical_core_speed", "bypass_ratio", *ROLLING]


def stage_telemetry(**files: pd.DataFrame) -> pd.DataFrame:
    """Stack the FD00x files (keyword = dataset name), name the settings and sensors, and key engines by
    dataset, since engine numbers restart at 1 in each file."""
    frames = []
    for dataset, raw in files.items():
        raw = raw.set_axis(RAW_COLUMNS, axis=1)
        staged = pd.DataFrame({
            "dataset": dataset,
            "engine_id": dataset + "_" + raw["id"].astype(int).map("{:03d}".format),
            "cycle": raw["cycle"].astype(int),
            "altitude_setting": raw["op_setting_1"],
            "mach_number_setting": raw["op_setting_2"],
            "throttle_resolver_angle": raw["op_setting_3"],
        })
        for i, name in enumerate(SENSOR_NAMES, start=1):
            staged[name] = raw[f"sensor_{i}"].astype(float)
        frames.append(staged)
    return pd.concat(frames, ignore_index=True).sort_values(["engine_id", "cycle"], ignore_index=True)


def stack_rul_truth(**files: pd.DataFrame) -> pd.DataFrame:
    """True remaining life of each test engine. RUL_FD00x has one value per line in engine order."""
    return pd.concat([
        pd.DataFrame({"engine_id": [f"{dataset}_{n:03d}" for n in range(1, len(raw) + 1)],
                      "dataset": dataset, "true_rul": raw.iloc[:, 0].astype(int).to_numpy()})
        for dataset, raw in files.items()
    ], ignore_index=True)


def _add_rolling(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Rolling features per engine over the current and 4 preceding cycles. The first cycle's standard
    deviation is missing (one value), as with SQL's STDDEV; XGBoost handles the gap."""
    df = telemetry.sort_values(["engine_id", "cycle"], ignore_index=True)
    grouped = df.groupby("engine_id")
    for feature, (column, how) in ROLLING.items():
        rolled = grouped[column].rolling(WINDOW, min_periods=1)
        df[feature] = getattr(rolled, how)().reset_index(level=0, drop=True)
    return df


def build_training_features(telemetry: pd.DataFrame, rul_cap: int) -> pd.DataFrame:
    """Feature table for the run-to-failure engines. The last recorded cycle is the failure, so remaining
    life is cycles to the last one, capped (piecewise RUL): a healthy engine looks the same at 250 or 200
    cycles left."""
    df = _add_rolling(telemetry)
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["target_rul"] = (max_cycle - df["cycle"]).clip(upper=rul_cap)
    return df[["engine_id", "cycle", "target_rul", *FEATURES]]


def build_scoring_features(test_telemetry: pd.DataFrame) -> pd.DataFrame:
    """Features at the last observed cycle of each test engine (histories stop before failure)."""
    df = _add_rolling(test_telemetry)
    latest = df.groupby("engine_id").tail(1)
    return latest[["engine_id", "cycle", *FEATURES]].reset_index(drop=True)
