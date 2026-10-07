"""Sensor windows for the GNN: load C-MAPSS, normalize by operating condition, cut sliding windows.

Every row (engine, cycle) becomes one sample: the last WINDOW cycles of the 14 informative
sensors. Engines with fewer than WINDOW cycles of history are left-padded with their first
reading, so every row the XGBoost baseline scores is also scored here.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate_local import DATA_DIR, DATASETS, RUL_CAP, TARGETS, split_engines  # noqa: E402

WINDOW = 30

# 14 of the 21 sensors carry degradation signal. The other 7 (T2, P2, P15, epr, farB,
# Nf_dmd, PCNfR_dmd) only move with operating condition and are flat once it is removed.
SENSORS = {
    "sensor_2": "T24",       # LPC outlet temperature
    "sensor_3": "T30",       # HPC outlet temperature
    "sensor_4": "T50",       # LPT outlet temperature
    "sensor_7": "P30",       # HPC outlet pressure
    "sensor_8": "Nf",        # physical fan speed
    "sensor_9": "Nc",        # physical core speed
    "sensor_11": "Ps30",     # HPC outlet static pressure
    "sensor_12": "phi",      # fuel flow / Ps30
    "sensor_13": "NRf",      # corrected fan speed
    "sensor_14": "NRc",      # corrected core speed
    "sensor_15": "BPR",      # bypass ratio
    "sensor_17": "htBleed",  # bleed enthalpy
    "sensor_20": "W31",      # HPT coolant bleed
    "sensor_21": "W32",      # LPT coolant bleed
}
NODE_NAMES = list(SENSORS.values())
COLUMNS = ["id", "cycle", "op1", "op2", "op3"] + [f"sensor_{i}" for i in range(1, 22)]


def _read(kind, fd):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{kind}_{fd}.txt"), sep=r"\s+", header=None).iloc[:, :26]
    df.columns = COLUMNS
    df["ENGINE_ID"] = fd + "_" + df["id"].astype(str).str.zfill(3)
    # The six flight conditions have distinct altitudes (0, 10, 20, 25, 35, 42 kft)
    df["CONDITION"] = df["op1"].round().astype(int)
    return df.rename(columns={"cycle": "CYCLE", **SENSORS})


def load_train():
    df = pd.concat([_read("train", fd) for fd in DATASETS], ignore_index=True)
    df = df.sort_values(["ENGINE_ID", "CYCLE"]).reset_index(drop=True)
    max_cycle = df.groupby("ENGINE_ID")["CYCLE"].transform("max")
    df["TARGET_RUL"] = (max_cycle - df["CYCLE"]).clip(upper=RUL_CAP)
    for name, horizon in TARGETS.items():
        df[name] = (df["TARGET_RUL"] <= horizon).astype(int)
    return df


def load_official_test():
    """NASA's held-out test trajectories, cut off before failure, with true RUL at the last cycle."""
    frames = []
    for fd in DATASETS:
        df = _read("test", fd)
        rul = pd.read_csv(os.path.join(DATA_DIR, f"RUL_{fd}.txt"), header=None)[0].to_numpy()
        last = df.groupby("ENGINE_ID")["CYCLE"].transform("max")
        df["TARGET_RUL"] = rul[df["id"] - 1] + (last - df["CYCLE"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sort_values(["ENGINE_ID", "CYCLE"]).reset_index(drop=True)


class ConditionScaler:
    """Z-score each sensor within each operating condition, using training engines only."""

    def fit(self, df):
        grouped = df.groupby("CONDITION")[NODE_NAMES]
        self.mean, self.std = grouped.mean(), grouped.std().replace(0, 1)
        return self

    def transform(self, df):
        out = df.copy()
        cond = df["CONDITION"]
        out[NODE_NAMES] = (df[NODE_NAMES].to_numpy() - self.mean.loc[cond].to_numpy()) / self.std.loc[cond].to_numpy()
        return out


def make_windows(df, window=WINDOW):
    """Return X with shape (rows, sensors, window), aligned with df's row order."""
    values = df[NODE_NAMES].to_numpy(np.float32)
    X = np.empty((len(df), len(NODE_NAMES), window), np.float32)
    starts = np.flatnonzero(df["ENGINE_ID"].ne(df["ENGINE_ID"].shift()).to_numpy())
    ends = np.append(starts[1:], len(df))
    for s, e in zip(starts, ends):
        eng = values[s:e]
        padded = np.vstack([np.repeat(eng[:1], window - 1, axis=0), eng])
        # padded[i : i + window] is the window ending at row i of this engine
        X[s:e] = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)
    return X


def prepare(seed=42):
    """Train/val/test splits (same engines as the XGBoost baseline) plus the official NASA test set."""
    df = load_train()
    parts = split_engines(df, seed=seed)
    scaler = ConditionScaler().fit(parts[0])
    parts = [scaler.transform(p).reset_index(drop=True) for p in parts]
    official = scaler.transform(load_official_test())
    return parts, official
