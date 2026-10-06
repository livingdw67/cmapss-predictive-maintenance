"""Local, Snowflake-free reproduction of the pipeline and modeling workflow.

Mirrors the SQL layers (STAGING -> CORE -> MARTS -> FEATURE_STORE) and the
modeling notebook in pandas, so results can be verified without a Snowflake account.

Usage:
    python 01_fetch_data.py      # downloads NASA C-MAPSS into data/raw
    python evaluate_local.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
DATASETS = ["FD001", "FD002", "FD003", "FD004"]
RUL_CAP = 130
TARGETS = {"EARLY_WARNING": 50, "CRITICAL_ACTION": 15}
FEATURES = ["HPC_OUTLET_TEMP", "PHYSICAL_CORE_SPEED", "BYPASS_RATIO",
            "HPC_TEMP_MA_5", "CORE_SPEED_MA_5", "BYPASS_RATIO_MA_5",
            "HPC_TEMP_STD_5", "CORE_SPEED_STD_5"]


def build_feature_store():
    """STAGING + CORE + MARTS + FEATURE_STORE in pandas."""
    frames = []
    for fd in DATASETS:
        df = pd.read_csv(os.path.join(DATA_DIR, f"train_{fd}.txt"), sep=r"\s+", header=None).iloc[:, :26]
        df.columns = ["id", "cycle", "op1", "op2", "op3"] + [f"sensor_{i}" for i in range(1, 22)]
        # Engine numbers restart at 1 in every file, so the dataset is part of the key
        df["ENGINE_ID"] = fd + "_" + df["id"].astype(str).str.zfill(3)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True).rename(columns={
        "cycle": "CYCLE", "sensor_3": "HPC_OUTLET_TEMP",
        "sensor_9": "PHYSICAL_CORE_SPEED", "sensor_15": "BYPASS_RATIO"})
    df = df.sort_values(["ENGINE_ID", "CYCLE"]).reset_index(drop=True)

    max_cycle = df.groupby("ENGINE_ID")["CYCLE"].transform("max")
    df["TARGET_RUL"] = (max_cycle - df["CYCLE"]).clip(upper=RUL_CAP)

    # ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    roll = df.groupby("ENGINE_ID")[["HPC_OUTLET_TEMP", "PHYSICAL_CORE_SPEED", "BYPASS_RATIO"]].rolling(5, min_periods=1)
    means = roll.mean().reset_index(level=0, drop=True)
    stds = roll.std().reset_index(level=0, drop=True)
    df["HPC_TEMP_MA_5"] = means["HPC_OUTLET_TEMP"]
    df["CORE_SPEED_MA_5"] = means["PHYSICAL_CORE_SPEED"]
    df["BYPASS_RATIO_MA_5"] = means["BYPASS_RATIO"]
    df["HPC_TEMP_STD_5"] = stds["HPC_OUTLET_TEMP"]
    df["CORE_SPEED_STD_5"] = stds["PHYSICAL_CORE_SPEED"]

    for name, horizon in TARGETS.items():
        df[name] = (df["TARGET_RUL"] <= horizon).astype(int)
    return df


def split_engines(df, seed=42):
    """60/20/20 engine-level split, stratified by lifespan. No engine appears in two sets."""
    stats = df.groupby("ENGINE_ID")["CYCLE"].max().rename("max_cycle").reset_index()
    stats["bucket"] = pd.cut(stats["max_cycle"], bins=[0, 200, 300, 400, 600],
                             labels=["short", "medium", "long", "very_long"])
    train, rest = train_test_split(stats, test_size=0.4, stratify=stats["bucket"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["bucket"], random_state=seed)
    return [df[df["ENGINE_ID"].isin(s["ENGINE_ID"])] for s in (train, val, test)]


def manual_smote(X, y, seed=42, k=5):
    rng = np.random.default_rng(seed)
    X_min, X_maj = X[y == 1].to_numpy(), X[y == 0].to_numpy()
    X_min_filled = np.nan_to_num(X_min)
    neighbors = NearestNeighbors(n_neighbors=k).fit(X_min_filled).kneighbors(X_min_filled, return_distance=False)
    idx = rng.integers(0, len(X_min), len(X_maj) - len(X_min))
    nbr = neighbors[idx, rng.integers(0, k, len(idx))]
    synthetic = X_min[idx] + rng.random((len(idx), 1)) * (X_min[nbr] - X_min[idx])
    X_out = pd.DataFrame(np.vstack([X.to_numpy(), synthetic]), columns=X.columns)
    return X_out, pd.Series(np.concatenate([y.to_numpy(), np.ones(len(idx), dtype=int)]))


def undersample_6x(X, y, seed=42):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = rng.choice(np.where(y == 0)[0], size=min(6 * len(pos), (y == 0).sum()), replace=False)
    keep = rng.permutation(np.concatenate([pos, neg]))
    return X.iloc[keep], y.iloc[keep]


def fit_xgb(X, y, weighted):
    return XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.08, eval_metric="aucpr",
                         scale_pos_weight=(y == 0).sum() / (y == 1).sum() if weighted else 1.0,
                         random_state=42, n_jobs=-1).fit(X, y)


def best_f2_threshold(y_true, proba):
    thresholds = np.arange(0.05, 0.95, 0.01)
    scores = [fbeta_score(y_true, proba >= t, beta=2) for t in thresholds]
    return thresholds[int(np.argmax(scores))], max(scores)


def main():
    df = build_feature_store()
    train, val, test = split_engines(df)
    print(f"Records: {len(df):,} | Engines: {df['ENGINE_ID'].nunique()}")
    for name, part in [("Train", train), ("Validation", val), ("Test", test)]:
        print(f"  {name:<10} {len(part):>7,} records | {part['ENGINE_ID'].nunique():>3} engines")

    for target in TARGETS:
        print(f"\n=== {target} (RUL <= {TARGETS[target]}) | positive rate {df[target].mean():.1%} ===")
        X_tr, y_tr = train[FEATURES], train[target]
        strategies = {
            "scale_pos_weight": (X_tr, y_tr, True),
            "SMOTE": (*manual_smote(X_tr, y_tr), False),
            "undersample_6x": (*undersample_6x(X_tr, y_tr), False),
        }
        # Choose strategy and threshold on VALIDATION engines only
        chosen = None
        print(f"  {'Strategy':<17} {'Val threshold':>13} {'Val F2':>8}")
        for name, (X, y, weighted) in strategies.items():
            model = fit_xgb(X, y, weighted)
            thresh, f2 = best_f2_threshold(val[target], model.predict_proba(val[FEATURES])[:, 1])
            print(f"  {name:<17} {thresh:>13.2f} {f2:>8.4f}")
            if chosen is None or f2 > chosen[3]:
                chosen = (name, model, thresh, f2)

        # Score the held-out TEST engines once
        name, model, thresh, _ = chosen
        pred = model.predict_proba(test[FEATURES])[:, 1] >= thresh
        print(f"  Selected: {name} @ {thresh:.2f} -> TEST F2 {fbeta_score(test[target], pred, beta=2):.4f} | "
              f"Recall {recall_score(test[target], pred):.4f} | Precision {precision_score(test[target], pred):.4f}")


if __name__ == "__main__":
    main()
