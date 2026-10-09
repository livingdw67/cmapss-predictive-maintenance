"""Precompute everything the public app's Modeling tab shows.

The free Streamlit host is too small to retrain these models on demand (the GNN took about
2.5 hours on 8 CPU cores), so this script trains every XGBoost configuration a visitor can pick
and packages the saved GNN results. Each XGBoost configuration is evaluated exactly like
evaluate_local.py and 10_sensor_graph_gnn.py: the F2 threshold is chosen on validation engines,
then the held-out test engines are scored once.

Usage:
    python 01_fetch_data.py
    python 17_build_model_explorer.py      # about 5 minutes; writes public_app/model_explorer.json
"""
import importlib.util
import itertools
import json
import os
import time

import numpy as np
from sklearn.metrics import average_precision_score, fbeta_score, precision_score, recall_score
from xgboost import XGBClassifier

import evaluate_local
from evaluate_local import TARGETS, best_f2_threshold
from gnn.data import NODE_NAMES, make_windows, prepare
from gnn.graphs import COMPONENTS, build_graphs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(BASE_DIR, "public_app", "model_explorer.json")

# The options shown as radio buttons; the middle values reproduce the published models
GRID = {"n_estimators": [100, 300, 600], "max_depth": [3, 5, 7], "learning_rate": [0.03, 0.08, 0.2]}
DEFAULT = {"n_estimators": 300, "max_depth": 7, "learning_rate": 0.08}
CURVE_THRESHOLDS = np.round(np.arange(0.02, 0.99, 0.02), 2)

spec = importlib.util.spec_from_file_location("gnn_script", os.path.join(BASE_DIR, "10_sensor_graph_gnn.py"))
gnn_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gnn_script)


def feature_sets():
    """The two inputs the repo's XGBoost models use, on the same engine splits."""
    df = evaluate_local.build_feature_store()
    train, val, test = evaluate_local.split_engines(df)
    feats = evaluate_local.FEATURES
    original = {"X": [p[feats].to_numpy() for p in (train, val, test)], "y": (train, val, test),
                "names": feats, "pos_weight": {t: (train[t] == 0).sum() / (train[t] == 1).sum() for t in TARGETS}}

    parts, _ = prepare()
    windows = [gnn_script.window_features(make_windows(p)) for p in parts]
    names = [f"{s} {stat}" for stat in ("last", "mean", "std", "slope") for s in NODE_NAMES]
    window = {"X": windows, "y": tuple(parts), "names": names, "pos_weight": gnn_script.POS_WEIGHT}
    return {"original": original, "window": window}


def evaluate_config(data, target, params):
    (Xtr, Xva, Xte), (tr, va, te) = data["X"], data["y"]
    model = XGBClassifier(**params, eval_metric="aucpr", scale_pos_weight=data["pos_weight"][target],
                          random_state=42, n_jobs=-1).fit(Xtr, tr[target])
    p_val, p_test = model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]
    thresh, val_f2 = best_f2_threshold(va[target], p_val)
    y_test = te[target].to_numpy()
    pred = p_test >= thresh
    curve = []
    for t in CURVE_THRESHOLDS:
        p = p_test >= t
        curve.append([float(t), round(fbeta_score(y_test, p, beta=2), 4),
                      round(recall_score(y_test, p), 4), round(precision_score(y_test, p, zero_division=0), 4)])
    gain = model.get_booster().get_score(importance_type="gain")
    importance = sorted(((data["names"][int(k[1:])], v) for k, v in gain.items()), key=lambda kv: -kv[1])[:8]
    total = sum(v for _, v in importance) or 1
    return {"threshold": round(float(thresh), 2), "val_f2": round(float(val_f2), 4),
            "test_f2": round(fbeta_score(y_test, pred, beta=2), 4), "recall": round(recall_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred), 4), "pr_auc": round(average_precision_score(y_test, p_test), 4),
            "curve": curve, "importance": [[n, round(v / total, 4)] for n, v in importance]}


def xgboost_grid():
    sets = feature_sets()
    configs = []
    combos = list(itertools.product(GRID["n_estimators"], GRID["max_depth"], GRID["learning_rate"]))
    for fs, (n, d, lr) in itertools.product(sets, combos):
        start = time.time()
        params = {"n_estimators": n, "max_depth": d, "learning_rate": lr}
        targets = {t: evaluate_config(sets[fs], t, params) for t in TARGETS}
        configs.append({"features": fs, **params, "targets": targets})
        print(f"  {fs:<8} trees={n:<3} depth={d} lr={lr:<4} "
              + " | ".join(f"{t} F2 {targets[t]['test_f2']:.4f}" for t in TARGETS) + f"  ({time.time() - start:.0f}s)")
    return configs


def check_defaults(configs):
    """The default configuration must reproduce the results already published in gnn_results.json."""
    saved = json.load(open(os.path.join(BASE_DIR, "gnn_results.json")))
    for fs, key in [("original", "xgb_original"), ("window", "xgb_window")]:
        cfg = next(c for c in configs if c["features"] == fs and all(c[k] == v for k, v in DEFAULT.items()))
        for t in TARGETS:
            ours, published = cfg["targets"][t]["test_f2"], saved[key]["classification"][t]["test_f2"]
            assert abs(ours - published) < 1e-3, f"{fs} {t}: {ours} != published {published}"
            print(f"  default {fs:<8} {t:<15} test F2 {ours:.4f} matches published {published:.4f}")


def summarize(runs):
    def stat(values):
        return {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values)), 4)}
    out = {t: stat([r["classification"][t]["test_f2"] for r in runs]) for t in TARGETS}
    out["test_rmse"] = stat([r["test_rmse"] for r in runs])
    out["official_rmse"] = stat([np.mean([v["rmse"] for v in r["official"].values()]) for r in runs])
    out["seeds"] = [{t: round(r["classification"][t]["test_f2"], 4) for t in TARGETS} | {"test_rmse": round(r["test_rmse"], 2)}
                    for r in runs]
    return out


def gnn_section():
    saved = json.load(open(os.path.join(BASE_DIR, "gnn_results.json")))
    parts, _ = prepare()
    graphs = build_graphs(parts[0])
    out = {"nodes": NODE_NAMES, "components": COMPONENTS, "graphs": {}}
    for name, edge_index in graphs.items():
        pairs = sorted({tuple(sorted((NODE_NAMES[s], NODE_NAMES[d]))) for s, d in edge_index.t().tolist()})
        out["graphs"][name] = {"edges": [list(p) for p in pairs], "results": summarize(saved[f"gnn_{name}"])}
    xw = saved["xgb_window"]
    out["xgb_window"] = {**{t: round(xw["classification"][t]["test_f2"], 4) for t in TARGETS},
                         "test_rmse": round(xw["test_rmse"], 2),
                         "official_rmse": round(float(np.mean([v["rmse"] for v in xw["official"].values()])), 2)}
    # Physics-graph attention: average both directions of each edge, then rank by the rise near failure
    att = saved["physics_attention"]
    undirected = {}
    for label in ("near_failure", "healthy"):
        for edge, value in att[label].items():
            key = " – ".join(sorted(edge.split("->")))
            undirected.setdefault(key, {}).setdefault(label, []).append(value)
    rows = [[k, round(float(np.mean(v["near_failure"])), 4), round(float(np.mean(v["healthy"])), 4)]
            for k, v in undirected.items()]
    out["attention"] = sorted(rows, key=lambda r: -(r[1] - r[2]))[:8]
    return out


def main():
    print("Training the XGBoost grid (2 feature sets x 27 settings x 2 targets)...")
    configs = xgboost_grid()
    check_defaults(configs)
    print("Packaging GNN results...")
    result = {"xgboost": {"grid": GRID, "default": DEFAULT, "configs": configs}, "gnn": gnn_section()}
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, separators=(",", ":"))
    print(f"Saved {os.path.relpath(OUT_FILE, BASE_DIR)} ({os.path.getsize(OUT_FILE) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
