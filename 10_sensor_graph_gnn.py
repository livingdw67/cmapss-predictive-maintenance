"""Graph neural network over engine sensors, compared with XGBoost on the same engines.

Each sample is a graph: 14 sensor nodes, each carrying its last 30 cycles of readings.
A GRU encodes each sensor's history and GATv2 layers pass messages between sensors along
one of four edge sets (physics, correlation, full, none). One multitask head predicts RUL and
the two maintenance flags. Thresholds are tuned for F2 on validation engines, then the test
engines are scored once, exactly as in evaluate_local.py.

Usage:
    python 01_fetch_data.py
    python 10_sensor_graph_gnn.py                # all graphs, 3 seeds (~2.5 hours on 8 CPU cores)
    python 10_sensor_graph_gnn.py --quick        # physics graph, 1 seed, 5 epochs
"""
import argparse
import json
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, fbeta_score, precision_score, recall_score
from xgboost import XGBClassifier, XGBRegressor

import evaluate_local
from evaluate_local import RUL_CAP, TARGETS, best_f2_threshold
from gnn.data import NODE_NAMES, make_windows, prepare
from gnn.graphs import build_graphs
from gnn.model import SensorGAT

POS_WEIGHT = {"EARLY_WARNING": 3.4, "CRITICAL_ACTION": 13.0}  # negatives / positives on train


def classification_metrics(val_y, val_p, test_y, test_p):
    thresh, val_f2 = best_f2_threshold(val_y, val_p)
    pred = test_p >= thresh
    return {"threshold": round(float(thresh), 2), "val_f2": val_f2,
            "test_f2": fbeta_score(test_y, pred, beta=2), "recall": recall_score(test_y, pred),
            "precision": precision_score(test_y, pred), "pr_auc": average_precision_score(test_y, test_p)}


def nasa_score(true, pred):
    """PHM08 asymmetric score: late predictions (overestimating RUL) cost more than early ones."""
    d = pred - true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def official_metrics(official, rul_pred):
    """RMSE and NASA score at each test engine's last cycle, per sub-dataset."""
    last = official.groupby("ENGINE_ID")["CYCLE"].transform("max").eq(official["CYCLE"]).to_numpy()
    true = np.minimum(official["TARGET_RUL"].to_numpy()[last], RUL_CAP)  # capped, as is standard
    pred, fd = rul_pred[last], official["ENGINE_ID"].str[:5].to_numpy()[last]
    out = {}
    for name in sorted(set(fd)):
        m = fd == name
        out[name] = {"rmse": float(np.sqrt(np.mean((pred[m] - true[m]) ** 2))), "score": nasa_score(true[m], pred[m])}
    return out


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# --- Baselines -----------------------------------------------------------------------------

def xgb_original():
    """The repo's existing model: 3 raw sensors + 5-cycle rolling stats, scale_pos_weight."""
    df = evaluate_local.build_feature_store()
    train, val, test = evaluate_local.split_engines(df)
    feats = evaluate_local.FEATURES
    out = {}
    for target in TARGETS:
        model = evaluate_local.fit_xgb(train[feats], train[target], weighted=True)
        out[target] = classification_metrics(val[target], model.predict_proba(val[feats])[:, 1],
                                             test[target], model.predict_proba(test[feats])[:, 1])
    return {"classification": out}


def window_features(X):
    """Flatten windows into tabular features: last value, mean, std and slope per sensor."""
    t = np.arange(X.shape[2]) - (X.shape[2] - 1) / 2
    slope = (X * t).sum(axis=2) / (t ** 2).sum()
    return np.hstack([X[:, :, -1], X.mean(axis=2), X.std(axis=2), slope])


def xgb_window(windows, parts, official):
    """Same inputs the GNN sees (14 normalized sensors x 30 cycles), flattened for XGBoost."""
    (Xtr, Xva, Xte, Xof), (train, val, test) = [window_features(w) for w in windows], parts
    out = {"classification": {}}
    for target in TARGETS:
        model = XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.08, eval_metric="aucpr",
                              scale_pos_weight=POS_WEIGHT[target], random_state=42, n_jobs=-1)
        model.fit(Xtr, train[target])
        out["classification"][target] = classification_metrics(
            val[target], model.predict_proba(Xva)[:, 1], test[target], model.predict_proba(Xte)[:, 1])
    reg = XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.08, random_state=42, n_jobs=-1)
    reg.fit(Xtr, train["TARGET_RUL"])
    out["test_rmse"] = rmse(reg.predict(Xte), test["TARGET_RUL"].to_numpy())
    out["official"] = official_metrics(official, np.clip(reg.predict(Xof), 0, RUL_CAP))
    return out


# --- GNN -----------------------------------------------------------------------------------

def predict(model, X, batch=2048):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            rul, ew, ca = model(X[i:i + batch])
            outs.append(torch.stack([rul * RUL_CAP, torch.sigmoid(ew), torch.sigmoid(ca)], 1))
    return torch.cat(outs).numpy()


def train_gnn(edge_index, data, seed, epochs, patience=4, batch=512, lr=1e-3, per_epoch=32_000):
    torch.manual_seed(seed)
    np.random.seed(seed)
    (Xtr, ytr), (Xva, yva) = data["train"], data["val"]
    model = SensorGAT(len(NODE_NAMES), edge_index)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = {t: torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT[t])) for t in TARGETS}

    def loss_fn(out, y):
        rul, ew, ca = out
        return (torch.nn.functional.mse_loss(rul, y[:, 0] / RUL_CAP) * 10
                + bce["EARLY_WARNING"](ew, y[:, 1]) + bce["CRITICAL_ACTION"](ca, y[:, 2]))

    best, best_state, bad = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        start = time.time()
        # Neighbouring windows overlap by 29 of 30 cycles, so each epoch samples a third of them
        for idx in torch.randperm(len(Xtr))[:per_epoch].split(batch):
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = np.mean([loss_fn(model(Xva[i:i + 2048]), yva[i:i + 2048]).item()
                                for i in range(0, len(Xva), 2048)])
        print(f"    epoch {epoch + 1:>2}  val loss {val_loss:.4f}  ({time.time() - start:.0f}s)", flush=True)
        if val_loss < best - 1e-4:
            best, bad = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def evaluate_gnn(model, data, parts, official_X, official):
    val, test = parts[1], parts[2]
    pv, pt = predict(model, data["val"][0]), predict(model, data["test"][0])
    out = {"classification": {}}
    for j, target in enumerate(TARGETS, start=1):
        out["classification"][target] = classification_metrics(val[target], pv[:, j], test[target], pt[:, j])
    out["test_rmse"] = rmse(pt[:, 0], test["TARGET_RUL"].to_numpy())
    out["official"] = official_metrics(official, predict(model, official_X)[:, 0])
    return out


def attention_by_health(model, X, rul, edge_names):
    """Mean first-layer attention per edge for near-failure (RUL <= 15) vs healthy (RUL = cap) windows."""
    model.eval()
    result = {}
    for label, mask in [("near_failure", rul <= 15), ("healthy", rul >= RUL_CAP)]:
        sample = X[torch.from_numpy(np.flatnonzero(mask)[:4000])]
        with torch.no_grad():
            *_, attention = model(sample, return_attention=True)
        edges, alpha = attention[0]
        n = len(NODE_NAMES)
        src, dst = edges[0] % n, edges[1] % n
        alpha = alpha.mean(dim=1)
        result[label] = {f"{NODE_NAMES[s]}->{NODE_NAMES[d]}": alpha[(src == s) & (dst == d)].mean().item()
                         for s, d in edge_names}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(8)

    parts, official = prepare()
    train, val, test = parts
    windows = [make_windows(d) for d in (train, val, test, official)]
    graphs = build_graphs(train)
    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    data = {name: (to_t(w), to_t(p[["TARGET_RUL", *TARGETS]].to_numpy(np.float32)))
            for name, w, p in zip(["train", "val", "test"], windows, parts)}
    official_X = to_t(windows[3])
    print(f"Windows: train {len(train):,} | val {len(val):,} | test {len(test):,} | official {len(official):,}")
    for name, e in graphs.items():
        print(f"  graph {name:<12} {e.size(1) // 2:>3} undirected edges")

    results = {}
    if not args.quick:
        print("\nXGBoost (original features)")
        results["xgb_original"] = xgb_original()
        print("XGBoost (GNN inputs, flattened)")
        results["xgb_window"] = xgb_window(windows, parts[:3], official)

    variants = ["physics"] if args.quick else ["none", "correlation", "physics", "full"]
    seeds = 1 if args.quick else args.seeds
    epochs = 5 if args.quick else args.epochs
    for name in variants:
        runs = []
        for seed in range(seeds):
            print(f"\nGNN graph={name} seed={seed}")
            model = train_gnn(graphs[name], data, seed, epochs)
            runs.append(evaluate_gnn(model, data, parts, official_X, official))
            print(f"  test RMSE {runs[-1]['test_rmse']:.2f} | " + " | ".join(
                f"{t} F2 {runs[-1]['classification'][t]['test_f2']:.4f}" for t in TARGETS))
            if name == "physics" and seed == 0:
                pairs = sorted({(int(s), int(d)) for s, d in graphs[name].t().tolist()})
                results["physics_attention"] = attention_by_health(
                    model, data["test"][0], test["TARGET_RUL"].to_numpy(), pairs)
                torch.save(model.state_dict(), "gnn_physics_seed0.pt")
        results[f"gnn_{name}"] = runs

    with open("gnn_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nSaved gnn_results.json")


if __name__ == "__main__":
    main()
