# Modeling tab for the public app: explore XGBoost settings and GNN graph types.
# Everything shown is precomputed by 17_build_model_explorer.py (the free host is too
# small to retrain on demand), so picking options is instant and costs no credits.
import json
import os

import altair as alt
import pandas as pd
import streamlit as st

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_explorer.json")
TARGET_LABELS = {"EARLY_WARNING": "Early warning (≤ 50 cycles left)",
                 "CRITICAL_ACTION": "Critical action (≤ 15 cycles left)"}
FEATURE_LABELS = {"original": "Original: 3 sensors + 5-cycle rolling stats (8 features)",
                  "window": "Sensor windows: 14 normalized sensors × 30 cycles (56 features)"}
GRAPH_INFO = {
    "physics": ("Physics", "Sensors are linked when they sit on the same component, on neighboring stages of the gas path "
                "(fan → LPC → HPC → combustor → HPT → LPT), on the same shaft, or are joined by bleed flows."),
    "correlation": ("Correlation", "Each sensor is linked to the 3 sensors it moves most closely with on training engines."),
    "full": ("Full", "Every pair of sensors is linked, so attention has to find the structure on its own."),
    "none": ("None (ablation)", "No links: each sensor is read on its own and no information passes between sensors. "
             "This tests whether the graph helps at all."),
}
# Sensors laid out along the gas path, left to right
NODE_POS = {"Nf": (0, 1), "NRf": (0, 0), "BPR": (0, -1), "T24": (1, 0),
            "T30": (2, 1.2), "P30": (2, 0.4), "Ps30": (2, -0.4), "htBleed": (2, -1.2),
            "phi": (3, 0.3), "Nc": (3, -1.4), "NRc": (4, -1.4), "W31": (4, 0.6),
            "T50": (5, 0.6), "W32": (5, -0.6)}
COMPONENT_LABELS = {"fan": "Fan", "lpc": "LPC", "hpc": "HPC", "combustor": "Combustor", "hpt": "HPT", "lpt": "LPT",
                    "shaft": "Core shaft speed"}
F2_NOTE = ("**F2** is the score used throughout: it weights recall twice as heavily as precision, because missing an "
           "engine that is about to fail costs far more than an extra inspection.")


@st.cache_data
def load():
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


def fmt(v, digits=3):
    return f"{v:.{digits}f}"


def render_xgboost(data):
    xgb = data["xgboost"]
    grid, default = xgb["grid"], xgb["default"]
    st.markdown("Two classifiers flag engines for maintenance. Pick a feature set and hyperparameters to see how "
                "the tuned model scores on engines it never saw. " + F2_NOTE)
    st.caption("For every setting, the decision threshold is chosen to maximize F2 on validation engines, then the "
               "held-out test engines are scored once. All 54 settings were trained ahead of time.")

    features = st.radio("Features", list(FEATURE_LABELS), index=1, format_func=FEATURE_LABELS.get, horizontal=True)
    c1, c2, c3 = st.columns(3)
    n = c1.radio("Number of trees", grid["n_estimators"], index=grid["n_estimators"].index(default["n_estimators"]), horizontal=True)
    d = c2.radio("Max tree depth", grid["max_depth"], index=grid["max_depth"].index(default["max_depth"]), horizontal=True)
    lr = c3.radio("Learning rate", grid["learning_rate"], index=grid["learning_rate"].index(default["learning_rate"]), horizontal=True)

    configs = [c for c in xgb["configs"] if c["features"] == features]
    pick = lambda **p: next(c for c in configs if all(c[k] == v for k, v in p.items()))
    cfg, base = pick(n_estimators=n, max_depth=d, learning_rate=lr), pick(**default)
    is_default = cfg is base

    cols = st.columns(2)
    for col, (t, label) in zip(cols, TARGET_LABELS.items()):
        r, b = cfg["targets"][t], base["targets"][t]
        with col:
            st.markdown(f"**{label}**")
            st.metric("Test F2", fmt(r["test_f2"]),
                      None if is_default else f"{r['test_f2'] - b['test_f2']:+.3f} vs default settings")
            st.caption(f"Recall {fmt(r['recall'])} · Precision {fmt(r['precision'])} · PR-AUC {fmt(r['pr_auc'])} · "
                       f"Threshold {r['threshold']:.2f} · Validation F2 {fmt(r['val_f2'])}")

    # F2 across every threshold on test engines, with the validation-chosen threshold marked
    rows = [{"Threshold": c[0], "F2": c[1], "Target": TARGET_LABELS[t].split(" (")[0]}
            for t in TARGET_LABELS for c in cfg["targets"][t]["curve"]]
    chosen = pd.DataFrame([{"Threshold": cfg["targets"][t]["threshold"], "Target": TARGET_LABELS[t].split(" (")[0]}
                           for t in TARGET_LABELS])
    color = alt.Color("Target:N", scale=alt.Scale(domain=["Early warning", "Critical action"], range=["#2a78d6", "#eb6834"]),
                      legend=alt.Legend(orient="bottom", title=None))
    # Zoom to the range the curves cover; on a 0-1 axis they look flat
    low = max(0.0, min(r["F2"] for r in rows) - 0.02)
    lines = alt.Chart(pd.DataFrame(rows)).mark_line(strokeWidth=2).encode(
        x=alt.X("Threshold:Q", title="Decision threshold"),
        y=alt.Y("F2:Q", title="Test F2", scale=alt.Scale(domain=[round(low, 2), 1], clamp=True)),
        color=color, tooltip=["Target", "Threshold", alt.Tooltip("F2:Q", format=".3f")])
    rules = alt.Chart(chosen).mark_rule(strokeDash=[4, 3]).encode(x="Threshold:Q", color=color)
    st.markdown("**F2 at every threshold** (dashed: the threshold picked on validation engines)")
    st.altair_chart((lines + rules).properties(height=260), width="stretch")

    left, right = st.columns(2)
    with left:
        target = st.radio("Tuning landscape for", list(TARGET_LABELS), format_func=lambda t: TARGET_LABELS[t].split(" (")[0],
                          horizontal=True, key="landscape_target")
        heat = pd.DataFrame([{"Max depth": c["max_depth"], "Learning rate": str(c["learning_rate"]),
                              "Test F2": c["targets"][target]["test_f2"]}
                             for c in configs if c["n_estimators"] == n])
        base_heat = alt.Chart(heat).encode(x=alt.X("Learning rate:O"), y=alt.Y("Max depth:O", sort="descending"))
        st.altair_chart(
            (base_heat.mark_rect().encode(color=alt.Color("Test F2:Q", scale=alt.Scale(scheme="blues"), legend=None),
                                          tooltip=["Max depth", "Learning rate", alt.Tooltip("Test F2:Q", format=".3f")])
             + base_heat.mark_text(fontSize=13).encode(text=alt.Text("Test F2:Q", format=".3f"),
                                                      color=alt.condition("datum['Test F2'] > " + str(heat["Test F2"].median()),
                                                                          alt.value("white"), alt.value("black"))))
            .properties(height=200, title=f"Test F2 with {n} trees"), width="stretch")
        best = max(configs, key=lambda c: sum(c["targets"][t]["val_f2"] for t in TARGET_LABELS))
        st.caption(f"Best on **validation** for this feature set: {best['n_estimators']} trees, depth {best['max_depth']}, "
                   f"learning rate {best['learning_rate']} (test F2 {fmt(best['targets']['EARLY_WARNING']['test_f2'])} / "
                   f"{fmt(best['targets']['CRITICAL_ACTION']['test_f2'])}). Settings are chosen on validation, "
                   "not on the test scores shown here.")
    with right:
        imp = pd.DataFrame(cfg["targets"][target]["importance"], columns=["Feature", "Share of gain"])
        st.altair_chart(alt.Chart(imp).mark_bar(color="#2a78d6", cornerRadiusEnd=3).encode(
            x=alt.X("Share of gain:Q", axis=alt.Axis(format="%")), y=alt.Y("Feature:N", sort="-x", title=None),
            tooltip=["Feature", alt.Tooltip("Share of gain:Q", format=".1%")]).properties(
            height=230, title="What the model relies on (top 8 features)"), width="stretch")


def graph_chart(edges, components):
    comp_of = {s: c for c, members in components.items() for s in members}
    nodes = pd.DataFrame([{"Sensor": s, "x": x, "y": y, "Component": COMPONENT_LABELS[comp_of.get(s, "shaft")]}
                          for s, (x, y) in NODE_POS.items()])
    edge_rows = [{"edge": i, "x": NODE_POS[p][0], "y": NODE_POS[p][1]} for i, e in enumerate(edges) for p in e]
    x_axis, y_axis = alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-0.5, 5.5])), alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-1.8, 1.6]))
    layers = []
    if edge_rows:
        layers.append(alt.Chart(pd.DataFrame(edge_rows)).mark_line(color="#9aa4ae", strokeWidth=1, opacity=0.6)
                      .encode(x=x_axis, y=y_axis, detail="edge:N"))
    layers.append(alt.Chart(nodes).mark_circle(size=520, opacity=1, stroke="white", strokeWidth=2).encode(
        x=x_axis, y=y_axis, color=alt.Color("Component:N", legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Sensor", "Component"]))
    layers.append(alt.Chart(nodes).mark_text(dy=-20, fontSize=12, fontWeight="bold").encode(x=x_axis, y=y_axis, text="Sensor"))
    return alt.layer(*layers).properties(height=340)


def render_gnn(data):
    g = data["gnn"]
    st.markdown("A graph neural network treats each engine snapshot as a graph: the 14 sensors are nodes, each "
                "carrying its last 30 cycles of readings, and attention layers pass information along the links. "
                "Pick which links the model is allowed to use. " + F2_NOTE)
    st.caption("Each graph type was trained with 3 random seeds; results are the mean ± standard deviation on test "
               "engines. Training all of them took about 2.5 hours on 8 CPU cores, too heavy for this free server, so "
               "these are saved results from 10_sensor_graph_gnn.py.")
    kind = st.radio("Graph type", list(GRAPH_INFO), format_func=lambda k: GRAPH_INFO[k][0], horizontal=True)
    info, res = g["graphs"][kind], g["graphs"][kind]["results"]
    st.markdown(GRAPH_INFO[kind][1] + f" **{len(info['edges'])} links.**")
    left, right = st.columns([3, 2])
    with left:
        st.altair_chart(graph_chart(info["edges"], g["components"]), width="stretch")
    with right:
        base = g["xgb_window"]
        for t, label in TARGET_LABELS.items():
            st.metric(f"Test F2 · {label.split(' (')[0]}", f"{res[t]['mean']:.3f} ± {res[t]['std']:.3f}",
                      f"{res[t]['mean'] - base[t]:+.3f} vs XGBoost on the same inputs")
        st.metric("RUL error (RMSE, cycles)", f"{res['test_rmse']['mean']:.1f} ± {res['test_rmse']['std']:.1f}",
                  f"{res['test_rmse']['mean'] - base['test_rmse']:+.1f} vs XGBoost", delta_color="inverse")
        seeds = pd.DataFrame(res["seeds"]).rename(columns={"EARLY_WARNING": "Early warning F2",
                                                           "CRITICAL_ACTION": "Critical action F2", "test_rmse": "RMSE"})
        seeds.index = [f"Seed {i}" for i in range(len(seeds))]
        st.dataframe(seeds, width="stretch")
    if kind == "physics":
        att = pd.DataFrame(g["attention"], columns=["Link", "Near failure", "Healthy"])
        st.markdown("**Where attention moves as engines degrade.** Links whose attention rises most between healthy "
                    "engines and engines within 15 cycles of failure (first attention layer, seed 0):")
        st.dataframe(att, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["Near failure", "Healthy"]})
    st.info("Takeaway: on the same inputs, the GNN matches XGBoost on early-warning F2 and is slightly ahead on "
            "critical action (by 0.004 to 0.007). Removing the graph (None) barely changes F2 but adds about 0.6 cycles "
            "of RUL error. Most of the gain over the original 8-feature model (F2 0.857 to about 0.93) comes from the "
            "condition-normalized sensor windows that both models share.")


def render_modeling():
    try:
        data = load()
    except FileNotFoundError:
        st.error("Model results are missing. Run 17_build_model_explorer.py to create public_app/model_explorer.json.")
        return
    xgb_tab, gnn_tab = st.tabs(["XGBoost", "Graph neural network"])
    with xgb_tab:
        render_xgboost(data)
    with gnn_tab:
        render_gnn(data)
