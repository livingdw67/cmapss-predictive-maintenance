# Fleet reliability report, shared by the Streamlit in Snowflake app
# (fleet_report.py) and the public app (public_app/app.py). Every number comes
# from the FLEET_RELIABILITY semantic view, so neither app can drift from the
# governed definitions. Each app passes in its own way of running SQL.
import altair as alt
import pandas as pd
import streamlit as st

SEMANTIC_VIEW = "PREDICTIVE_MAINTENANCE.SEMANTIC.FLEET_RELIABILITY"

# {where} is filled with an optional WHERE clause on dataset attributes
QUERIES = {
    "filters": """
        SELECT * FROM SEMANTIC_VIEW({sv}
            DIMENSIONS datasets.dataset_id, datasets.operating_conditions, datasets.fault_modes)
        ORDER BY dataset_id
    """,
    "kpis": """
        SELECT * FROM SEMANTIC_VIEW({sv}
            METRICS engines.engine_count, engines.mean_time_to_failure,
                    engines.median_time_to_failure, engines.shortest_life,
                    cycles.critical_action_cycle_share
            {where})
    """,
    "by_dataset": """
        SELECT * FROM SEMANTIC_VIEW({sv}
            DIMENSIONS datasets.dataset_id, datasets.operating_conditions, datasets.fault_modes
            METRICS engines.engine_count, engines.mean_time_to_failure,
                    engines.median_time_to_failure, engines.shortest_life,
                    engines.longest_life, engines.life_stddev,
                    cycles.hpc_temp_rise_to_failure, cycles.lpt_temp_rise_to_failure
            {where})
        ORDER BY dataset_id
    """,
    "engine_lives": """
        SELECT * FROM SEMANTIC_VIEW({sv}
            DIMENSIONS engines.engine_id, datasets.fault_modes
            METRICS engines.mean_time_to_failure
            {where})
    """,
    "by_stage": """
        SELECT * FROM SEMANTIC_VIEW({sv}
            DIMENSIONS cycles.health_stage
            METRICS cycles.total_cycles, cycles.avg_hpc_outlet_temp, cycles.avg_lpt_outlet_temp
            {where})
    """,
}
STAGE_ORDER = ["Healthy", "Early warning", "Critical action"]
FAULT_COLORS = alt.Scale(domain=["HPC degradation", "HPC and fan degradation"], range=["#2a78d6", "#eb6834"])


def sql_list(values):
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


def where_clause(conditions, faults):
    parts = []
    if conditions:
        parts.append(f"datasets.operating_conditions IN ({sql_list(conditions)})")
    if faults:
        parts.append(f"datasets.fault_modes IN ({sql_list(faults)})")
    return ("WHERE " + " AND ".join(parts)) if parts else ""


@st.cache_data(ttl=600)
def run(_fetch, name, where=""):
    # _fetch(sql) -> DataFrame; the leading underscore keeps it out of the cache key
    df = _fetch(QUERIES[name].format(sv=SEMANTIC_VIEW, where=where))
    df.columns = [c.lower() for c in df.columns]
    return df


def render_report(fetch):
    options = run(fetch, "filters")
    c1, c2 = st.columns(2)
    conditions = c1.multiselect("Operating conditions", sorted(options["operating_conditions"].unique()))
    faults = c2.multiselect("Fault modes", sorted(options["fault_modes"].unique()))
    where = where_clause(conditions, faults)

    k = run(fetch, "kpis", where).iloc[0]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Engines", f"{int(k['engine_count']):,}")
    m2.metric("MTTF (cycles)", f"{float(k['mean_time_to_failure']):.1f}")
    m3.metric("Median life", f"{float(k['median_time_to_failure']):.1f}")
    m4.metric("Shortest life", f"{int(k['shortest_life'])}")
    m5.metric("Cycles in critical window", f"{float(k['critical_action_cycle_share']):.1%}")

    left, right = st.columns(2)
    with left:
        st.subheader("Lifespan distribution")
        lives = run(fetch, "engine_lives", where).rename(columns={"mean_time_to_failure": "cycles_to_failure"})
        lives["cycles_to_failure"] = lives["cycles_to_failure"].astype(float)
        st.altair_chart(
            alt.Chart(lives).mark_bar(opacity=0.85).encode(
                x=alt.X("cycles_to_failure:Q", bin=alt.Bin(step=25), title="Cycles to failure"),
                y=alt.Y("count():Q", title="Engines", stack=None),
                color=alt.Color("fault_modes:N", scale=FAULT_COLORS, title="Fault modes"),
                tooltip=[alt.Tooltip("count():Q", title="Engines"), "fault_modes:N"],
            ),
            width="stretch",
        )
    with right:
        st.subheader("Temperatures by health stage")
        stage = run(fetch, "by_stage", where)
        stage["health_stage"] = pd.Categorical(stage["health_stage"], STAGE_ORDER, ordered=True)
        stage = stage.sort_values("health_stage")
        st.dataframe(
            stage.rename(columns={"health_stage": "Health stage", "total_cycles": "Cycles",
                                  "avg_hpc_outlet_temp": "Avg T30 (°R)", "avg_lpt_outlet_temp": "Avg T50 (°R)"}),
            hide_index=True, width="stretch",
            column_config={"Avg T30 (°R)": st.column_config.NumberColumn(format="%.1f"),
                           "Avg T50 (°R)": st.column_config.NumberColumn(format="%.1f")},
        )
        selected_conditions = set(conditions) if conditions else set(options["operating_conditions"])
        if len(selected_conditions) > 1:
            st.warning("Raw sensor levels shift with operating condition. Filter to one operating "
                       "condition to compare temperatures across stages.")

    st.subheader("By sub-fleet")
    by_ds = run(fetch, "by_dataset", where)
    st.dataframe(
        by_ds.rename(columns={
            "dataset_id": "Dataset", "operating_conditions": "Conditions", "fault_modes": "Fault modes",
            "engine_count": "Engines", "mean_time_to_failure": "MTTF", "median_time_to_failure": "Median",
            "shortest_life": "Shortest", "longest_life": "Longest", "life_stddev": "Std dev",
            "hpc_temp_rise_to_failure": "T30 rise (°R)", "lpt_temp_rise_to_failure": "T50 rise (°R)"}),
        hide_index=True, width="stretch",
        column_config={c: st.column_config.NumberColumn(format="%.1f")
                       for c in ["MTTF", "Median", "Std dev", "T30 rise (°R)", "T50 rise (°R)"]},
    )

    with st.expander("Queries behind this page"):
        for name in ["kpis", "by_dataset", "engine_lives", "by_stage"]:
            st.code(QUERIES[name].format(sv=SEMANTIC_VIEW, where=where).strip(), language="sql")
