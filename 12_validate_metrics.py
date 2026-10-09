import math
import os
import sys

import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
from scipy import stats
from scipy.special import gamma

load_dotenv()

SEMANTIC_VIEW = "PREDICTIVE_MAINTENANCE.SEMANTIC.FLEET_RELIABILITY"
# Two independent computations of the same metric should agree to float precision
REL_TOL = 1e-6
# A parametric Weibull mean is a different estimator, so it only needs to land close
WEIBULL_TOL = 0.05

ENGINE_METRICS = ["engine_count", "mean_time_to_failure", "median_time_to_failure",
                  "shortest_life", "longest_life", "life_stddev"]
CYCLE_METRICS = ["total_cycles", "early_warning_cycle_share", "critical_action_cycle_share",
                 "avg_hpc_outlet_temp", "avg_lpt_outlet_temp", "avg_core_speed",
                 "avg_static_pressure", "avg_bypass_ratio",
                 "hpc_temp_rise_to_failure", "lpt_temp_rise_to_failure"]

# ---------------------------------------------------------
# REFERENCE SQL
# ---------------------------------------------------------
# Hand-written against STAGING, below the layers the semantic view reads from
# (CORE, MARTS, SEMANTIC). RUL, failure cycle and the alert horizons are
# re-derived here rather than reused, so a bug in any upstream layer or in the
# view's own definitions shows up as a mismatch.
REFERENCE_SQL = """
WITH cyc AS (
    SELECT
        dataset,
        engine_id,
        cycle,
        MAX(cycle) OVER (PARTITION BY engine_id) - cycle AS rul,
        hpc_outlet_temp, lpt_outlet_temp, physical_core_speed,
        static_pressure_hpc_outlet, bypass_ratio
    FROM PREDICTIVE_MAINTENANCE.STAGING.STG_TELEMETRY
),
eng AS (
    SELECT dataset, engine_id, MAX(cycle) AS life
    FROM PREDICTIVE_MAINTENANCE.STAGING.STG_TELEMETRY
    GROUP BY dataset, engine_id
),
eng_agg AS (
    SELECT {group_select}
        COUNT(*)            AS engine_count,
        AVG(life)           AS mean_time_to_failure,
        MEDIAN(life)        AS median_time_to_failure,
        MIN(life)           AS shortest_life,
        MAX(life)           AS longest_life,
        STDDEV_SAMP(life)   AS life_stddev
    FROM eng {group_by}
),
cyc_agg AS (
    SELECT {group_select}
        COUNT(*)                                     AS total_cycles,
        AVG(CASE WHEN rul <= 50 THEN 1 ELSE 0 END)   AS early_warning_cycle_share,
        AVG(CASE WHEN rul <= 15 THEN 1 ELSE 0 END)   AS critical_action_cycle_share,
        AVG(hpc_outlet_temp)                         AS avg_hpc_outlet_temp,
        AVG(lpt_outlet_temp)                         AS avg_lpt_outlet_temp,
        AVG(physical_core_speed)                     AS avg_core_speed,
        AVG(static_pressure_hpc_outlet)              AS avg_static_pressure,
        AVG(bypass_ratio)                            AS avg_bypass_ratio,
        AVG(CASE WHEN rul <= 15 THEN hpc_outlet_temp END)
            - AVG(CASE WHEN cycle <= 20 THEN hpc_outlet_temp END) AS hpc_temp_rise_to_failure,
        AVG(CASE WHEN rul <= 15 THEN lpt_outlet_temp END)
            - AVG(CASE WHEN cycle <= 20 THEN lpt_outlet_temp END) AS lpt_temp_rise_to_failure
    FROM cyc {group_by}
)
SELECT * FROM eng_agg {join_clause} cyc_agg {join_on}
"""

def reference_query(by_dataset):
    if by_dataset:
        return REFERENCE_SQL.format(group_select="dataset,", group_by="GROUP BY dataset",
                                    join_clause="JOIN", join_on="USING (dataset) ORDER BY dataset")
    return REFERENCE_SQL.format(group_select="", group_by="", join_clause="CROSS JOIN", join_on="")

def semantic_query(by_dataset):
    metrics = ", ".join([f"engines.{m}" for m in ENGINE_METRICS] + [f"cycles.{m}" for m in CYCLE_METRICS])
    dims = "DIMENSIONS datasets.dataset_id" if by_dataset else ""
    order = "ORDER BY dataset_id" if by_dataset else ""
    return f"SELECT * FROM SEMANTIC_VIEW({SEMANTIC_VIEW} {dims} METRICS {metrics}) {order}"

def fetch(cursor, sql):
    cursor.execute(sql)
    cols = [c[0].lower() for c in cursor.description]
    return pd.DataFrame(cursor.fetchall(), columns=cols)

def compare(label, semantic_df, reference_df):
    """Compare every metric cell; return a list of (scope, metric, semantic, reference, ok)."""
    rows = []
    for i in range(len(reference_df)):
        scope = reference_df["dataset"].iloc[i] if "dataset" in reference_df else "Fleet"
        for m in ENGINE_METRICS + CYCLE_METRICS:
            s, r = float(semantic_df[m].iloc[i]), float(reference_df[m].iloc[i])
            ok = math.isclose(s, r, rel_tol=REL_TOL, abs_tol=1e-9)
            rows.append((scope, m, s, r, ok))
    return rows

def validate_metrics():
    print("Connecting to Snowflake...")
    try:
        conn = snowflake.connector.connect(
            user=os.getenv('SNOWFLAKE_USER'),
            private_key_file=os.getenv('SNOWFLAKE_PRIVATE_KEY_FILE'),
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
            database=os.getenv('SNOWFLAKE_DATABASE'),
            role=os.getenv('SNOWFLAKE_ROLE')
        )
        cursor = conn.cursor()

        # ---------------------------------------------------------
        # 1. RECONCILE EVERY METRIC: SEMANTIC VIEW vs. REFERENCE SQL
        # ---------------------------------------------------------
        # Checked at two scopes: the whole fleet, and each sub-dataset (which
        # exercises the engines -> datasets and cycles -> engines joins).
        results = []
        for by_dataset in (False, True):
            sem = fetch(cursor, semantic_query(by_dataset))
            ref = fetch(cursor, reference_query(by_dataset))
            if by_dataset:
                sem = sem.rename(columns={"dataset_id": "dataset"})
                assert list(sem["dataset"]) == list(ref["dataset"]), "Dataset keys differ"
            results += compare("dataset" if by_dataset else "fleet", sem, ref)

        failures = [r for r in results if not r[4]]
        print(f"\nReconciled {len(results)} metric values "
              f"({len(ENGINE_METRICS) + len(CYCLE_METRICS)} metrics x 5 scopes): "
              f"{len(results) - len(failures)} match, {len(failures)} differ")
        for scope, m, s, r, _ in failures:
            print(f"  MISMATCH {scope:<6} {m:<28} semantic={s:.6f} reference={r:.6f}")

        # ---------------------------------------------------------
        # 2. GRAIN CHECK: ENGINE METRICS MUST NOT FAN OUT TO CYCLES
        # ---------------------------------------------------------
        # Grouping an engine-grain metric by a cycle-grain dimension is the
        # classic double-count. Each engine passes through all three health
        # stages, so the correct answer is 709 per stage; a fan-out would
        # return cycle counts instead.
        stage = fetch(cursor, f"""
            SELECT * FROM SEMANTIC_VIEW({SEMANTIC_VIEW}
                DIMENSIONS cycles.health_stage
                METRICS engines.engine_count, cycles.total_cycles)
        """).set_index("health_stage")
        n_engines = int(results[0][3])
        grain_ok = all(int(v) == n_engines for v in stage["engine_count"])
        # Every engine flies exactly 16 cycles with RUL 0..15 and 35 with RUL 16..50
        stage_cycles_ok = (int(stage.loc["Critical action", "total_cycles"]) == n_engines * 16
                           and int(stage.loc["Early warning", "total_cycles"]) == n_engines * 35)
        print(f"\nGrain check: engine_count by health_stage = "
              f"{ {k: int(v) for k, v in stage['engine_count'].items()} } -> {'OK' if grain_ok else 'FAN-OUT'}")
        print(f"Stage cycle counts: critical {int(stage.loc['Critical action', 'total_cycles'])} "
              f"(expect {n_engines * 16}), early warning {int(stage.loc['Early warning', 'total_cycles'])} "
              f"(expect {n_engines * 35}) -> {'OK' if stage_cycles_ok else 'WRONG'}")

        # ---------------------------------------------------------
        # 3. CROSS-CHECK MTTF AGAINST A WEIBULL FIT
        # ---------------------------------------------------------
        # The 09 notebook models the fleet as two populations by fault mode.
        # The mean of a fitted 2-parameter Weibull, eta * Gamma(1 + 1/beta), is an
        # independent estimate of MTTF; it should land near the empirical mean.
        lives = fetch(cursor, """
            SELECT d.fault_modes, e.failure_cycle
            FROM PREDICTIVE_MAINTENANCE.SEMANTIC.ENGINE_LIFECYCLE e
            JOIN PREDICTIVE_MAINTENANCE.SEMANTIC.DIM_DATASET d USING (dataset)
        """)
        mttf = fetch(cursor, f"""
            SELECT * FROM SEMANTIC_VIEW({SEMANTIC_VIEW}
                DIMENSIONS datasets.fault_modes METRICS engines.mean_time_to_failure)
        """).set_index("fault_modes")["mean_time_to_failure"].astype(float)
        print("\nWeibull cross-check (2-parameter fit per fault mode):")
        weibull_ok = True
        for mode, grp in lives.groupby("fault_modes"):
            beta, _, eta = stats.weibull_min.fit(grp["failure_cycle"].astype(float), floc=0)
            weibull_mean = eta * gamma(1 + 1 / beta)
            diff = weibull_mean / mttf[mode] - 1
            ok = abs(diff) <= WEIBULL_TOL
            weibull_ok &= ok
            print(f"  {mode:<24} beta={beta:.2f} eta={eta:.1f}  Weibull mean={weibull_mean:.1f}  "
                  f"view MTTF={mttf[mode]:.1f}  diff={diff:+.1%} -> {'OK' if ok else 'OFF'}")

        all_ok = not failures and grain_ok and stage_cycles_ok and weibull_ok
        print("\nAll checks passed." if all_ok else "\nSome checks FAILED.")
        return all_ok

    except Exception as e:
        print(f"Failed to validate metrics: {e}")
        return False
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    sys.exit(0 if validate_metrics() else 1)
