# Databricks notebook source
# MAGIC %md
# MAGIC # 05 Semantic layer: Unity Catalog metric views
# MAGIC
# MAGIC Port of `11_build_semantic_view.py`. Snowflake's `FLEET_RELIABILITY` semantic view becomes two
# MAGIC Unity Catalog metric views, one per grain, so engine metrics never fan out to cycle rows:
# MAGIC
# MAGIC | Metric view | Grain | Examples |
# MAGIC |---|---|---|
# MAGIC | `fleet_engines` | one row per engine | engine count, MTTF, median, shortest and longest life |
# MAGIC | `fleet_cycles` | one row per engine per cycle | health stage, cycle shares, sensor averages, temperature rise to failure |
# MAGIC
# MAGIC Both join `dim_dataset` (operating conditions and fault modes from the NASA readme) and carry the
# MAGIC comments and synonyms that the Genie space uses to map questions to the right definition.
# MAGIC The last cell reconciles metric-view results against plain SQL on the tables, like `12_validate_metrics.py`.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "cmapss")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
fq = f"{catalog}.{schema}"
spark.sql(f"USE {fq}")

# COMMAND ----------

# Dataset reference table (operating regime and fault modes per sub-dataset, from the NASA readme)
spark.sql("""
CREATE OR REPLACE TABLE dim_dataset (
    dataset STRING NOT NULL COMMENT 'CMAPSS sub-dataset code',
    operating_conditions STRING COMMENT 'Single or Six operating conditions',
    fault_modes STRING COMMENT 'Component degradation that led to failure'
) COMMENT 'One row per CMAPSS sub-dataset (from the NASA readme)'
""")
spark.sql("""
INSERT INTO dim_dataset VALUES
    ('FD001', 'Single', 'HPC degradation'),
    ('FD002', 'Six', 'HPC degradation'),
    ('FD003', 'Single', 'HPC and fan degradation'),
    ('FD004', 'Six', 'HPC and fan degradation')
""")

# COMMAND ----------

# Engine grain: lifespan metrics must be one row per engine, or long-lived engines get weighted by cycle count
spark.sql("""
CREATE OR REPLACE VIEW engine_lifecycle
COMMENT 'One row per engine with the cycle it failed at. Every engine was run until failure.'
AS SELECT e.engine_id, e.dataset, MAX(f.cycle) AS failure_cycle
FROM dim_engine e JOIN fct_telemetry f ON f.engine_id = e.engine_id
GROUP BY e.engine_id, e.dataset
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW fleet_engines
WITH METRICS
LANGUAGE YAML
AS $$
version: 1.1
comment: "Engine-grain reliability metrics for the NASA CMAPSS turbofan fleet: one row per engine, all 709 run until failure, so lifespan metrics describe completed lives. Use fleet_cycles for sensor readings and health stages."
source: {fq}.engine_lifecycle
joins:
  - name: datasets
    source: {fq}.dim_dataset
    on: source.dataset = datasets.dataset
dimensions:
  - name: dataset_id
    expr: source.dataset
    comment: "CMAPSS sub-dataset code. Values: FD001, FD002, FD003, FD004."
    synonyms: [dataset, FD number, sub-dataset, sub-fleet, fleet]
  - name: operating_conditions
    expr: datasets.operating_conditions
    comment: "Number of operating conditions the sub-fleet flew under: Single (sea level only) or Six."
    synonyms: [flight conditions, operating regime]
  - name: fault_modes
    expr: datasets.fault_modes
    comment: "Degradation that led to failure: HPC degradation, or HPC and fan degradation."
    synonyms: [failure mode, fault type, degradation mode]
  - name: engine_id
    expr: source.engine_id
    comment: "Unique engine key like FD002_001 (dataset plus unit number; unit numbers restart in each dataset)."
    synonyms: [engine, unit, serial, tail]
  - name: failure_cycle
    expr: source.failure_cycle
    comment: "Cycles the engine completed before it failed. Use as a dimension to filter or bucket engines by life."
    synonyms: [lifespan, time to failure, cycles to failure, engine life]
measures:
  - name: engine_count
    expr: COUNT(1)
    comment: "Number of engines. Every engine failed, so this is also the number of failures."
    synonyms: [number of engines, fleet size, failures]
  - name: mean_time_to_failure
    expr: AVG(source.failure_cycle)
    comment: "Average cycles to failure across engines."
    synonyms: [MTTF, average lifespan, average life, mean life]
  - name: median_time_to_failure
    expr: MEDIAN(source.failure_cycle)
    comment: "Median cycles to failure across engines."
    synonyms: [median life, median lifespan]
  - name: shortest_life
    expr: MIN(source.failure_cycle)
    comment: "Fewest cycles any engine completed before failing."
    synonyms: [earliest failure, minimum life]
  - name: longest_life
    expr: MAX(source.failure_cycle)
    comment: "Most cycles any engine completed before failing."
    synonyms: [maximum life]
  - name: life_stddev
    expr: STDDEV(source.failure_cycle)
    comment: "Sample standard deviation of cycles to failure."
    synonyms: [lifespan variability, spread of life]
$$
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW fleet_cycles
WITH METRICS
LANGUAGE YAML
AS $$
version: 1.1
comment: "Cycle-grain metrics for the NASA CMAPSS turbofan fleet: one row per engine per operating cycle (one flight) with remaining useful life and key sensor readings. Raw sensor levels shift with operating condition, so compare sensor values only within one operating condition. Temperatures are in degrees Rankine."
source: {fq}.mart_engine_lifespan
joins:
  - name: datasets
    source: {fq}.dim_dataset
    on: left(source.engine_id, 5) = datasets.dataset
dimensions:
  - name: dataset_id
    expr: datasets.dataset
    comment: "CMAPSS sub-dataset code. Values: FD001, FD002, FD003, FD004."
    synonyms: [dataset, FD number, sub-dataset, sub-fleet, fleet]
  - name: operating_conditions
    expr: datasets.operating_conditions
    comment: "Number of operating conditions the sub-fleet flew under: Single (sea level only) or Six."
    synonyms: [flight conditions, operating regime]
  - name: fault_modes
    expr: datasets.fault_modes
    comment: "Degradation that led to failure: HPC degradation, or HPC and fan degradation."
    synonyms: [failure mode, fault type, degradation mode]
  - name: engine_id
    expr: source.engine_id
    comment: "Unique engine key like FD002_001."
    synonyms: [engine, unit]
  - name: cycle_number
    expr: source.cycle
    comment: "Operating cycle number, counted from 1 at the start of the engine record."
  - name: remaining_useful_life
    expr: source.linear_rul
    comment: "Cycles left before this engine failed, measured from this cycle."
    synonyms: [RUL, cycles remaining, cycles until failure]
  - name: health_stage
    expr: "CASE WHEN source.linear_rul <= 15 THEN 'Critical action' WHEN source.linear_rul <= 50 THEN 'Early warning' ELSE 'Healthy' END"
    comment: "Healthy (more than 50 cycles left), Early warning (16 to 50 left) or Critical action (15 or fewer left). Same horizons as the deployed classifiers."
    synonyms: [alert level, health status, maintenance stage]
measures:
  - name: total_cycles
    expr: COUNT(1)
    comment: "Total operating cycles flown."
    synonyms: [operating cycles, flights flown, number of readings]
  - name: engines_reaching_stage
    expr: COUNT(DISTINCT source.engine_id)
    comment: "Number of distinct engines with at least one cycle in the group (by health stage: engines that reached that stage)."
    synonyms: [engine count]
  - name: early_warning_cycle_share
    expr: AVG(IF(source.linear_rul <= 50, 1, 0))
    comment: "Fraction of operating cycles flown within 50 cycles of failure (0 to 1)."
  - name: critical_action_cycle_share
    expr: AVG(IF(source.linear_rul <= 15, 1, 0))
    comment: "Fraction of operating cycles flown within 15 cycles of failure (0 to 1)."
  - name: avg_hpc_outlet_temp
    expr: AVG(source.hpc_outlet_temp)
    comment: "Average HPC outlet temperature (T30), degrees Rankine."
    synonyms: [average T30, compressor temperature]
  - name: avg_lpt_outlet_temp
    expr: AVG(source.lpt_outlet_temp)
    comment: "Average LPT outlet temperature (T50), degrees Rankine."
    synonyms: [average T50, turbine temperature, exhaust temperature]
  - name: avg_core_speed
    expr: AVG(source.physical_core_speed)
    comment: "Average physical core speed (Nc), rpm."
  - name: avg_static_pressure
    expr: AVG(source.static_pressure_hpc_outlet)
    comment: "Average HPC outlet static pressure (Ps30), psia."
  - name: avg_bypass_ratio
    expr: AVG(source.bypass_ratio)
    comment: "Average bypass ratio."
  - name: hpc_temp_rise_to_failure
    expr: AVG(IF(source.linear_rul <= 15, source.hpc_outlet_temp, NULL)) - AVG(IF(source.cycle <= 20, source.hpc_outlet_temp, NULL))
    comment: "Average HPC outlet temperature in the last 15 cycles before failure minus the average over the first 20 cycles, degrees Rankine. Only meaningful within one operating condition."
    synonyms: [compressor temperature drift, T30 degradation, HPC temperature rise]
  - name: lpt_temp_rise_to_failure
    expr: AVG(IF(source.linear_rul <= 15, source.lpt_outlet_temp, NULL)) - AVG(IF(source.cycle <= 20, source.lpt_outlet_temp, NULL))
    comment: "Average LPT outlet temperature in the last 15 cycles before failure minus the average over the first 20 cycles, degrees Rankine. Only meaningful within one operating condition."
    synonyms: [turbine temperature drift, T50 degradation, EGT margin loss]
$$
""")

# COMMAND ----------

# Reconcile metric views against plain SQL on the tables; fail the task on any mismatch
import math

checks = {
    "engine_count": ("SELECT MEASURE(engine_count) FROM fleet_engines", "SELECT COUNT(*) FROM dim_engine"),
    "mean_time_to_failure": ("SELECT MEASURE(mean_time_to_failure) FROM fleet_engines",
                             "SELECT AVG(c) FROM (SELECT MAX(cycle) c FROM fct_telemetry GROUP BY engine_id)"),
    "median_time_to_failure": ("SELECT MEASURE(median_time_to_failure) FROM fleet_engines",
                               "SELECT MEDIAN(c) FROM (SELECT MAX(cycle) c FROM fct_telemetry GROUP BY engine_id)"),
    "total_cycles": ("SELECT MEASURE(total_cycles) FROM fleet_cycles", "SELECT COUNT(*) FROM stg_telemetry"),
    "critical_cycle_share": ("SELECT MEASURE(critical_action_cycle_share) FROM fleet_cycles",
                             "SELECT AVG(IF(linear_rul <= 15, 1, 0)) FROM mart_engine_lifespan"),
    "fd001_engines": ("SELECT MEASURE(engine_count) FROM fleet_engines WHERE dataset_id = 'FD001'",
                      "SELECT COUNT(*) FROM dim_engine WHERE dataset = 'FD001'"),
    "fd001_t30_rise": ("SELECT MEASURE(hpc_temp_rise_to_failure) FROM fleet_cycles WHERE dataset_id = 'FD001'",
                       """SELECT AVG(IF(m.linear_rul <= 15, s.hpc_outlet_temp, NULL)) - AVG(IF(s.cycle <= 20, s.hpc_outlet_temp, NULL))
                          FROM stg_telemetry s JOIN mart_engine_lifespan m USING (engine_id, cycle) WHERE s.dataset = 'FD001'"""),
}
failures = []
for name, (metric_sql, direct_sql) in checks.items():
    got, expected = (float(spark.sql(q).first()[0]) for q in (metric_sql, direct_sql))
    ok = math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-9)
    print(f"{'OK  ' if ok else 'FAIL'} {name:<24} metric view {got:,.4f} | direct SQL {expected:,.4f}")
    if not ok:
        failures.append(name)
assert not failures, f"Metric views disagree with direct SQL: {failures}"
