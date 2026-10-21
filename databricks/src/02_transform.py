# Databricks notebook source
# MAGIC %md
# MAGIC # 02 Transform: staging, core, marts and feature store
# MAGIC
# MAGIC Same layered model as the Snowflake build (`04`–`07`), as Delta tables in one Unity Catalog schema:
# MAGIC
# MAGIC | Layer | Table |
# MAGIC |---|---|
# MAGIC | Staging | `stg_telemetry` (typed, physical sensor names, dataset-aware engine IDs) |
# MAGIC | Core | `dim_engine`, `fct_telemetry` |
# MAGIC | Marts | `mart_engine_lifespan` (linear and piecewise RUL capped at 130) |
# MAGIC | Feature store | `fs_turbofan_predictive_features` (5-cycle rolling means and standard deviations) |

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "cmapss")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
spark.sql(f"USE {catalog}.{schema}")

# COMMAND ----------

# Staging. Engine numbers restart at 1 in each FD00x file, so the dataset is part of the key.
sensor_names = [
    "fan_inlet_temp", "lpc_outlet_temp", "hpc_outlet_temp", "lpt_outlet_temp", "fan_inlet_pressure",
    "bypass_duct_pressure", "hpc_outlet_pressure", "physical_fan_speed", "physical_core_speed",
    "engine_pressure_ratio", "static_pressure_hpc_outlet", "ratio_fuel_flow_ps30", "corrected_fan_speed",
    "corrected_core_speed", "bypass_ratio", "burner_fuel_air_ratio", "bleed_enthalpy", "demanded_fan_speed",
    "demanded_corrected_fan_speed", "hpt_coolant_bleed", "lpt_coolant_bleed",
]
sensor_casts = ",\n    ".join(f"CAST(sensor_{i} AS DOUBLE) AS {name}" for i, name in enumerate(sensor_names, start=1))

def build_staging(raw_table, stg_table):
    spark.sql(f"""
    CREATE OR REPLACE TABLE {stg_table} AS
    SELECT
        regexp_extract(source_file, 'FD00[1-4]', 0) AS dataset,
        regexp_extract(source_file, 'FD00[1-4]', 0) || '_' || lpad(id, 3, '0') AS engine_id,
        CAST(cycle AS INT) AS cycle,
        CAST(op_setting_1 AS DOUBLE) AS altitude_setting,
        CAST(op_setting_2 AS DOUBLE) AS mach_number_setting,
        CAST(op_setting_3 AS DOUBLE) AS throttle_resolver_angle,
        {sensor_casts}
    FROM {raw_table}
    """)

build_staging("raw_train", "stg_telemetry")
build_staging("raw_test", "stg_test_telemetry")

# COMMAND ----------

# Core: engine dimension and telemetry fact (engine_id + cycle is the key)
spark.sql("CREATE OR REPLACE TABLE dim_engine AS SELECT DISTINCT engine_id, dataset FROM stg_telemetry")
spark.sql("CREATE OR REPLACE TABLE fct_telemetry AS SELECT * EXCEPT (dataset) FROM stg_telemetry")

# COMMAND ----------

# Marts: run-to-failure data, so the last recorded cycle is the failure point.
# Piecewise RUL caps at 130 because healthy engines look the same at 250 or 200 cycles left.
spark.sql("""
CREATE OR REPLACE TABLE mart_engine_lifespan AS
SELECT
    engine_id,
    cycle,
    MAX(cycle) OVER (PARTITION BY engine_id) AS max_cycle,
    MAX(cycle) OVER (PARTITION BY engine_id) - cycle AS linear_rul,
    LEAST(MAX(cycle) OVER (PARTITION BY engine_id) - cycle, 130) AS piecewise_rul,
    hpc_outlet_temp,
    lpt_outlet_temp,
    bypass_duct_pressure,
    physical_core_speed,
    static_pressure_hpc_outlet,
    ratio_fuel_flow_ps30,
    bypass_ratio,
    bleed_enthalpy,
    hpt_coolant_bleed,
    lpt_coolant_bleed
FROM fct_telemetry
""")

# COMMAND ----------

# Feature store: 5-cycle moving averages (trend) and standard deviations (volatility rises as parts degrade).
# The same features are built for the test engines, which have no target (their true RUL is in test_rul_truth).
rolling_features = """
    hpc_outlet_temp,
    physical_core_speed,
    bypass_ratio,
    AVG(hpc_outlet_temp)        OVER w AS hpc_temp_ma_5,
    AVG(physical_core_speed)    OVER w AS core_speed_ma_5,
    AVG(bypass_ratio)           OVER w AS bypass_ratio_ma_5,
    STDDEV(hpc_outlet_temp)     OVER w AS hpc_temp_std_5,
    STDDEV(physical_core_speed) OVER w AS core_speed_std_5
"""
window = "WINDOW w AS (PARTITION BY engine_id ORDER BY cycle ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)"

spark.sql(f"""
CREATE OR REPLACE TABLE fs_turbofan_predictive_features AS
SELECT engine_id, CAST(cycle AS BIGINT) AS cycle, piecewise_rul AS target_rul, {rolling_features}
FROM mart_engine_lifespan
{window}
""")
spark.sql(f"""
CREATE OR REPLACE TABLE fs_test_features AS
SELECT engine_id, cycle, {rolling_features}
FROM stg_test_telemetry
{window}
""")
spark.sql("ALTER TABLE fs_turbofan_predictive_features ALTER COLUMN engine_id SET NOT NULL")
spark.sql("ALTER TABLE fs_turbofan_predictive_features ALTER COLUMN cycle SET NOT NULL")
# A primary key with a TIMESERIES column makes this a Unity Catalog feature table
spark.sql("""
ALTER TABLE fs_turbofan_predictive_features
ADD CONSTRAINT fs_turbofan_pk PRIMARY KEY (engine_id, cycle TIMESERIES)
""")

# COMMAND ----------

for t in ["stg_telemetry", "dim_engine", "fct_telemetry", "mart_engine_lifespan",
          "fs_turbofan_predictive_features", "stg_test_telemetry", "fs_test_features"]:
    print(f"{t:<34} {spark.table(t).count():>8,} rows")
