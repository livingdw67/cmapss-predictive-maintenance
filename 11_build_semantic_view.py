import os
from decimal import Decimal
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

# Alert horizons, kept identical to the tandem classifiers (08 notebook, evaluate_local.py)
EARLY_WARNING_RUL = 50
CRITICAL_ACTION_RUL = 15
# Cycles treated as the healthy baseline at the start of each engine's life
BASELINE_CYCLES = 20

def build_semantic_layer():
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
        # 1. SETUP THE SEMANTIC SCHEMA
        # ---------------------------------------------------------
        # The semantic layer sits on top of CORE and MARTS. It holds no new
        # telemetry; it defines what the business means by "an engine", "a failure",
        # and "mean time to failure" once, so every consumer (BI, Cortex Analyst,
        # agents) gets the same answer.
        print("Creating SEMANTIC schema...")
        cursor.execute("CREATE SCHEMA IF NOT EXISTS PREDICTIVE_MAINTENANCE.SEMANTIC")
        cursor.execute("USE SCHEMA PREDICTIVE_MAINTENANCE.SEMANTIC")

        # ---------------------------------------------------------
        # 2. DATASET REFERENCE TABLE (DIM_DATASET)
        # ---------------------------------------------------------
        # Each CMAPSS sub-dataset is a distinct operating regime. These attributes
        # come from NASA's readme and are what users actually group by
        # ("do engines with fan degradation fail sooner?").
        print("Building DIM_DATASET reference table...")
        cursor.execute("""
            CREATE OR REPLACE TABLE DIM_DATASET (
                dataset VARCHAR PRIMARY KEY,
                operating_conditions VARCHAR,
                fault_modes VARCHAR
            ) AS
            SELECT * FROM VALUES
                ('FD001', 'Single', 'HPC degradation'),
                ('FD002', 'Six', 'HPC degradation'),
                ('FD003', 'Single', 'HPC and fan degradation'),
                ('FD004', 'Six', 'HPC and fan degradation');
        """)

        # ---------------------------------------------------------
        # 3. ENGINE-GRAIN VIEW (ENGINE_LIFECYCLE)
        # ---------------------------------------------------------
        # Lifespan metrics (MTTF, shortest life) must be computed at one row per
        # engine. Averaging the per-cycle MAX_CYCLE column would weight long-lived
        # engines by their cycle count, so lifespan gets its own grain.
        print("Building ENGINE_LIFECYCLE view...")
        cursor.execute("""
            CREATE OR REPLACE VIEW ENGINE_LIFECYCLE AS
            SELECT
                e.engine_id,
                e.dataset,
                MAX(f.cycle) AS failure_cycle
            FROM PREDICTIVE_MAINTENANCE.CORE.DIM_ENGINE e
            JOIN PREDICTIVE_MAINTENANCE.CORE.FCT_TELEMETRY f
                ON f.engine_id = e.engine_id
            GROUP BY e.engine_id, e.dataset;
        """)

        # ---------------------------------------------------------
        # 4. THE SEMANTIC VIEW
        # ---------------------------------------------------------
        # Three logical tables at three grains (dataset > engine > cycle). Each metric
        # is aggregated at its own table's grain before joining, so engine counts
        # never fan out to cycle counts: engine_count by health_stage returns the
        # 709 engines that reached each stage, not 160k telemetry rows.
        print("Building FLEET_RELIABILITY semantic view...")
        cursor.execute(f"""
            CREATE OR REPLACE SEMANTIC VIEW FLEET_RELIABILITY
              TABLES (
                datasets AS PREDICTIVE_MAINTENANCE.SEMANTIC.DIM_DATASET
                  PRIMARY KEY (dataset)
                  WITH SYNONYMS ('fleet', 'sub-fleet', 'operating regime', 'test campaign')
                  COMMENT = 'One row per CMAPSS sub-dataset. Each is a separate fleet of the same engine type flown under a given set of operating conditions and fault modes.',
                engines AS PREDICTIVE_MAINTENANCE.SEMANTIC.ENGINE_LIFECYCLE
                  PRIMARY KEY (engine_id)
                  WITH SYNONYMS ('units', 'turbofans', 'assets')
                  COMMENT = 'One row per engine. Every engine in this data was run until failure.',
                cycles AS PREDICTIVE_MAINTENANCE.MARTS.MART_ENGINE_LIFESPAN
                  PRIMARY KEY (engine_id, cycle)
                  WITH SYNONYMS ('flights', 'telemetry', 'sensor readings')
                  COMMENT = 'One row per engine per operating cycle (one flight), with remaining useful life and key sensor readings.'
              )
              RELATIONSHIPS (
                engine_in_dataset AS engines (dataset) REFERENCES datasets,
                cycle_of_engine AS cycles (engine_id) REFERENCES engines
              )
              FACTS (
                engines.failure_cycle AS failure_cycle
                  WITH SYNONYMS ('lifespan', 'time to failure', 'cycles to failure', 'engine life')
                  COMMENT = 'Number of cycles the engine completed before it failed.',
                cycles.cycle_number AS cycle
                  COMMENT = 'Operating cycle number, counted from 1 at the start of the engine record.',
                cycles.remaining_useful_life AS linear_rul
                  WITH SYNONYMS ('RUL', 'cycles remaining', 'cycles until failure')
                  COMMENT = 'Cycles left before this engine failed, measured from this cycle.',
                cycles.is_early_warning AS IFF(linear_rul <= {EARLY_WARNING_RUL}, 1, 0)
                  COMMENT = 'Flag: 1 when the engine was within {EARLY_WARNING_RUL} cycles of failure.',
                cycles.is_critical_action AS IFF(linear_rul <= {CRITICAL_ACTION_RUL}, 1, 0)
                  COMMENT = 'Flag: 1 when the engine was within {CRITICAL_ACTION_RUL} cycles of failure.',
                cycles.is_baseline AS IFF(cycle <= {BASELINE_CYCLES}, 1, 0)
                  COMMENT = 'Flag: 1 for the first {BASELINE_CYCLES} cycles of an engine record, treated as healthy baseline.',
                cycles.hpc_outlet_temp_reading AS hpc_outlet_temp
                  COMMENT = 'High-pressure compressor outlet temperature (T30), degrees Rankine.',
                cycles.lpt_outlet_temp_reading AS lpt_outlet_temp
                  COMMENT = 'Low-pressure turbine outlet temperature (T50), degrees Rankine.',
                cycles.core_speed_reading AS physical_core_speed
                  COMMENT = 'Physical core speed (Nc), rpm.',
                cycles.static_pressure_reading AS static_pressure_hpc_outlet
                  COMMENT = 'Static pressure at HPC outlet (Ps30), psia.',
                cycles.bypass_ratio_reading AS bypass_ratio
                  COMMENT = 'Engine bypass ratio (BPR), dimensionless.'
              )
              DIMENSIONS (
                datasets.dataset_id AS dataset
                  WITH SYNONYMS ('dataset', 'FD number', 'sub-dataset')
                  COMMENT = 'CMAPSS sub-dataset code. Values: ''FD001'', ''FD002'', ''FD003'', ''FD004''.',
                datasets.operating_conditions AS operating_conditions
                  WITH SYNONYMS ('flight conditions', 'operating regime')
                  COMMENT = 'Number of operating conditions the sub-fleet flew under. Values: ''Single'' (sea level only) or ''Six''. Raw sensor levels shift with operating condition, so compare sensor values only within one condition.',
                datasets.fault_modes AS fault_modes
                  WITH SYNONYMS ('failure mode', 'fault type', 'degradation mode')
                  COMMENT = 'Component degradation that led to failure. Values: ''HPC degradation'' or ''HPC and fan degradation''.',
                engines.engine_id AS engine_id
                  WITH SYNONYMS ('engine', 'unit', 'serial', 'tail')
                  COMMENT = 'Unique engine key in the form FD002_001 (dataset plus unit number; unit numbers restart in each dataset).',
                cycles.health_stage AS CASE
                    WHEN linear_rul <= {CRITICAL_ACTION_RUL} THEN 'Critical action'
                    WHEN linear_rul <= {EARLY_WARNING_RUL} THEN 'Early warning'
                    ELSE 'Healthy'
                  END
                  WITH SYNONYMS ('alert level', 'health status', 'maintenance stage')
                  COMMENT = 'Healthy (more than {EARLY_WARNING_RUL} cycles left), Early warning ({CRITICAL_ACTION_RUL + 1} to {EARLY_WARNING_RUL} left) or Critical action ({CRITICAL_ACTION_RUL} or fewer left). Same horizons as the deployed classifiers.'
              )
              METRICS (
                engines.engine_count AS COUNT(engines.engine_id)
                  WITH SYNONYMS ('number of engines', 'fleet size', 'failures')
                  COMMENT = 'Number of engines. Every engine failed, so this is also the number of failures.',
                engines.mean_time_to_failure AS AVG(engines.failure_cycle)
                  WITH SYNONYMS ('MTTF', 'average lifespan', 'average life', 'mean life')
                  COMMENT = 'Average cycles to failure across engines.',
                engines.median_time_to_failure AS MEDIAN(engines.failure_cycle)
                  WITH SYNONYMS ('median life', 'median lifespan')
                  COMMENT = 'Median cycles to failure across engines.',
                engines.shortest_life AS MIN(engines.failure_cycle)
                  WITH SYNONYMS ('earliest failure', 'minimum life')
                  COMMENT = 'Fewest cycles any engine completed before failing.',
                engines.longest_life AS MAX(engines.failure_cycle)
                  WITH SYNONYMS ('maximum life')
                  COMMENT = 'Most cycles any engine completed before failing.',
                engines.life_stddev AS STDDEV(engines.failure_cycle)
                  WITH SYNONYMS ('lifespan variability', 'spread of life')
                  COMMENT = 'Sample standard deviation of cycles to failure.',
                cycles.total_cycles AS COUNT(cycles.cycle_number)
                  WITH SYNONYMS ('operating cycles', 'flights flown', 'number of readings')
                  COMMENT = 'Total operating cycles flown.',
                cycles.early_warning_cycle_share AS AVG(cycles.is_early_warning)
                  COMMENT = 'Fraction of operating cycles flown within {EARLY_WARNING_RUL} cycles of failure (0 to 1).',
                cycles.critical_action_cycle_share AS AVG(cycles.is_critical_action)
                  COMMENT = 'Fraction of operating cycles flown within {CRITICAL_ACTION_RUL} cycles of failure (0 to 1).',
                cycles.avg_hpc_outlet_temp AS AVG(cycles.hpc_outlet_temp_reading)
                  WITH SYNONYMS ('average T30', 'compressor temperature')
                  COMMENT = 'Average HPC outlet temperature, degrees Rankine.',
                cycles.avg_lpt_outlet_temp AS AVG(cycles.lpt_outlet_temp_reading)
                  WITH SYNONYMS ('average T50', 'turbine temperature', 'exhaust temperature')
                  COMMENT = 'Average LPT outlet temperature, degrees Rankine.',
                cycles.avg_core_speed AS AVG(cycles.core_speed_reading)
                  COMMENT = 'Average physical core speed, rpm.',
                cycles.avg_static_pressure AS AVG(cycles.static_pressure_reading)
                  COMMENT = 'Average HPC outlet static pressure, psia.',
                cycles.avg_bypass_ratio AS AVG(cycles.bypass_ratio_reading)
                  COMMENT = 'Average bypass ratio.',
                cycles.hpc_temp_rise_to_failure AS
                  AVG(IFF(cycles.is_critical_action = 1, cycles.hpc_outlet_temp_reading, NULL))
                  - AVG(IFF(cycles.is_baseline = 1, cycles.hpc_outlet_temp_reading, NULL))
                  WITH SYNONYMS ('compressor temperature drift', 'T30 degradation', 'HPC temperature rise')
                  COMMENT = 'Average HPC outlet temperature in the critical-action window minus the average over the first {BASELINE_CYCLES} cycles, degrees Rankine. Only meaningful within a single operating condition.',
                cycles.lpt_temp_rise_to_failure AS
                  AVG(IFF(cycles.is_critical_action = 1, cycles.lpt_outlet_temp_reading, NULL))
                  - AVG(IFF(cycles.is_baseline = 1, cycles.lpt_outlet_temp_reading, NULL))
                  WITH SYNONYMS ('turbine temperature drift', 'T50 degradation', 'EGT margin loss')
                  COMMENT = 'Average LPT outlet temperature in the critical-action window minus the average over the first {BASELINE_CYCLES} cycles, degrees Rankine. Only meaningful within a single operating condition.'
              )
              COMMENT = 'Reliability and degradation metrics for the NASA CMAPSS turbofan fleet (709 engines, all run to failure).'
              AI_SQL_GENERATION 'All engines in this data were run until failure, so lifespan metrics describe completed lives, not a live fleet. Raw sensor averages differ by operating condition: when comparing sensor values or temperature rise across datasets, group by operating_conditions or warn that Six-condition fleets are not directly comparable to Single-condition fleets. Report temperatures in degrees Rankine.'
        """)

        # ---------------------------------------------------------
        # 5. SMOKE TEST
        # ---------------------------------------------------------
        # Query through the semantic view the same way BI tools and Cortex Analyst will.
        print("\nSemantic view built. Sample query (lifespan by fault mode and conditions):")
        cursor.execute("""
            SELECT * FROM SEMANTIC_VIEW(
                FLEET_RELIABILITY
                DIMENSIONS datasets.operating_conditions, datasets.fault_modes
                METRICS engines.engine_count, engines.mean_time_to_failure,
                        engines.shortest_life, cycles.hpc_temp_rise_to_failure
            )
            ORDER BY operating_conditions, fault_modes
        """)
        cols = [c[0] for c in cursor.description]
        print("  " + " | ".join(cols))
        for row in cursor.fetchall():
            print("  " + " | ".join(f"{float(v):.1f}" if isinstance(v, (float, Decimal)) else str(v) for v in row))

    except Exception as e:
        print(f"Failed to build semantic layer: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    build_semantic_layer()
