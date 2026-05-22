import os
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

def build_marts_layer():
    print("Connecting to Snowflake...")
    try:
        conn = snowflake.connector.connect(
            user=os.getenv('SNOWFLAKE_USER'),
            password=os.getenv('SNOWFLAKE_PASSWORD'),
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
            database=os.getenv('SNOWFLAKE_DATABASE'),
            role=os.getenv('SNOWFLAKE_ROLE')
        )
        cursor = conn.cursor()

        # ---------------------------------------------------------
        # 1. SETUP THE MARTS SCHEMA
        # ---------------------------------------------------------
        # The MARTS schema is where we apply specific business logic.
        # Instead of just storing data, we are generating the exact target 
        # variables and KPIs that downstream consumers (like ML models) need.
        print("Creating MARTS schema...")
        cursor.execute("CREATE SCHEMA IF NOT EXISTS PREDICTIVE_MAINTENANCE.MARTS")
        cursor.execute("USE SCHEMA PREDICTIVE_MAINTENANCE.MARTS")

        # ---------------------------------------------------------
        # 2. CALCULATE REMAINING USEFUL LIFE (RUL)
        # ---------------------------------------------------------
        # Since this is a run-to-failure dataset, we know the last recorded 
        # cycle for an engine is the moment it broke down. We use Window Functions 
        # to find that maximum cycle per engine, and subtract the current cycle from it.
        print("Calculating RUL in MART_ENGINE_LIFESPAN...")
        marts_sql = """
            CREATE OR REPLACE TABLE MART_ENGINE_LIFESPAN AS
            SELECT
                engine_id,
                cycle,
                
                -- Find the absolute failure point for this specific engine across all rows
                MAX(cycle) OVER (PARTITION BY engine_id) AS max_cycle,
                
                -- Standard Linear RUL: A simple countdown to failure (Max - Current)
                (MAX(cycle) OVER (PARTITION BY engine_id) - cycle) AS linear_rul,
                
                -- Piecewise RUL (Industry Standard for CMAPSS models)
                -- We cap the RUL at 130. If an engine is perfectly healthy (e.g., 250 cycles left),
                -- we don't want the ML model trying to predict the exact number, because healthy 
                -- sensors look identical regardless of whether they have 250 or 200 cycles left.
                -- Capping it forces the model to only care about the degradation phase.
                IFF(
                    (MAX(cycle) OVER (PARTITION BY engine_id) - cycle) > 130, 
                    130, 
                    (MAX(cycle) OVER (PARTITION BY engine_id) - cycle)
                ) AS piecewise_rul,
                
                -- Carry over the highly correlated physical sensors for the next step
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
                
            FROM PREDICTIVE_MAINTENANCE.CORE.FCT_TELEMETRY
            ORDER BY engine_id, cycle;
        """
        cursor.execute(marts_sql)

        print("MARTS layer built successfully!")

    except Exception as e:
        print(f"Failed to build MARTS layer: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    build_marts_layer()