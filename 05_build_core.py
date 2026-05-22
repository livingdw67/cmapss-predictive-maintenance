import os
import snowflake.connector
from dotenv import load_dotenv

# Load the environment variables from the .env file to keep credentials secure
load_dotenv()

def build_core_layer():
    print("Connecting to Snowflake...")
    try:
        # Establish the connection using the secure variables
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
        # 1. SETUP THE CORE SCHEMA
        # ---------------------------------------------------------
        # The CORE schema acts as the "Integration" layer in a 5-schema design.
        # This is where we move from flat, disparate logs into a unified, 
        # heavily structured relational model (like a Star Schema).
        print("Creating CORE schema...")
        cursor.execute("CREATE SCHEMA IF NOT EXISTS PREDICTIVE_MAINTENANCE.CORE")
        cursor.execute("USE SCHEMA PREDICTIVE_MAINTENANCE.CORE")

        # ---------------------------------------------------------
        # 2. BUILD THE DIMENSION TABLE (DIM_ENGINE)
        # ---------------------------------------------------------
        # Dimension tables hold the "entities" or "nouns" of the business.
        # In CMAPSS, the primary entity is the Engine itself. By pulling the DISTINCT
        # engine_ids, we create a master dimension lookup table.
        # In a real-world scenario, this table would also hold static attributes like:
        # Engine Model, Date of Manufacture, Tail Number, or Fleet Location.
        print("Building DIM_ENGINE table...")
        dim_engine_sql = """
            CREATE OR REPLACE TABLE DIM_ENGINE AS
            SELECT DISTINCT 
                engine_id
            FROM PREDICTIVE_MAINTENANCE.STAGING.STG_TELEMETRY
            ORDER BY engine_id;
        """
        cursor.execute(dim_engine_sql)

        # ---------------------------------------------------------
        # 3. BUILD THE FACT TABLE (FCT_TELEMETRY)
        # ---------------------------------------------------------
        # Fact tables hold the "verbs" or "events" of the business.
        # Here, the event is a single sensor ping at a specific cycle in time.
        # The combination of (engine_id + cycle) serves as the primary key.
        print("Building FCT_TELEMETRY table...")
        fct_telemetry_sql = """
            CREATE OR REPLACE TABLE FCT_TELEMETRY AS
            SELECT 
                -- Foreign key linking back to DIM_ENGINE
                engine_id,
                
                -- The sequence/time dimension
                cycle,
                
                -- The operational settings
                altitude_setting,
                mach_number_setting,
                throttle_resolver_angle,
                
                -- The physical sensor readings
                fan_inlet_temp,
                lpc_outlet_temp,
                hpc_outlet_temp,
                lpt_outlet_temp,
                fan_inlet_pressure,
                bypass_duct_pressure,
                hpc_outlet_pressure,
                physical_fan_speed,
                physical_core_speed,
                engine_pressure_ratio,
                static_pressure_hpc_outlet,
                ratio_fuel_flow_ps30,
                corrected_fan_speed,
                corrected_core_speed,
                bypass_ratio,
                burner_fuel_air_ratio,
                bleed_enthalpy,
                demanded_fan_speed,
                demanded_corrected_fan_speed,
                hpt_coolant_bleed,
                lpt_coolant_bleed
            FROM PREDICTIVE_MAINTENANCE.STAGING.STG_TELEMETRY
            ORDER BY engine_id, cycle;
        """
        cursor.execute(fct_telemetry_sql)

        print("CORE layer built successfully!")

    except Exception as e:
        print(f"Failed to build core layer: {e}")
    finally:
        # Always close the cursor and connection to prevent memory leaks in the Snowflake warehouse
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    build_core_layer()