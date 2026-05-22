import os
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

def build_staging_layer():
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

        # 1. Create and switch to the STAGING schema
        print("Creating STAGING schema...")
        cursor.execute("CREATE SCHEMA IF NOT EXISTS PREDICTIVE_MAINTENANCE.STAGING")
        cursor.execute("USE SCHEMA PREDICTIVE_MAINTENANCE.STAGING")

        # 2. Build the Staging Table with proper casts and physical column names
        print("Transforming and loading STG_TELEMETRY...")
        
        # We use a CTAS (Create Table As Select) statement here.
        # This reads from the RAW layer, applies the types/names, and materializes it in STAGING.
        staging_sql = """
            CREATE OR REPLACE TABLE STG_TELEMETRY AS 
            SELECT 
                -- Identifiers
                CAST(id AS INTEGER) AS engine_id,
                CAST(cycle AS INTEGER) AS cycle,
                
                -- Operational Settings
                CAST(op_setting_1 AS FLOAT) AS altitude_setting,
                CAST(op_setting_2 AS FLOAT) AS mach_number_setting,
                CAST(op_setting_3 AS FLOAT) AS throttle_resolver_angle,
                
                -- Physical Sensors mapped from CMAPSS documentation
                CAST(sensor_1 AS FLOAT) AS fan_inlet_temp,
                CAST(sensor_2 AS FLOAT) AS lpc_outlet_temp,
                CAST(sensor_3 AS FLOAT) AS hpc_outlet_temp,
                CAST(sensor_4 AS FLOAT) AS lpt_outlet_temp,
                CAST(sensor_5 AS FLOAT) AS fan_inlet_pressure,
                CAST(sensor_6 AS FLOAT) AS bypass_duct_pressure,
                CAST(sensor_7 AS FLOAT) AS hpc_outlet_pressure,
                CAST(sensor_8 AS FLOAT) AS physical_fan_speed,
                CAST(sensor_9 AS FLOAT) AS physical_core_speed,
                CAST(sensor_10 AS FLOAT) AS engine_pressure_ratio,
                CAST(sensor_11 AS FLOAT) AS static_pressure_hpc_outlet,
                CAST(sensor_12 AS FLOAT) AS ratio_fuel_flow_ps30,
                CAST(sensor_13 AS FLOAT) AS corrected_fan_speed,
                CAST(sensor_14 AS FLOAT) AS corrected_core_speed,
                CAST(sensor_15 AS FLOAT) AS bypass_ratio,
                CAST(sensor_16 AS FLOAT) AS burner_fuel_air_ratio,
                CAST(sensor_17 AS FLOAT) AS bleed_enthalpy,
                CAST(sensor_18 AS FLOAT) AS demanded_fan_speed,
                CAST(sensor_19 AS FLOAT) AS demanded_corrected_fan_speed,
                CAST(sensor_20 AS FLOAT) AS hpt_coolant_bleed,
                CAST(sensor_21 AS FLOAT) AS lpt_coolant_bleed
            FROM PREDICTIVE_MAINTENANCE.RAW.RAW_TRAIN;
        """
        
        cursor.execute(staging_sql)
        
        print("Staging layer built successfully!")
        
        # Verify the schema
        cursor.execute("DESCRIBE TABLE STG_TELEMETRY")
        print("\nNew Table Schema:")
        for row in cursor.fetchall():
            # row[0] is column name, row[1] is data type
            print(f"- {row[0]}: {row[1]}")

    except Exception as e:
        print(f"Failed to build staging layer: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    build_staging_layer()