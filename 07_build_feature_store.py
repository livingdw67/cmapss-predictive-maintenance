import os
import snowflake.connector
from dotenv import load_dotenv

# Load the environment variables to keep the connection secure
load_dotenv()

def build_feature_store():
    print("Connecting to Snowflake...")
    try:
        # Establish the connection
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
        # 1. SETUP THE FEATURE STORE SCHEMA
        # ---------------------------------------------------------
        # This is the final layer of the architecture. The Feature Store 
        # acts as the dedicated serving layer for the machine learning models.
        # Data here is highly aggregated, mathematically transformed, and 
        # completely decoupled from the messy raw ingestion layers.
        print("Creating FEATURE_STORE schema...")
        cursor.execute("CREATE SCHEMA IF NOT EXISTS PREDICTIVE_MAINTENANCE.FEATURE_STORE")
        cursor.execute("USE SCHEMA PREDICTIVE_MAINTENANCE.FEATURE_STORE")

        # ---------------------------------------------------------
        # 2. CALCULATE ROLLING WINDOW FEATURES
        # ---------------------------------------------------------
        # We use Snowflake's Window Functions (ROWS BETWEEN) to calculate
        # rolling 5-cycle metrics. We focus on highly correlated sensors:
        # High-Pressure Compressor (HPC) Temp, Core Speed, and Bypass Ratio.
        print("Building FS_TURBOFAN_PREDICTIVE_FEATURES...")
        feature_store_sql = """
            CREATE OR REPLACE TABLE FS_TURBOFAN_PREDICTIVE_FEATURES AS
            SELECT
                engine_id,
                cycle,
                
                -- This is our target variable for the model to predict
                piecewise_rul AS target_rul,
                
                -- Include raw sensors for baseline reference
                hpc_outlet_temp,
                physical_core_speed,
                bypass_ratio,
                
                -- 5-Cycle Moving Averages
                -- Captures the underlying trend, smoothing out random sensor noise.
                AVG(hpc_outlet_temp) OVER (
                    PARTITION BY engine_id 
                    ORDER BY cycle 
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                ) AS hpc_temp_ma_5,
                
                AVG(physical_core_speed) OVER (
                    PARTITION BY engine_id 
                    ORDER BY cycle 
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                ) AS core_speed_ma_5,
                
                AVG(bypass_ratio) OVER (
                    PARTITION BY engine_id 
                    ORDER BY cycle 
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                ) AS bypass_ratio_ma_5,
                
                -- 5-Cycle Rolling Standard Deviations
                -- Captures volatility. As a mechanical component degrades, 
                -- vibration and temperature fluctuations increase rapidly even 
                -- if the overall average remains relatively flat.
                STDDEV(hpc_outlet_temp) OVER (
                    PARTITION BY engine_id 
                    ORDER BY cycle 
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                ) AS hpc_temp_std_5,
                
                STDDEV(physical_core_speed) OVER (
                    PARTITION BY engine_id 
                    ORDER BY cycle 
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                ) AS core_speed_std_5
                
            FROM PREDICTIVE_MAINTENANCE.MARTS.MART_ENGINE_LIFESPAN
            ORDER BY engine_id, cycle;
        """
        cursor.execute(feature_store_sql)

        print("FEATURE_STORE layer built successfully!")
        
        # ---------------------------------------------------------
        # 3. VERIFY THE FINAL DATASET
        # ---------------------------------------------------------
        cursor.execute("SELECT COUNT(*) FROM FS_TURBOFAN_PREDICTIVE_FEATURES")
        row_count = cursor.fetchone()[0]
        print(f"Total feature records ready for MLflow training: {row_count}")

    except Exception as e:
        print(f"Failed to build Feature Store layer: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    build_feature_store()