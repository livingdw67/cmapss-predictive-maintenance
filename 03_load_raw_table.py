import os
import snowflake.connector
from dotenv import load_dotenv

# Load credentials from the .env file
load_dotenv()

def load_raw_data():
    print("Connecting to Snowflake...")
    try:
        conn = snowflake.connector.connect(
            user=os.getenv('SNOWFLAKE_USER'),
            password=os.getenv('SNOWFLAKE_PASSWORD'),
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
            database=os.getenv('SNOWFLAKE_DATABASE'),
            schema=os.getenv('SNOWFLAKE_SCHEMA'),
            role=os.getenv('SNOWFLAKE_ROLE')
        )
        cursor = conn.cursor()

        # Enforce the correct database and schema context
        cursor.execute("USE DATABASE PREDICTIVE_MAINTENANCE")
        cursor.execute("USE SCHEMA RAW")

        print("Configuring file format...")
        # Ensure our file format is set to read space-delimited files properly
        cursor.execute("""
            CREATE OR REPLACE FILE FORMAT cmapss_csv_format
                TYPE = 'CSV'
                FIELD_DELIMITER = ' '
                SKIP_HEADER = 0
                ERROR_ON_COLUMN_COUNT_MISMATCH = FALSE
        """)

        print("Rebuilding RAW_TRAIN table to hold 26 columns...")
        # We keep everything as STRINGs in the RAW layer. 
        # Casting to INT/FLOAT happens in the STAGING layer.
        cursor.execute("""
            CREATE OR REPLACE TABLE RAW_TRAIN (
                id STRING, cycle STRING, op_setting_1 STRING, op_setting_2 STRING, op_setting_3 STRING,
                sensor_1 STRING, sensor_2 STRING, sensor_3 STRING, sensor_4 STRING, sensor_5 STRING,
                sensor_6 STRING, sensor_7 STRING, sensor_8 STRING, sensor_9 STRING, sensor_10 STRING,
                sensor_11 STRING, sensor_12 STRING, sensor_13 STRING, sensor_14 STRING, sensor_15 STRING,
                sensor_16 STRING, sensor_17 STRING, sensor_18 STRING, sensor_19 STRING, sensor_20 STRING,
                sensor_21 STRING
            )
        """)

        print("Executing COPY INTO command...")
        # The raw text files often have trailing spaces that Snowflake misinterprets as a 27th column.
        # Selecting $1 through $26 explicitly prevents column mismatch errors.
        cursor.execute("""
            COPY INTO RAW_TRAIN
            FROM (
                SELECT $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                       $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
                       $21, $22, $23, $24, $25, $26
                FROM @cmapss_stage
            )
            PATTERN='.*train_FD.*\\.txt\\.gz'
            FILE_FORMAT = (FORMAT_NAME = 'cmapss_csv_format')
        """)

        print("Data loaded successfully!")
        
        # Verify the load by checking the row count
        cursor.execute("SELECT COUNT(*) FROM RAW_TRAIN")
        count = cursor.fetchone()[0]
        print(f"Total rows loaded into RAW_TRAIN: {count}")

    except Exception as e:
        print(f"Failed to load data: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    load_raw_data()