import os
import snowflake.connector
from dotenv import load_dotenv

# Load credentials from the .env file
load_dotenv()

def upload_raw_data_to_snowflake():
    # Define the path to your local data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', 'raw')
    
    print("Connecting to Snowflake...")
    try:
        conn = snowflake.connector.connect(
            user=os.getenv('SNOWFLAKE_USER'),
            private_key_file=os.getenv('SNOWFLAKE_PRIVATE_KEY_FILE'),
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
            database=os.getenv('SNOWFLAKE_DATABASE'),
            schema=os.getenv('SNOWFLAKE_SCHEMA'),
            role=os.getenv('SNOWFLAKE_ROLE')
        )
        cursor = conn.cursor()

        # Create the database, RAW schema, and internal stage on a fresh account.
        # IF NOT EXISTS makes this safe to re-run.
        cursor.execute("CREATE DATABASE IF NOT EXISTS PREDICTIVE_MAINTENANCE")
        cursor.execute("CREATE SCHEMA IF NOT EXISTS PREDICTIVE_MAINTENANCE.RAW")
        cursor.execute("USE SCHEMA PREDICTIVE_MAINTENANCE.RAW")
        cursor.execute("CREATE STAGE IF NOT EXISTS cmapss_stage")

        # We will upload all the train_FD*.txt files
        # The file:// syntax is required by Snowflake's PUT command for local files
        
        # Windows paths need to be formatted correctly for the PUT command
        # We replace backslashes with forward slashes for Snowflake's parser
        windows_path = os.path.join(data_dir, 'train_FD*.txt').replace('\\', '/')
        put_command = f"PUT file://{windows_path} @cmapss_stage AUTO_COMPRESS=TRUE"
        
        print(f"Executing PUT command...")
        cursor.execute(put_command)
        
        print("Upload successful! Here are the files currently in the stage:")
        cursor.execute("LIST @cmapss_stage")
        for row in cursor.fetchall():
            print(f"- {row[0]} (Size: {row[1]} bytes)")
            
    except Exception as e:
        print(f"Failed to upload data: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    upload_raw_data_to_snowflake()