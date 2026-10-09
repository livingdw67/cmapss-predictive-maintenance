import os
import snowflake.connector
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from dotenv import load_dotenv

load_dotenv()

DB = "PREDICTIVE_MAINTENANCE"
DEMO_USER = "FLEET_DEMO_APP"
DEMO_ROLE = "FLEET_DEMO"
DEMO_WH = "FLEET_DEMO_WH"
MONITOR = "FLEET_DEMO_MONITOR"
# Warehouse credits per month before Snowflake suspends the demo warehouse.
# Cortex Analyst is billed separately (about 0.067 credits per question) and is
# capped by the app's daily question limit, which reads DEMO.QUESTION_LOG.
MONTHLY_WAREHOUSE_CREDITS = 10

# The app's private key and Streamlit secrets live outside the repo
KEY_DIR = os.path.join(os.path.expanduser("~"), ".snowflake")
KEY_FILE = os.path.join(KEY_DIR, "fleet_demo_app.p8")
SECRETS_FILE = os.path.join(KEY_DIR, "fleet_demo_streamlit_secrets.toml")

def load_or_create_key():
    # Reuse the key on re-runs so a deployed app keeps working
    if os.path.exists(KEY_FILE):
        key = serialization.load_pem_private_key(open(KEY_FILE, 'rb').read(), password=None)
    else:
        os.makedirs(KEY_DIR, exist_ok=True)
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        with open(KEY_FILE, 'wb') as f:
            f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                      serialization.NoEncryption()))
    pem_public = key.public_key().public_bytes(serialization.Encoding.PEM,
                                               serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    public_body = "".join(l for l in pem_public.splitlines() if "PUBLIC KEY" not in l)
    return open(KEY_FILE, encoding='utf-8').read(), public_body

def setup_public_demo():
    print("Connecting to Snowflake as ACCOUNTADMIN...")
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
        # 1. A SEPARATE, CAPPED WAREHOUSE
        # ---------------------------------------------------------
        # Public traffic never touches COMPUTE_WH. The demo warehouse is the
        # smallest size, sleeps after 60 idle seconds, kills any query over
        # 30 seconds, and is suspended by a resource monitor at the monthly quota.
        print(f"Creating warehouse {DEMO_WH} and resource monitor {MONITOR}...")
        cursor.execute(f"""
            CREATE WAREHOUSE IF NOT EXISTS {DEMO_WH}
              WAREHOUSE_SIZE = XSMALL AUTO_SUSPEND = 60 AUTO_RESUME = TRUE INITIALLY_SUSPENDED = TRUE
              COMMENT = 'Public Ask the Fleet demo'
        """)
        cursor.execute(f"ALTER WAREHOUSE {DEMO_WH} SET STATEMENT_TIMEOUT_IN_SECONDS = 30 "
                       f"STATEMENT_QUEUED_TIMEOUT_IN_SECONDS = 30 MAX_CLUSTER_COUNT = 1")
        cursor.execute(f"""
            CREATE OR REPLACE RESOURCE MONITOR {MONITOR}
              WITH CREDIT_QUOTA = {MONTHLY_WAREHOUSE_CREDITS} FREQUENCY = MONTHLY START_TIMESTAMP = IMMEDIATELY
              TRIGGERS ON 80 PERCENT DO NOTIFY ON 100 PERCENT DO SUSPEND_IMMEDIATE
        """)
        cursor.execute(f"ALTER WAREHOUSE {DEMO_WH} SET RESOURCE_MONITOR = {MONITOR}")

        # ---------------------------------------------------------
        # 2. QUESTION LOG (THE DAILY CAP READS THIS)
        # ---------------------------------------------------------
        print("Creating DEMO.QUESTION_LOG...")
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {DB}.DEMO")
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {DB}.DEMO.QUESTION_LOG (
                asked_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
                session_id VARCHAR,
                question VARCHAR,
                status VARCHAR,
                generated_sql VARCHAR,
                row_count NUMBER,
                latency_ms NUMBER
            )
        """)

        # ---------------------------------------------------------
        # 3. READ-ONLY ROLE
        # ---------------------------------------------------------
        # Reads the semantic layer and the tables beneath it. Its only write
        # is appending to the question log.
        print(f"Creating role {DEMO_ROLE}...")
        grants = [
            f"CREATE ROLE IF NOT EXISTS {DEMO_ROLE}",
            f"GRANT USAGE ON WAREHOUSE {DEMO_WH} TO ROLE {DEMO_ROLE}",
            f"GRANT USAGE ON DATABASE {DB} TO ROLE {DEMO_ROLE}",
            f"GRANT DATABASE ROLE SNOWFLAKE.CORTEX_ANALYST_USER TO ROLE {DEMO_ROLE}",
            f"GRANT USAGE ON SCHEMA {DB}.DEMO TO ROLE {DEMO_ROLE}",
            f"GRANT SELECT, INSERT ON TABLE {DB}.DEMO.QUESTION_LOG TO ROLE {DEMO_ROLE}",
            f"GRANT SELECT ON SEMANTIC VIEW {DB}.SEMANTIC.FLEET_RELIABILITY TO ROLE {DEMO_ROLE}",
        ]
        for schema in ("CORE", "MARTS", "SEMANTIC"):
            grants += [f"GRANT USAGE ON SCHEMA {DB}.{schema} TO ROLE {DEMO_ROLE}",
                       f"GRANT SELECT ON ALL TABLES IN SCHEMA {DB}.{schema} TO ROLE {DEMO_ROLE}",
                       f"GRANT SELECT ON ALL VIEWS IN SCHEMA {DB}.{schema} TO ROLE {DEMO_ROLE}"]
        for sql in grants:
            cursor.execute(sql)

        # ---------------------------------------------------------
        # 4. SERVICE USER WITH KEY-PAIR LOGIN ONLY
        # ---------------------------------------------------------
        # TYPE = SERVICE users cannot log in with a password or to Snowsight.
        # Secondary roles are off, so the session runs with FLEET_DEMO alone.
        print(f"Creating service user {DEMO_USER}...")
        private_pem, public_body = load_or_create_key()
        cursor.execute(f"""
            CREATE USER IF NOT EXISTS {DEMO_USER}
              TYPE = SERVICE DEFAULT_ROLE = {DEMO_ROLE} DEFAULT_WAREHOUSE = {DEMO_WH}
              DEFAULT_SECONDARY_ROLES = ()
              COMMENT = 'Public Ask the Fleet demo on Streamlit Community Cloud'
        """)
        cursor.execute(f"ALTER USER {DEMO_USER} SET RSA_PUBLIC_KEY = '{public_body}'")
        cursor.execute(f"GRANT ROLE {DEMO_ROLE} TO USER {DEMO_USER}")

        # ---------------------------------------------------------
        # 5. SECRETS FOR STREAMLIT COMMUNITY CLOUD
        # ---------------------------------------------------------
        account = os.getenv('SNOWFLAKE_ACCOUNT')
        with open(SECRETS_FILE, 'w', encoding='utf-8') as f:
            f.write("[snowflake]\n"
                    f'account = "{account}"\n'
                    f'user = "{DEMO_USER}"\n'
                    f'role = "{DEMO_ROLE}"\n'
                    f'warehouse = "{DEMO_WH}"\n'
                    f'private_key = """{private_pem.strip()}"""\n')
        print(f"\nDone. Streamlit secrets written to {SECRETS_FILE}")
        print("Paste that file's contents into the app's Secrets box on Streamlit Community Cloud.")

    except Exception as e:
        print(f"Failed to set up public demo: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    setup_public_demo()
