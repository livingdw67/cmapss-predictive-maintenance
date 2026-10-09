import ast
import os
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'streamlit_app')
APP_FILE = os.path.join(APP_DIR, 'fleet_report.py')
# Queries and page layout, shared with the public app
REPORT_FILE = os.path.join(APP_DIR, 'report.py')
# Pins the Streamlit version so Snowflake runs the version the app was tested on
ENV_FILE = os.path.join(APP_DIR, 'environment.yml')

def load_app_queries():
    # Pull QUERIES and SEMANTIC_VIEW out of report.py without importing it
    # (the app needs a Snowpark session that only exists inside Snowflake).
    tree = ast.parse(open(REPORT_FILE, encoding='utf-8').read())
    found = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ('QUERIES', 'SEMANTIC_VIEW')):
            found[node.targets[0].id] = ast.literal_eval(node.value)
    return found['QUERIES'], found['SEMANTIC_VIEW']

def deploy_fleet_report():
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
        cursor.execute("USE SCHEMA PREDICTIVE_MAINTENANCE.SEMANTIC")

        # ---------------------------------------------------------
        # 1. PRE-FLIGHT: RUN EVERY QUERY THE APP WILL RUN
        # ---------------------------------------------------------
        # Catch a broken semantic-view query here, not in front of a stakeholder.
        # Each query runs unfiltered and with a filter on both dataset attributes.
        queries, sv = load_app_queries()
        sample_where = ("WHERE datasets.operating_conditions IN ('Single') "
                        "AND datasets.fault_modes IN ('HPC degradation')")
        print("Pre-flight: running the app's queries against the semantic view...")
        for name, sql in queries.items():
            for where in ("", sample_where):
                cursor.execute(sql.format(sv=sv, where=where))
                rows = cursor.fetchall()
                if not rows:
                    raise ValueError(f"Query '{name}' returned no rows (where={where!r})")
            print(f"  {name:<13} OK")

        # ---------------------------------------------------------
        # 2. UPLOAD AND CREATE THE STREAMLIT APP
        # ---------------------------------------------------------
        print("Uploading app to stage...")
        cursor.execute("CREATE STAGE IF NOT EXISTS APP_STAGE")
        for path in (APP_FILE, REPORT_FILE, ENV_FILE):
            path = path.replace('\\', '/')
            cursor.execute(f"PUT file://{path} @APP_STAGE/fleet_report AUTO_COMPRESS=FALSE OVERWRITE=TRUE")

        print("Creating FLEET_REPORT Streamlit app...")
        cursor.execute("""
            CREATE OR REPLACE STREAMLIT FLEET_REPORT
              FROM @PREDICTIVE_MAINTENANCE.SEMANTIC.APP_STAGE/fleet_report
              MAIN_FILE = 'fleet_report.py'
              QUERY_WAREHOUSE = COMPUTE_WH
              TITLE = 'Fleet Reliability'
              COMMENT = 'Fleet reliability report built only on the FLEET_RELIABILITY semantic view'
        """)
        print("Deployed. Open it in Snowsight: Projects > Streamlit > Fleet Reliability.")

    except Exception as e:
        print(f"Failed to deploy fleet report: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    deploy_fleet_report()
