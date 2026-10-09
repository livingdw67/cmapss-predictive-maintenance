import json
import os
import requests
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

DB, SCHEMA, SERVER = "PREDICTIVE_MAINTENANCE", "SEMANTIC", "FLEET_RELIABILITY_MCP"
READER_ROLE = "FLEET_READER"
TEST_QUESTION = "Which fault mode has the shorter mean time to failure, and by how many cycles?"

def connect(role):
    return snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USER'),
        private_key_file=os.getenv('SNOWFLAKE_PRIVATE_KEY_FILE'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
        database=os.getenv('SNOWFLAKE_DATABASE'),
        role=role
    )

def mcp_call(conn, method, params=None, req_id=1):
    """Send one JSON-RPC request to the Snowflake-managed MCP server."""
    resp = requests.post(
        f"https://{conn.host}/api/v2/databases/{DB}/schemas/{SCHEMA}/mcp-servers/{SERVER}",
        headers={"Authorization": f'Snowflake Token="{conn.rest.token}"',
                 "Content-Type": "application/json", "Accept": "application/json"},
        json={"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}},
        timeout=180,
    )
    resp.raise_for_status()
    body = resp.json()
    if "error" in body:
        raise RuntimeError(f"MCP {method} failed: {body['error']}")
    return body["result"]

def create_mcp_server():
    try:
        # ---------------------------------------------------------
        # 1. LEAST-PRIVILEGE ROLE FOR MCP CLIENTS
        # ---------------------------------------------------------
        # Agents connect as FLEET_READER, not as an admin. The role can read the
        # semantic layer and the tables beneath it, and nothing else, so the
        # SQL tool below cannot modify or drop anything.
        print("Connecting to Snowflake as ACCOUNTADMIN...")
        admin = connect(os.getenv('SNOWFLAKE_ROLE'))
        cursor = admin.cursor()
        print(f"Creating read-only role {READER_ROLE}...")
        for sql in [
            f"CREATE ROLE IF NOT EXISTS {READER_ROLE}",
            f"GRANT USAGE ON WAREHOUSE {os.getenv('SNOWFLAKE_WAREHOUSE')} TO ROLE {READER_ROLE}",
            f"GRANT USAGE ON DATABASE {DB} TO ROLE {READER_ROLE}",
            f"GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE {READER_ROLE}",
        ]:
            cursor.execute(sql)
        # The semantic view resolves to tables in these schemas at query time
        for schema in ("CORE", "MARTS", "SEMANTIC"):
            cursor.execute(f"GRANT USAGE ON SCHEMA {DB}.{schema} TO ROLE {READER_ROLE}")
            cursor.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA {DB}.{schema} TO ROLE {READER_ROLE}")
            cursor.execute(f"GRANT SELECT ON ALL VIEWS IN SCHEMA {DB}.{schema} TO ROLE {READER_ROLE}")
        cursor.execute(f"GRANT SELECT ON SEMANTIC VIEW {DB}.{SCHEMA}.FLEET_RELIABILITY TO ROLE {READER_ROLE}")
        cursor.execute(f"GRANT ROLE {READER_ROLE} TO USER {os.getenv('SNOWFLAKE_USER')}")

        # ---------------------------------------------------------
        # 2. THE MCP SERVER
        # ---------------------------------------------------------
        # Two tools: Cortex Analyst turns a question into SQL against the
        # semantic view; the SQL tool runs it under the caller's role. The tool
        # descriptions are what the client's model reads to choose a tool.
        print(f"Creating MCP server {SERVER}...")
        cursor.execute(f"""
            CREATE OR REPLACE MCP SERVER {DB}.{SCHEMA}.{SERVER}
              FROM SPECIFICATION $$
                tools:
                  - name: "fleet_reliability_analyst"
                    type: "CORTEX_ANALYST_MESSAGE"
                    identifier: "{DB}.{SCHEMA}.FLEET_RELIABILITY"
                    title: "Fleet reliability metrics"
                    description: "Answers questions about the NASA CMAPSS turbofan fleet (709 engines, all run to failure) using governed metric definitions: engine counts, mean/median time to failure, shortest and longest life, cycles flown, time in early-warning and critical windows, sensor averages and temperature rise toward failure, by dataset, operating conditions, fault mode, engine or health stage. Returns SQL to run with run_sql."
                  - name: "run_sql"
                    type: "SYSTEM_EXECUTE_SQL"
                    title: "Run read-only SQL"
                    description: "Runs a SQL query with the caller's role and returns the rows. Use it to execute SQL produced by fleet_reliability_analyst."
              $$
        """)
        cursor.execute(f"GRANT USAGE ON MCP SERVER {DB}.{SCHEMA}.{SERVER} TO ROLE {READER_ROLE}")
        admin.close()

        # ---------------------------------------------------------
        # 3. END-TO-END TEST AS THE READ-ONLY ROLE
        # ---------------------------------------------------------
        # The same two calls an agent makes: ask the analyst tool, run its SQL.
        print(f"\nConnecting as {READER_ROLE} to test the server like an MCP client...")
        reader = connect(READER_ROLE)
        tools = mcp_call(reader, "tools/list")["tools"]
        print("tools/list ->", [t["name"] for t in tools])

        print(f"\nQuestion: {TEST_QUESTION}")
        answer = mcp_call(reader, "tools/call", {"name": "fleet_reliability_analyst",
                                                  "arguments": {"message": TEST_QUESTION}}, req_id=2)
        # The tool returns its analyst message as a JSON list of {"text"} and {"statement"} items
        items = json.loads(answer["content"][0]["text"])
        sql = next((i["statement"] for i in items if "statement" in i), None)
        if not sql:
            raise RuntimeError(f"Analyst returned no SQL: {json.dumps(items)[:500]}")
        print("Analyst SQL:\n  " + sql.strip().replace("\n", "\n  "))

        rows = mcp_call(reader, "tools/call", {"name": "run_sql", "arguments": {"sql": sql}}, req_id=3)
        print("\nrun_sql result:")
        for item in rows["content"]:
            print("  " + item.get("text", json.dumps(item))[:1500])

        # The read-only role must not be able to change anything through run_sql.
        # Tool failures come back as a result with isError set, not as a JSON-RPC error.
        probe = mcp_call(reader, "tools/call", {"name": "run_sql", "arguments": {
            "sql": f"CREATE TABLE {DB}.SEMANTIC.MCP_WRITE_PROBE (x INT)"}}, req_id=4)
        blocked = probe.get("isError", False)
        print(f"\nWrite attempt through run_sql blocked: {blocked}")
        if blocked:
            print("  " + probe["content"][0]["text"].splitlines()[1])
        reader.close()

    except Exception as e:
        print(f"Failed to create MCP server: {e}")

if __name__ == "__main__":
    create_mcp_server()
