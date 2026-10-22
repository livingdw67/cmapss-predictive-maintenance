"""Set up the public "Ask the Fleet" demo on Databricks (the Databricks version of 16_setup_public_demo.py).

Run after the cmapss-pipeline Job has built the metric views. Safe to rerun:
  1. A service principal for the Streamlit app, with an OAuth secret written to
     public_app/.streamlit/secrets.toml (git-ignored; the secret is never printed).
  2. Read-only grants on the metric views, plus write access to the question log only.
  3. A Genie space over the metric views with the instructions and sample questions
     ported from the Snowflake semantic view, which the service principal can run.

    python databricks/setup_public_demo.py [--profile free] [--rotate-secret]
"""
import argparse
import json
import pathlib
import time
import uuid

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import iam
from databricks.sdk.service.sql import StatementState

CATALOG, SCHEMA = "workspace", "cmapss"
SP_NAME = "ask-the-fleet-demo"
SPACE_TITLE = "Ask the Fleet"
METRIC_VIEWS = [f"{CATALOG}.{SCHEMA}.fleet_cycles", f"{CATALOG}.{SCHEMA}.fleet_engines"]
QUESTION_LOG = f"{CATALOG}.{SCHEMA}.demo_question_log"
SECRETS_FILE = pathlib.Path(__file__).resolve().parent.parent / "public_app" / ".streamlit" / "secrets.toml"

INSTRUCTIONS = (
    "All engines in this data were run until failure, so lifespan metrics describe completed lives, not a live fleet. "
    "Use fleet_engines for engine counts and lifespan metrics (one row per engine) and fleet_cycles for cycle counts, "
    "health stages and sensor readings (one row per engine per cycle); never count engines from fleet_cycles rows. "
    "Raw sensor averages differ by operating condition: when comparing sensor values or temperature rise across "
    "datasets, group by operating_conditions or warn that Six-condition fleets are not directly comparable to "
    "Single-condition fleets. Report temperatures in degrees Rankine. When the user asks for a difference or ratio, "
    "compute it in the final SELECT."
)
SAMPLE_QUESTIONS = [
    "What is the mean time to failure for each fault mode?",
    "Which dataset has the most variable engine life?",
    "How many engines failed before 150 cycles?",
    "Which engine lasted the longest, and for how many cycles?",
    "How much does HPC outlet temperature rise before failure in FD001?",
    "What share of cycles were flown within 15 cycles of failure?",
]


def run_sql(w, warehouse_id, sql):
    r = w.statement_execution.execute_statement(statement=sql, warehouse_id=warehouse_id, wait_timeout="50s")
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(1)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"{sql.splitlines()[0]}: {r.status.error.message if r.status.error else r.status.state}")


def get_or_create_sp(w):
    existing = list(w.service_principals.list(filter=f'displayName eq "{SP_NAME}"'))
    sp = existing[0] if existing else w.service_principals.create(display_name=SP_NAME, active=True)
    # SQL access is the only workspace entitlement it gets (no notebooks, no workspace UI)
    w.service_principals.patch(sp.id, schemas=[iam.PatchSchema.URN_IETF_PARAMS_SCIM_API_MESSAGES_2_0_PATCH_OP],
                               operations=[iam.Patch(op=iam.PatchOp.ADD, path="entitlements",
                                                     value=[{"value": "databricks-sql-access"}])])
    print(f"Service principal {SP_NAME}: application id {sp.application_id}, SQL access only")
    return sp


def write_secrets(w, sp, warehouse_id, space_id, rotate):
    current = SECRETS_FILE.read_text() if SECRETS_FILE.exists() else ""
    if "client_secret" in current and not rotate:
        secret_line = next(line for line in current.splitlines() if line.startswith("client_secret"))
        print(f"Keeping the existing secret in {SECRETS_FILE}")
    else:
        # Two years, the longest Databricks allows; rerun with --rotate-secret before it expires
        secret = w.service_principal_secrets_proxy.create(service_principal_id=sp.id, lifetime=f"{730 * 86400}s")
        secret_line = f'client_secret = "{secret.secret}"'
        print(f"Created a new OAuth secret (expires {secret.expire_time}); written to {SECRETS_FILE}")
    SECRETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SECRETS_FILE.write_text(
        "[databricks]\n"
        f'host = "{w.config.host}"\n'
        f'client_id = "{sp.application_id}"\n'
        f"{secret_line}\n"
        f'warehouse_id = "{warehouse_id}"\n'
        f'genie_space_id = "{space_id}"\n'
    )


def grant_data_access(w, warehouse_id, app_id):
    run_sql(w, warehouse_id, f"""
        CREATE TABLE IF NOT EXISTS {QUESTION_LOG} (
            asked_at TIMESTAMP, session_id STRING, question STRING, status STRING,
            generated_sql STRING, row_count INT, latency_ms INT
        ) COMMENT 'Questions asked in the public Ask the Fleet demo'
    """)
    for sql in [
        f"GRANT USE CATALOG ON CATALOG {CATALOG} TO `{app_id}`",
        f"GRANT USE SCHEMA ON SCHEMA {CATALOG}.{SCHEMA} TO `{app_id}`",
        *[f"GRANT SELECT ON TABLE {v} TO `{app_id}`" for v in METRIC_VIEWS],
        f"GRANT SELECT, MODIFY ON TABLE {QUESTION_LOG} TO `{app_id}`",
    ]:
        run_sql(w, warehouse_id, sql)
    w.permissions.update("warehouses", warehouse_id, access_control_list=[
        iam.AccessControlRequest(service_principal_name=app_id, permission_level=iam.PermissionLevel.CAN_USE)])
    print("Granted: read on the metric views, read/write on the question log, use of the warehouse")


def serialized_space():
    ids = sorted(uuid.uuid4().hex for _ in range(len(SAMPLE_QUESTIONS) + 1))
    return json.dumps({
        "version": 2,
        "config": {"sample_questions": [{"id": i, "question": [q]} for i, q in zip(ids[1:], SAMPLE_QUESTIONS)]},
        "data_sources": {"tables": [{"identifier": v} for v in METRIC_VIEWS]},
        "instructions": {"text_instructions": [{"id": ids[0], "content": [INSTRUCTIONS]}]},
    })


def get_or_create_space(w, warehouse_id, app_id):
    spaces = w.genie.list_spaces().spaces or []
    match = next((s for s in spaces if s.title == SPACE_TITLE), None)
    description = "Reliability questions about 709 NASA C-MAPSS turbofan engines, all run to failure."
    if match:
        space = w.genie.update_space(match.space_id, serialized_space=serialized_space(), warehouse_id=warehouse_id,
                                     title=SPACE_TITLE, description=description)
    else:
        space = w.genie.create_space(warehouse_id, serialized_space(), title=SPACE_TITLE, description=description)
    w.permissions.update("genie", space.space_id, access_control_list=[
        iam.AccessControlRequest(service_principal_name=app_id, permission_level=iam.PermissionLevel.CAN_RUN)])
    print(f"Genie space '{SPACE_TITLE}': {space.space_id} (service principal can run it)")
    return space.space_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="free")
    parser.add_argument("--rotate-secret", action="store_true")
    args = parser.parse_args()

    w = WorkspaceClient(profile=args.profile)
    warehouse_id = next(wh.id for wh in w.warehouses.list() if wh.enable_serverless_compute)
    sp = get_or_create_sp(w)
    grant_data_access(w, warehouse_id, sp.application_id)
    space_id = get_or_create_space(w, warehouse_id, sp.application_id)
    write_secrets(w, sp, warehouse_id, space_id, args.rotate_secret)
    print(f"\nPaste {SECRETS_FILE} into the Streamlit app's Settings > Secrets.")


if __name__ == "__main__":
    main()
