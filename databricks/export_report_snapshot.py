"""Snapshot the Fleet report for the public app, so the report never spends Free Edition compute.

Runs every report query (streamlit_app/report.py, Databricks backend) for every filter combination
against the metric views and writes the results to public_app/report_snapshot.json, keyed by the
exact SQL text. The app's report reads that file through the same render code, so the numbers are
the metric views' own answers. The NASA data is fixed, so rerun only when the metric views change.

    python databricks/export_report_snapshot.py [--profile free]
"""
import argparse
import itertools
import json
import pathlib
import sys
import time
from datetime import datetime, timezone

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "streamlit_app"))
from report import DATABRICKS_QUERIES, query_text, where_clause  # noqa: E402

OUT = ROOT / "public_app" / "report_snapshot.json"
NUMERIC_TYPES = {"BYTE", "SHORT", "INT", "LONG", "FLOAT", "DOUBLE", "DECIMAL"}


def run(w, warehouse_id, sql):
    r = w.statement_execution.execute_statement(statement=sql, warehouse_id=warehouse_id, wait_timeout="50s")
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(1)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(r.status.error.message if r.status.error else r.status.state)
    columns = r.manifest.schema.columns
    numeric = [c.type_name and c.type_name.value in NUMERIC_TYPES for c in columns]
    rows = [[float(v) if is_num and v is not None else v for v, is_num in zip(row, numeric)]
            for row in (r.result.data_array or [])]
    return {"columns": [c.name for c in columns], "rows": rows}


def subsets(values):
    return [list(s) for n in range(len(values) + 1) for s in itertools.combinations(values, n)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="free")
    args = parser.parse_args()
    w = WorkspaceClient(profile=args.profile)
    warehouse_id = next(wh.id for wh in w.warehouses.list() if wh.enable_serverless_compute)

    options = run(w, warehouse_id, query_text("databricks", "filters"))
    conditions = sorted({r[1] for r in options["rows"]})
    faults = sorted({r[2] for r in options["rows"]})
    results = {query_text("databricks", "filters"): options}
    wheres = {where_clause(c, f, prefix="") for c, f in itertools.product(subsets(conditions), subsets(faults))}
    for where in sorted(wheres):
        for name in DATABRICKS_QUERIES:
            if name != "filters":
                results[query_text("databricks", name, where)] = run(w, warehouse_id, query_text("databricks", name, where))
    OUT.write_text(json.dumps({"exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "queries": results}, indent=1))
    print(f"Wrote {len(results)} query results to {OUT}")


if __name__ == "__main__":
    main()
