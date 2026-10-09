import json
import math
import os
import sys
import time
from datetime import date

import requests
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

SEMANTIC_VIEW = "PREDICTIVE_MAINTENANCE.SEMANTIC.FLEET_RELIABILITY"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(BASE_DIR, 'cortex_analyst_eval.json')
# Generated SQL may round or order differently; numbers must agree to 0.5%
REL_TOL = 0.005
RUNS = 3

# ---------------------------------------------------------
# GROUND TRUTH
# ---------------------------------------------------------
# Every expected answer is computed from STAGING with hand-written SQL, never
# from the semantic view, so the eval cannot grade the view against itself.
# Dataset attributes are restated from NASA's readme for the same reason.
TRUTH_CTE = """
WITH ds AS (
    SELECT * FROM VALUES
        ('FD001', 'Single', 'HPC degradation'),
        ('FD002', 'Six', 'HPC degradation'),
        ('FD003', 'Single', 'HPC and fan degradation'),
        ('FD004', 'Six', 'HPC and fan degradation') AS v(dataset, conditions, faults)
),
eng AS (
    SELECT t.dataset, t.engine_id, MAX(t.cycle) AS life, ds.conditions, ds.faults
    FROM PREDICTIVE_MAINTENANCE.STAGING.STG_TELEMETRY t JOIN ds USING (dataset)
    GROUP BY t.dataset, t.engine_id, ds.conditions, ds.faults
),
cyc AS (
    SELECT t.*, MAX(t.cycle) OVER (PARTITION BY t.engine_id) - t.cycle AS rul, ds.conditions, ds.faults
    FROM PREDICTIVE_MAINTENANCE.STAGING.STG_TELEMETRY t JOIN ds USING (dataset)
)
"""

# Each case: question, ground-truth SELECT, and scoring options.
#   first_row: the first truth row must appear in the analyst's first row (for "which ..." questions)
#   percent_ok: accept a share expressed as a percentage (0.07 or 7.0)
CASES = [
    # Fleet lifespan
    {"id": "fleet_size", "q": "How many engines are in the fleet?",
     "truth": "SELECT COUNT(*) FROM eng"},
    {"id": "fleet_mttf", "q": "What is the mean time to failure across the whole fleet?",
     "truth": "SELECT AVG(life) FROM eng"},
    {"id": "fleet_median", "q": "What is the median engine lifespan in cycles?",
     "truth": "SELECT MEDIAN(life) FROM eng"},
    {"id": "shortest", "q": "What was the shortest engine life?",
     "truth": "SELECT MIN(life) FROM eng"},
    {"id": "longest", "q": "What's the longest any engine lasted before failing?",
     "truth": "SELECT MAX(life) FROM eng"},
    # Grouped lifespan
    {"id": "mttf_by_dataset", "q": "Show MTTF for each dataset.",
     "truth": "SELECT dataset, AVG(life) FROM eng GROUP BY 1"},
    {"id": "mttf_by_fault", "q": "Compare average engine life for HPC degradation versus HPC and fan degradation.",
     "truth": "SELECT faults, AVG(life) FROM eng GROUP BY 1"},
    {"id": "mttf_by_conditions", "q": "Does the number of operating conditions change mean time to failure? Show MTTF by operating conditions.",
     "truth": "SELECT conditions, AVG(life) FROM eng GROUP BY 1"},
    {"id": "count_fd002", "q": "How many engines are in FD002?",
     "truth": "SELECT COUNT(*) FROM eng WHERE dataset = 'FD002'"},
    {"id": "top_mttf_dataset", "q": "Which dataset has the highest MTTF?",
     "truth": "SELECT dataset FROM eng GROUP BY 1 ORDER BY AVG(life) DESC LIMIT 1", "first_row": True},
    {"id": "most_variable", "q": "Which sub-fleet has the most variable engine life?",
     "truth": "SELECT dataset FROM eng GROUP BY 1 ORDER BY STDDEV_SAMP(life) DESC LIMIT 1", "first_row": True},
    {"id": "longest_engine", "q": "Which engine had the longest life, and how many cycles did it last?",
     "truth": "SELECT engine_id, life FROM eng ORDER BY life DESC LIMIT 1", "first_row": True},
    # Filters on engine facts
    {"id": "under_150", "q": "How many engines failed before reaching 150 cycles?",
     "truth": "SELECT COUNT(*) FROM eng WHERE life < 150"},
    {"id": "fan_over_300", "q": "How many engines with fan degradation lasted more than 300 cycles?",
     "truth": "SELECT COUNT(*) FROM eng WHERE faults = 'HPC and fan degradation' AND life > 300"},
    {"id": "fd001_long_mttf", "q": "What is the average life of FD001 engines that lasted more than 200 cycles?",
     "truth": "SELECT AVG(life) FROM eng WHERE dataset = 'FD001' AND life > 200"},
    # Exposure
    {"id": "total_cycles", "q": "How many operating cycles were flown in total?",
     "truth": "SELECT COUNT(*) FROM cyc"},
    {"id": "fd004_cycles", "q": "How many cycles did FD004 engines fly?",
     "truth": "SELECT COUNT(*) FROM cyc WHERE dataset = 'FD004'"},
    {"id": "critical_share", "q": "What share of all cycles were flown in the critical action window?",
     "truth": "SELECT AVG(IFF(rul <= 15, 1, 0)) FROM cyc", "percent_ok": True},
    {"id": "early_share", "q": "What fraction of cycles were flown within 50 cycles of failure?",
     "truth": "SELECT AVG(IFF(rul <= 50, 1, 0)) FROM cyc", "percent_ok": True},
    {"id": "critical_cycles", "q": "How many cycles did engines spend in the critical action stage?",
     "truth": "SELECT COUNT(*) FROM cyc WHERE rul <= 15"},
    # Sensors and degradation
    {"id": "t30_by_stage_single", "q": "For the single sea-level fleets, what is the average HPC outlet temperature in each health stage?",
     "truth": """SELECT CASE WHEN rul <= 15 THEN 'Critical action' WHEN rul <= 50 THEN 'Early warning' ELSE 'Healthy' END,
                        AVG(hpc_outlet_temp)
                 FROM cyc WHERE conditions = 'Single' GROUP BY 1"""},
    {"id": "t30_rise_fd001", "q": "How much does T30 rise before failure in FD001?",
     "truth": """SELECT AVG(IFF(rul <= 15, hpc_outlet_temp, NULL)) - AVG(IFF(cycle <= 20, hpc_outlet_temp, NULL))
                 FROM cyc WHERE dataset = 'FD001'"""},
    {"id": "t50_rise_by_dataset", "q": "What is the turbine outlet temperature rise to failure for each dataset?",
     "truth": """SELECT dataset, AVG(IFF(rul <= 15, lpt_outlet_temp, NULL)) - AVG(IFF(cycle <= 20, lpt_outlet_temp, NULL))
                 FROM cyc GROUP BY 1"""},
    {"id": "bypass_fd001", "q": "What is the average bypass ratio in FD001?",
     "truth": "SELECT AVG(bypass_ratio) FROM cyc WHERE dataset = 'FD001'"},
    {"id": "core_speed_six", "q": "What's the average core speed for engines flown under six operating conditions?",
     "truth": "SELECT AVG(physical_core_speed) FROM cyc WHERE conditions = 'Six'"},
]


def to_number(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def cell_matches(expected, actual, percent_ok):
    e, a = to_number(expected), to_number(actual)
    if e is not None and a is not None:
        candidates = [a, a / 100] if percent_ok else [a]
        return any(math.isclose(e, c, rel_tol=REL_TOL, abs_tol=1e-9) for c in candidates)
    if e is None and a is not None or e is not None and a is None:
        return False
    return str(expected).strip().lower() == str(actual).strip().lower()


def row_found(truth_row, analyst_rows, percent_ok):
    # A truth row is found if a single analyst row contains every one of its values
    return any(all(any(cell_matches(t, a, percent_ok) for a in arow) for t in truth_row)
               for arow in analyst_rows)


def score(case, truth_rows, analyst_rows):
    percent_ok = case.get("percent_ok", False)
    if not analyst_rows:
        return False, "no rows returned"
    if case.get("first_row"):
        ok = row_found(truth_rows[0], analyst_rows[:1], percent_ok)
        return ok, "" if ok else f"first row {analyst_rows[0]} != expected {truth_rows[0]}"
    missing = [r for r in truth_rows if not row_found(r, analyst_rows, percent_ok)]
    return not missing, "" if not missing else f"missing expected rows {missing[:3]}"


def ask_analyst(conn, question, retries=2):
    # Retry transport errors (timeouts, 5xx) so one slow call doesn't sink a
    # 75-call run; the caller scores a call that still fails as a failed answer.
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                f"https://{conn.host}/api/v2/cortex/analyst/message",
                headers={"Authorization": f'Snowflake Token="{conn.rest.token}"',
                         "Content-Type": "application/json", "Accept": "application/json"},
                json={"messages": [{"role": "user", "content": [{"type": "text", "text": question}]}],
                      "semantic_view": SEMANTIC_VIEW},
                timeout=120,
            )
            resp.raise_for_status()
            break
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(5 * (attempt + 1))
    body = resp.json()
    content = body["message"]["content"]
    sql = next((c["statement"] for c in content if c["type"] == "sql"), None)
    text = " ".join(c["text"] for c in content if c["type"] == "text")
    return sql, text, body.get("response_metadata", {})


def fmt_row(row):
    return tuple(round(float(v), 4) if to_number(v) is not None and not isinstance(v, str) else v for v in row)


def run_eval():
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

        # Each question is asked RUNS times: SQL generation is not deterministic,
        # so a single pass/fail overstates how reliable an answer is.
        results = []
        print(f"Evaluating Cortex Analyst on {len(CASES)} questions x {RUNS} runs against {SEMANTIC_VIEW}\n")
        for case in CASES:
            cursor.execute(TRUTH_CTE + case["truth"])
            truth_rows = [fmt_row(r) for r in cursor.fetchall()]

            attempts = []
            for _ in range(RUNS):
                start = time.time()
                try:
                    sql, text, meta = ask_analyst(conn, case["q"])
                    api_error = None
                except requests.RequestException as e:
                    sql, text, meta, api_error = None, "", {}, str(e)[:300]
                latency = time.time() - start

                analyst_rows, error = [], api_error
                if api_error:
                    ok, reason = False, "API call failed after retries"
                elif sql is None:
                    ok, reason = False, "no SQL generated: " + text[:200]
                else:
                    try:
                        cursor.execute(sql)
                        analyst_rows = [fmt_row(r) for r in cursor.fetchall()]
                        ok, reason = score(case, truth_rows, analyst_rows)
                    except Exception as e:
                        ok, reason, error = False, "generated SQL failed", str(e)[:300]
                attempts.append({"passed": ok, "reason": reason, "latency_s": round(latency, 2),
                                 "returned": analyst_rows[:10], "sql": sql, "error": error,
                                 "model": meta.get("model_names"), "category": meta.get("question_category")})

            n_pass = sum(a["passed"] for a in attempts)
            results.append({"id": case["id"], "question": case["q"], "expected": truth_rows,
                            "passes": n_pass, "runs": RUNS, "attempts": attempts})
            reasons = "; ".join(sorted({a["reason"] for a in attempts if not a["passed"]}))
            print(f"  {n_pass}/{RUNS}  {case['id']:<22} {reasons}")

        attempts = [a for r in results for a in r["attempts"]]
        passed = sum(a["passed"] for a in attempts)
        latencies = sorted(a["latency_s"] for a in attempts)
        summary = {"date": date.today().isoformat(), "semantic_view": SEMANTIC_VIEW,
                   "questions": len(results), "runs_per_question": RUNS,
                   "attempts": len(attempts), "passed": passed, "accuracy": round(passed / len(attempts), 3),
                   "questions_always_correct": sum(r["passes"] == RUNS for r in results),
                   "median_latency_s": latencies[len(latencies) // 2],
                   "models": sorted({m for a in attempts for m in (a["model"] or [])})}
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump({"summary": summary, "results": results}, f, indent=2, default=str)

        print(f"\nAccuracy: {passed}/{len(attempts)} answers ({summary['accuracy']:.1%}); "
              f"{summary['questions_always_correct']}/{len(results)} questions correct on every run; "
              f"median latency {summary['median_latency_s']}s; models {summary['models']}")
        print(f"Saved {os.path.basename(RESULTS_FILE)}")
        return passed == len(attempts)

    except Exception as e:
        print(f"Failed to run Cortex Analyst eval: {e}")
        return False
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    sys.exit(0 if run_eval() else 1)
