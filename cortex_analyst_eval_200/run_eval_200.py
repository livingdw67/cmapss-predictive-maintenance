"""Run the 200-question set through Cortex Analyst, 3 runs per question.

Reuses ask_analyst/score/fmt_row from the repo's 14_eval_cortex_analyst.py.
Saves after every question and skips questions already in the results file,
so an interrupted run resumes without paying for answers twice.

Usage (from anywhere):
    <repo>\\.venv\\Scripts\\python run_eval_200.py
    <repo>\\.venv\\Scripts\\python run_eval_200.py --only id1,id2 --out rerun.json
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from datetime import date

import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
QUESTIONS = os.path.join(HERE, "cortex_analyst_questions_200.json")
RESULTS = os.path.join(HERE, "cortex_analyst_eval_200_results.json")
RUNS = 3

from dotenv import load_dotenv
load_dotenv(os.path.join(REPO, ".env"))
spec = importlib.util.spec_from_file_location("ev", os.path.join(REPO, "14_eval_cortex_analyst.py"))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def summarize(results, cases):
    meta = {c["id"]: c for c in cases}
    attempts = [a for r in results for a in r["attempts"]]
    passed = sum(a["passed"] for a in attempts)
    lat = sorted(a["latency_s"] for a in attempts)
    groups = {}
    for key in ("category", "difficulty"):
        g = defaultdict(lambda: [0, 0])
        for r in results:
            k = meta[r["id"]][key]
            g[k][0] += r["passes"]
            g[k][1] += r["runs"]
        groups[key] = {k: {"passed": p, "attempts": n, "accuracy": round(p / n, 3)} for k, (p, n) in sorted(g.items())}
    return {"date": date.today().isoformat(), "semantic_view": ev.SEMANTIC_VIEW,
            "questions": len(results), "runs_per_question": RUNS, "attempts": len(attempts),
            "passed": passed, "accuracy": round(passed / len(attempts), 3) if attempts else None,
            "questions_always_correct": sum(r["passes"] == r["runs"] for r in results),
            "questions_never_correct": sum(r["passes"] == 0 for r in results),
            "median_latency_s": lat[len(lat) // 2] if lat else None,
            "models": sorted({m for a in attempts for m in (a["model"] or [])}),
            "by_category": groups["category"], "by_difficulty": groups["difficulty"]}


def main():
    global RESULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated case ids to run")
    ap.add_argument("--out", help="results file name (default cortex_analyst_eval_200_results.json)")
    args = ap.parse_args()
    if args.out:
        RESULTS = os.path.join(HERE, args.out)
    cases = json.load(open(QUESTIONS, encoding="utf-8"))["cases"]
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
        assert len(cases) == len(wanted), "unknown id in --only"
    results = []
    if os.path.exists(RESULTS):
        results = json.load(open(RESULTS, encoding="utf-8"))["results"]
    done = {r["id"] for r in results}

    conn = ev.snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USER'), private_key_file=os.getenv('SNOWFLAKE_PRIVATE_KEY_FILE'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'), warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
        database=os.getenv('SNOWFLAKE_DATABASE'), role=os.getenv('SNOWFLAKE_ROLE'))
    cur = conn.cursor()
    print(f"{len(cases)} questions x {RUNS} runs; {len(done)} already done", flush=True)

    for i, case in enumerate(cases, 1):
        if case["id"] in done:
            continue
        truth_rows = [ev.fmt_row(r) for r in case["expected"]]
        attempts = []
        for _ in range(RUNS):
            start = time.time()
            try:
                sql, text, meta = ev.ask_analyst(conn, case["q"])
                api_error = None
            except requests.RequestException as e:
                sql, text, meta, api_error = None, "", {}, str(e)[:300]
            latency = time.time() - start
            rows, error = [], api_error
            if api_error:
                ok, reason = False, "API call failed after retries"
            elif sql is None:
                ok, reason = False, "no SQL generated: " + text[:200]
            else:
                try:
                    cur.execute(sql)
                    rows = [ev.fmt_row(r) for r in cur.fetchall()]
                    ok, reason = ev.score(case, truth_rows, rows)
                except Exception as e:
                    ok, reason, error = False, "generated SQL failed", str(e)[:300]
            attempts.append({"passed": ok, "reason": reason, "latency_s": round(latency, 2),
                             "returned": rows[:10], "sql": sql, "text": text[:500], "error": error,
                             "model": meta.get("model_names"), "category": meta.get("question_category")})
        n = sum(a["passed"] for a in attempts)
        results.append({"id": case["id"], "category": case["category"], "difficulty": case["difficulty"],
                        "question": case["q"], "expected": truth_rows, "passes": n, "runs": RUNS,
                        "attempts": attempts})
        with open(RESULTS, "w", encoding="utf-8") as f:
            json.dump({"summary": summarize(results, cases), "results": results}, f, indent=2, default=str)
        reasons = "; ".join(sorted({a["reason"] for a in attempts if not a["passed"]}))[:160]
        print(f"[{i:3}/{len(cases)}] {n}/{RUNS} {case['id']:<30} {reasons}", flush=True)

    conn.close()
    s = summarize(results, cases)
    print(json.dumps({k: s[k] for k in ("accuracy", "passed", "attempts", "questions_always_correct",
                                         "questions_never_correct", "median_latency_s", "models")}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
