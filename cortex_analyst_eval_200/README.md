# Cortex Analyst eval: 200 questions

Generated 2026-10-12 for this repo. The scripts find the repo root (its `.env` and `14_eval_cortex_analyst.py`) from their own location.
It tests Cortex Analyst against the semantic view `PREDICTIVE_MAINTENANCE.SEMANTIC.FLEET_RELIABILITY`.

## Status (2026-10-12)

1. **Full run, original view:** 591/600 answers correct (98.5%). 196/200 questions were right on all 3 runs. Median latency was 3.5 s. The model was `claude-sonnet-4-6`.
2. **The fix:** in 5 of the 9 wrong answers, Cortex returned the two numbers a question asked it to compare but never subtracted them. One sentence was added to the view's `AI_SQL_GENERATION` in `11_build_semantic_view.py`: *"When the user asks for a difference or ratio, compute it in the final SELECT."* The view was rebuilt.
3. **Rerun of the 4 failing questions after the fix:** 12/12 correct.
   - `fleet_range`, `mttf_diff_fault` and `crit_vs_healthy_t50_fd001` are now right on all 3 runs.
   - `t30_rise_fleet` was a **test defect, now fixed**. The view tells Cortex to split sensor metrics by operating conditions, so the expected answer is now split that way (Single 14.7, Six 9.8). It used to be one fleet-wide 11.2.
   - Cortex's stored answers were rescored without new calls. That moves the full run to **594/600 (99.0%)**. The original 591/600 is the score against the old expected answer.
4. **Regression check:** the original 25 questions were rerun once against the changed view and all 25 were correct (`regression_original25_after_fix.json`). The other 196 new questions have not been rerun. One run of them costs about 13 credits.
5. **Merged:** the change is in `main` via PR #10 (https://github.com/livingdw67/cmapss-predictive-maintenance/pull/10).

## Files

- `cortex_analyst_questions_200.json`: the question set. Each case has:
  - `id`, `category` (10 categories) and `difficulty` (96 easy, 74 medium, 30 hard)
  - `q`: the question in plain English
  - `truth`: ground-truth SQL, run after the `truth_cte` at the top of the file
  - `expected`: rows that `truth` returned on 2026-10-12, already computed
  - optional `first_row` and `percent_ok` flags, used the same way as in `14_eval_cortex_analyst.py`
- `gen_questions.py`: rebuilds the question JSON. Usage: `<repo>\.venv\Scripts\python gen_questions.py <out_dir>`.
- `run_eval_200.py`: runs the eval and reuses `ask_analyst` and `score` from the repo's `14_eval_cortex_analyst.py`.
  - It saves after every question and skips questions already in the results file, so an interrupted run resumes.
  - `--only id1,id2` runs only those questions, and `--out name.json` writes to a different results file.
- `cortex_analyst_eval_200_results.json`: the full run on the original view. It has every attempt with its SQL and returned rows, a summary, and accuracy `by_category` and `by_difficulty`.
- `rerun_after_fix_results.json`: the 4 questions rerun after the fix.
- `run.log`: progress log from the full run.

## How the ground truth was built

- Every truth query reads `STAGING.STG_TELEMETRY`, never the semantic view, so the eval can't grade the view against itself.
- All 200 truth queries ran without errors or nulls.
- Ranking questions were checked for ties. FD001_039 and FD002_244 tie at 128 cycles, so `shortest_engine` expects both engines.
- None of these questions repeat the 25 in the existing `cortex_analyst_eval.json`.

## Cost and role

Cortex Analyst costs about 0.067 credits per question.
Run under the owner role from the repo's `.env`, not the public demo's `FLEET_DEMO` role, which is capped at 10 credits a month.
Confirm the run count with the user before a large run.

## Scoring notes

- A number is correct when it is within 0.5% of the expected value.
- `first_row` means the first expected row must appear in Cortex's first row.
- `percent_ok` accepts a share given either as a fraction or as a percentage.
- Questions that don't name a stage threshold use the view's definitions: critical action means RUL ≤ 15, early warning means RUL ≤ 50, and the baseline is the first 20 cycles.
