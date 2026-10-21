"""Build 200 Cortex Analyst eval questions with ground truth computed from STAGING.

Writes cortex_analyst_questions_200.json into the output folder given as argv[1].
"""
import json
import os
import sys
from datetime import date
from decimal import Decimal

import snowflake.connector
from dotenv import load_dotenv

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(REPO, ".env"))
sys.path.insert(0, REPO)

# Same ground-truth CTE as 14_eval_cortex_analyst.py, so cases drop straight into it
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

STAGE = "CASE WHEN rul <= 15 THEN 'Critical action' WHEN rul <= 50 THEN 'Early warning' ELSE 'Healthy' END"
RISE = "AVG(IFF(rul <= 15, {c}, NULL)) - AVG(IFF(cycle <= 20, {c}, NULL))"

CASES = []


def add(id, cat, diff, q, truth, **opts):
    CASES.append({"id": id, "category": cat, "difficulty": diff, "q": q, "truth": " ".join(truth.split()), **opts})


# ---------------------------------------------------------------- 1. fleet lifespan (scalar)
c = "fleet_lifespan"
add("fleet_count_units", c, "easy", "How many turbofan units do we have data for?", "SELECT COUNT(*) FROM eng")
add("fleet_failures", c, "easy", "How many failures are recorded in total?", "SELECT COUNT(*) FROM eng")
add("fleet_avg_life", c, "easy", "What's the average life of an engine in cycles?", "SELECT AVG(life) FROM eng")
add("fleet_mean_life", c, "easy", "On average, how many cycles does an engine survive before it fails?", "SELECT AVG(life) FROM eng")
add("fleet_median_ttf", c, "easy", "Give me the median time to failure for all engines.", "SELECT MEDIAN(life) FROM eng")
add("fleet_min_life", c, "easy", "What is the minimum number of cycles any engine completed?", "SELECT MIN(life) FROM eng")
add("fleet_earliest_failure", c, "easy", "When did the earliest failure happen, in cycles?", "SELECT MIN(life) FROM eng")
add("fleet_max_life", c, "easy", "What is the maximum engine life in the data?", "SELECT MAX(life) FROM eng")
add("fleet_stddev", c, "easy", "What is the standard deviation of engine lifespan?", "SELECT STDDEV_SAMP(life) FROM eng")
add("fleet_spread", c, "medium", "How spread out are engine lifespans? Give the standard deviation in cycles.", "SELECT STDDEV_SAMP(life) FROM eng")
add("fleet_range", c, "medium", "What's the difference between the longest and shortest engine life?", "SELECT MAX(life) - MIN(life) FROM eng")
add("fleet_mttf_and_median", c, "medium", "Show both the mean and the median time to failure for the fleet.", "SELECT AVG(life), MEDIAN(life) FROM eng")
add("fleet_min_max", c, "easy", "What are the shortest and longest engine lives?", "SELECT MIN(life), MAX(life) FROM eng")
add("fleet_summary", c, "medium", "Summarize engine life: count, mean, median, min and max.", "SELECT COUNT(*), AVG(life), MEDIAN(life), MIN(life), MAX(life) FROM eng")
add("fleet_assets", c, "easy", "How many assets are in the fleet?", "SELECT COUNT(*) FROM eng")

# ---------------------------------------------------------------- 2. per-dataset scalar
c = "dataset_lifespan"
names = {"FD001": "FD001", "FD002": "FD002", "FD003": "FD003", "FD004": "FD004"}
for ds in names:
    n = ds.lower()
    add(f"{n}_count", c, "easy", f"How many engines does {ds} contain?", f"SELECT COUNT(*) FROM eng WHERE dataset = '{ds}'")
    add(f"{n}_mttf", c, "easy", f"What's the MTTF for {ds}?", f"SELECT AVG(life) FROM eng WHERE dataset = '{ds}'")
    add(f"{n}_median", c, "easy", f"Median engine life in {ds}?", f"SELECT MEDIAN(life) FROM eng WHERE dataset = '{ds}'")
    add(f"{n}_shortest", c, "easy", f"What was the shortest-lived engine's life in {ds}?", f"SELECT MIN(life) FROM eng WHERE dataset = '{ds}'")
    add(f"{n}_longest", c, "easy", f"Longest life of any {ds} engine?", f"SELECT MAX(life) FROM eng WHERE dataset = '{ds}'")
add("fd002_stddev", c, "easy", "What is the lifespan standard deviation in FD002?", "SELECT STDDEV_SAMP(life) FROM eng WHERE dataset = 'FD002'")
add("fd003_stddev", c, "easy", "How variable is engine life in FD003 (std dev)?", "SELECT STDDEV_SAMP(life) FROM eng WHERE dataset = 'FD003'")

# ---------------------------------------------------------------- 3. grouped lifespan
c = "grouped_lifespan"
add("count_by_dataset", c, "easy", "How many engines are in each dataset?", "SELECT dataset, COUNT(*) FROM eng GROUP BY 1")
add("median_by_dataset", c, "easy", "Show median life by sub-dataset.", "SELECT dataset, MEDIAN(life) FROM eng GROUP BY 1")
add("min_by_dataset", c, "easy", "What is the shortest life in each FD dataset?", "SELECT dataset, MIN(life) FROM eng GROUP BY 1")
add("max_by_dataset", c, "easy", "List the maximum lifespan per dataset.", "SELECT dataset, MAX(life) FROM eng GROUP BY 1")
add("stddev_by_dataset", c, "easy", "Show lifespan variability (standard deviation) for each dataset.", "SELECT dataset, STDDEV_SAMP(life) FROM eng GROUP BY 1")
add("count_by_fault", c, "easy", "How many engines failed from each fault mode?", "SELECT faults, COUNT(*) FROM eng GROUP BY 1")
add("count_by_conditions", c, "easy", "Break down the engine count by operating conditions.", "SELECT conditions, COUNT(*) FROM eng GROUP BY 1")
add("median_by_fault", c, "easy", "What's the median life for each degradation mode?", "SELECT faults, MEDIAN(life) FROM eng GROUP BY 1")
add("median_by_conditions", c, "easy", "Median time to failure by flight conditions?", "SELECT conditions, MEDIAN(life) FROM eng GROUP BY 1")
add("min_by_fault", c, "easy", "Earliest failure for each fault type?", "SELECT faults, MIN(life) FROM eng GROUP BY 1")
add("max_by_conditions", c, "easy", "Longest life under each operating regime (single vs six conditions)?", "SELECT conditions, MAX(life) FROM eng GROUP BY 1")
add("mttf_cond_fault", c, "medium", "Show MTTF broken down by both operating conditions and fault mode.", "SELECT conditions, faults, AVG(life) FROM eng GROUP BY 1, 2")
add("count_cond_fault", c, "medium", "How many engines are there for each combination of operating conditions and fault mode?", "SELECT conditions, faults, COUNT(*) FROM eng GROUP BY 1, 2")
add("stats_by_dataset", c, "medium", "For each dataset give engine count, MTTF and shortest life.", "SELECT dataset, COUNT(*), AVG(life), MIN(life) FROM eng GROUP BY 1")
add("stddev_by_fault", c, "easy", "Is engine life more variable with fan degradation? Show std dev of life by fault mode.", "SELECT faults, STDDEV_SAMP(life) FROM eng GROUP BY 1")
add("stddev_by_conditions", c, "easy", "Lifespan standard deviation by operating conditions.", "SELECT conditions, STDDEV_SAMP(life) FROM eng GROUP BY 1")
add("mttf_by_fault_single", c, "medium", "Among single-condition fleets only, compare MTTF by fault mode.", "SELECT faults, AVG(life) FROM eng WHERE conditions = 'Single' GROUP BY 1")
add("mttf_by_fault_six", c, "medium", "Within the six-condition fleets, what's the mean life for each fault mode?", "SELECT faults, AVG(life) FROM eng WHERE conditions = 'Six' GROUP BY 1")
add("mttf_by_cond_fan", c, "medium", "For engines with HPC and fan degradation, how does MTTF differ between single and six operating conditions?", "SELECT conditions, AVG(life) FROM eng WHERE faults = 'HPC and fan degradation' GROUP BY 1")

# ---------------------------------------------------------------- 4. rankings
c = "ranking"
add("lowest_mttf_dataset", c, "easy", "Which dataset has the lowest mean time to failure?", "SELECT dataset FROM eng GROUP BY 1 ORDER BY AVG(life) ASC LIMIT 1", first_row=True)
add("most_engines_dataset", c, "easy", "Which dataset has the most engines?", "SELECT dataset FROM eng GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 1", first_row=True)
add("fewest_cycles_dataset", c, "medium", "Which sub-fleet flew the fewest total cycles?", "SELECT dataset FROM cyc GROUP BY 1 ORDER BY COUNT(*) ASC LIMIT 1", first_row=True)
add("most_cycles_dataset", c, "medium", "Which dataset logged the most operating cycles, and how many?", "SELECT dataset, COUNT(*) FROM cyc GROUP BY 1 ORDER BY 2 DESC LIMIT 1", first_row=True)
add("highest_median_dataset", c, "easy", "Which dataset has the highest median engine life?", "SELECT dataset FROM eng GROUP BY 1 ORDER BY MEDIAN(life) DESC LIMIT 1", first_row=True)
add("least_variable_dataset", c, "medium", "Which sub-fleet has the most consistent engine life (lowest standard deviation)?", "SELECT dataset FROM eng GROUP BY 1 ORDER BY STDDEV_SAMP(life) ASC LIMIT 1", first_row=True)
# Two engines tie at the minimum life, so the truth lists both
add("shortest_engine", c, "medium", "Which engine or engines failed soonest, and after how many cycles?", "SELECT engine_id, life FROM eng WHERE life = (SELECT MIN(life) FROM eng)")
add("top5_longest", c, "medium", "List the five longest-lived engines with their lifespans.", "SELECT engine_id, life FROM eng ORDER BY life DESC, engine_id LIMIT 5")
add("top5_shortest", c, "medium", "Show the 5 engines with the shortest life.", "SELECT engine_id, life FROM eng ORDER BY life ASC, engine_id LIMIT 5")
for ds in names:
    n = ds.lower()
    add(f"{n}_longest_engine", c, "medium", f"Which {ds} engine lasted the longest?", f"SELECT engine_id, life FROM eng WHERE dataset = '{ds}' ORDER BY life DESC, engine_id LIMIT 1", first_row=True)
    add(f"{n}_shortest_engine", c, "medium", f"Which engine in {ds} had the shortest life?", f"SELECT engine_id, life FROM eng WHERE dataset = '{ds}' ORDER BY life ASC, engine_id LIMIT 1", first_row=True)
add("longest_fault_mode", c, "easy", "Which fault mode is associated with longer average engine life?", "SELECT faults FROM eng GROUP BY 1 ORDER BY AVG(life) DESC LIMIT 1", first_row=True)
add("hottest_t30_single_dataset", c, "hard", "Among the single-condition datasets, which has the higher average HPC outlet temperature?", "SELECT dataset FROM cyc WHERE conditions = 'Single' GROUP BY 1 ORDER BY AVG(hpc_outlet_temp) DESC LIMIT 1", first_row=True)
add("biggest_t30_rise_dataset", c, "hard", "Which dataset shows the largest T30 rise before failure?", f"SELECT dataset FROM cyc GROUP BY 1 ORDER BY {RISE.format(c='hpc_outlet_temp')} DESC LIMIT 1", first_row=True)
add("highest_critical_share_dataset", c, "hard", "Which dataset spends the largest share of its cycles in the critical action window?", "SELECT dataset FROM cyc GROUP BY 1 ORDER BY AVG(IFF(rul <= 15, 1, 0)) DESC LIMIT 1", first_row=True)
add("top3_fd002", c, "medium", "Top 3 longest-lived engines in FD002?", "SELECT engine_id, life FROM eng WHERE dataset = 'FD002' ORDER BY life DESC, engine_id LIMIT 3")

# ---------------------------------------------------------------- 5. single-engine lookups
c = "engine_lookup"
for eid in ["FD001_001", "FD001_069", "FD002_118", "FD003_055", "FD004_118", "FD004_249"]:
    n = eid.lower()
    add(f"{n}_life", c, "easy", f"How many cycles did engine {eid} last?", f"SELECT life FROM eng WHERE engine_id = '{eid}'")
add("fd002_007_cycles", c, "easy", "How many flights did unit FD002_007 fly?", "SELECT COUNT(*) FROM cyc WHERE engine_id = 'FD002_007'")
add("fd003_010_t30", c, "medium", "What was the average HPC outlet temperature for engine FD003_010?", "SELECT AVG(hpc_outlet_temp) FROM cyc WHERE engine_id = 'FD003_010'")
add("fd001_050_t50", c, "medium", "Average LPT outlet temperature for FD001_050?", "SELECT AVG(lpt_outlet_temp) FROM cyc WHERE engine_id = 'FD001_050'")
add("fd001_020_stage", c, "medium", "How many cycles did FD001_020 spend in each health stage?", f"SELECT {STAGE}, COUNT(*) FROM cyc WHERE engine_id = 'FD001_020' GROUP BY 1")
add("fd003_099_rise", c, "hard", "How much did T30 rise before failure for engine FD003_099?", f"SELECT {RISE.format(c='hpc_outlet_temp')} FROM cyc WHERE engine_id = 'FD003_099'")
add("fd001_017_bpr", c, "medium", "What was engine FD001_017's average bypass ratio?", "SELECT AVG(bypass_ratio) FROM cyc WHERE engine_id = 'FD001_017'")
add("fd001_017_core_crit", c, "hard", "What was FD001_017's average core speed during its critical action window?", "SELECT AVG(physical_core_speed) FROM cyc WHERE engine_id = 'FD001_017' AND rul <= 15")
add("fd003_001_vs_median", c, "hard", "Did engine FD003_001 last longer than the FD003 median? Give its life and the dataset median.",
    "SELECT (SELECT life FROM eng WHERE engine_id = 'FD003_001'), (SELECT MEDIAN(life) FROM eng WHERE dataset = 'FD003')")

# ---------------------------------------------------------------- 6. lifespan filters
c = "lifespan_filter"
add("over_200", c, "easy", "How many engines lasted more than 200 cycles?", "SELECT COUNT(*) FROM eng WHERE life > 200")
add("over_300", c, "easy", "How many engines survived beyond 300 cycles?", "SELECT COUNT(*) FROM eng WHERE life > 300")
add("under_175", c, "easy", "Count engines that failed in under 175 cycles.", "SELECT COUNT(*) FROM eng WHERE life < 175")
add("between_150_250", c, "medium", "How many engines had a life between 150 and 250 cycles inclusive?", "SELECT COUNT(*) FROM eng WHERE life BETWEEN 150 AND 250")
add("at_least_250", c, "easy", "How many engines reached at least 250 cycles?", "SELECT COUNT(*) FROM eng WHERE life >= 250")
add("over_200_by_dataset", c, "medium", "For each dataset, how many engines lasted more than 200 cycles?", "SELECT dataset, COUNT(*) FROM eng WHERE life > 200 GROUP BY 1")
add("under_150_by_fault", c, "medium", "By fault mode, how many engines failed before 150 cycles?", "SELECT faults, COUNT(*) FROM eng WHERE life < 150 GROUP BY 1")
add("fd002_over_250", c, "easy", "How many FD002 engines lasted over 250 cycles?", "SELECT COUNT(*) FROM eng WHERE dataset = 'FD002' AND life > 250")
add("fd004_under_200", c, "easy", "In FD004, how many engines failed before cycle 200?", "SELECT COUNT(*) FROM eng WHERE dataset = 'FD004' AND life < 200")
add("fd003_over_300", c, "easy", "How many FD003 units ran past 300 cycles?", "SELECT COUNT(*) FROM eng WHERE dataset = 'FD003' AND life > 300")
add("hpc_only_under_150", c, "medium", "How many HPC-degradation-only engines failed before 150 cycles?", "SELECT COUNT(*) FROM eng WHERE faults = 'HPC degradation' AND life < 150")
add("six_over_300", c, "medium", "How many six-condition engines lasted over 300 cycles?", "SELECT COUNT(*) FROM eng WHERE conditions = 'Six' AND life > 300")
add("mttf_over_250", c, "medium", "What's the average life of engines that made it past 250 cycles?", "SELECT AVG(life) FROM eng WHERE life > 250")
add("mttf_under_150", c, "medium", "Among engines that failed before 150 cycles, what's the average life?", "SELECT AVG(life) FROM eng WHERE life < 150")
add("fd004_mttf_over_200", c, "medium", "Mean life of FD004 engines that exceeded 200 cycles?", "SELECT AVG(life) FROM eng WHERE dataset = 'FD004' AND life > 200")
add("median_over_200_by_ds", c, "hard", "For engines that lasted over 200 cycles, what is the median life in each dataset?", "SELECT dataset, MEDIAN(life) FROM eng WHERE life > 200 GROUP BY 1")
add("pct_over_200", c, "hard", "What percentage of engines lasted more than 200 cycles?", "SELECT AVG(IFF(life > 200, 1, 0)) FROM eng", percent_ok=True)
add("pct_under_150_fd001", c, "hard", "What fraction of FD001 engines failed before 150 cycles?", "SELECT AVG(IFF(life < 150, 1, 0)) FROM eng WHERE dataset = 'FD001'", percent_ok=True)
add("pct_over_300_by_fault", c, "hard", "For each fault mode, what share of engines lasted over 300 cycles?", "SELECT faults, AVG(IFF(life > 300, 1, 0)) FROM eng GROUP BY 1", percent_ok=True)
add("above_fleet_mttf", c, "hard", "How many engines outlived the fleet-wide mean time to failure?", "SELECT COUNT(*) FROM eng WHERE life > (SELECT AVG(life) FROM eng)")
add("fd001_above_own_mttf", c, "hard", "How many FD001 engines lasted longer than FD001's own MTTF?", "SELECT COUNT(*) FROM eng WHERE dataset = 'FD001' AND life > (SELECT AVG(life) FROM eng WHERE dataset = 'FD001')")
add("count_200_to_300_six", c, "medium", "Among six-condition engines, how many lasted between 200 and 300 cycles inclusive?", "SELECT COUNT(*) FROM eng WHERE conditions = 'Six' AND life BETWEEN 200 AND 300")

# ---------------------------------------------------------------- 7. exposure / health stages
c = "exposure"
add("total_flights", c, "easy", "How many flights are in the telemetry in total?", "SELECT COUNT(*) FROM cyc")
add("cycles_by_dataset", c, "easy", "Total operating cycles per dataset?", "SELECT dataset, COUNT(*) FROM cyc GROUP BY 1")
add("cycles_by_fault", c, "easy", "How many cycles were flown by each fault mode group?", "SELECT faults, COUNT(*) FROM cyc GROUP BY 1")
add("cycles_by_conditions", c, "easy", "Number of readings by operating conditions.", "SELECT conditions, COUNT(*) FROM cyc GROUP BY 1")
add("cycles_by_stage", c, "easy", "How many cycles fall into each health stage?", f"SELECT {STAGE}, COUNT(*) FROM cyc GROUP BY 1")
add("healthy_cycles", c, "easy", "How many cycles were flown while engines were healthy?", "SELECT COUNT(*) FROM cyc WHERE rul > 50")
add("early_warning_cycles", c, "medium", "How many cycles were in the early warning stage (not critical)?", "SELECT COUNT(*) FROM cyc WHERE rul > 15 AND rul <= 50")
add("within_50_cycles", c, "easy", "How many cycles were flown within 50 cycles of failure?", "SELECT COUNT(*) FROM cyc WHERE rul <= 50")
add("stage_by_dataset", c, "medium", "Show cycle counts by dataset and health stage.", f"SELECT dataset, {STAGE}, COUNT(*) FROM cyc GROUP BY 1, 2")
add("critical_share_by_dataset", c, "medium", "What fraction of cycles is in the critical action window for each dataset?", "SELECT dataset, AVG(IFF(rul <= 15, 1, 0)) FROM cyc GROUP BY 1", percent_ok=True)
add("early_share_by_dataset", c, "medium", "Early-warning cycle share by dataset?", "SELECT dataset, AVG(IFF(rul <= 50, 1, 0)) FROM cyc GROUP BY 1", percent_ok=True)
add("early_share_by_fault", c, "medium", "By fault mode, what share of cycles were within 50 cycles of failure?", "SELECT faults, AVG(IFF(rul <= 50, 1, 0)) FROM cyc GROUP BY 1", percent_ok=True)
add("critical_share_by_cond", c, "medium", "Critical action cycle share by operating conditions.", "SELECT conditions, AVG(IFF(rul <= 15, 1, 0)) FROM cyc GROUP BY 1", percent_ok=True)
add("critical_share_fd002", c, "easy", "What percent of FD002's cycles were in the critical action stage?", "SELECT AVG(IFF(rul <= 15, 1, 0)) FROM cyc WHERE dataset = 'FD002'", percent_ok=True)
add("early_share_fd003", c, "easy", "What share of FD003 cycles were flown in early warning or worse?", "SELECT AVG(IFF(rul <= 50, 1, 0)) FROM cyc WHERE dataset = 'FD003'", percent_ok=True)
add("healthy_share", c, "hard", "What fraction of all cycles were flown in the healthy stage?", "SELECT AVG(IFF(rul > 50, 1, 0)) FROM cyc", percent_ok=True)
add("fd001_critical_cycles", c, "easy", "How many critical action cycles did FD001 have?", "SELECT COUNT(*) FROM cyc WHERE dataset = 'FD001' AND rul <= 15")
add("fd004_healthy_cycles", c, "medium", "How many healthy cycles did FD004 fly?", "SELECT COUNT(*) FROM cyc WHERE dataset = 'FD004' AND rul > 50")
add("baseline_cycles", c, "medium", "How many cycles fall in the baseline window (first 20 cycles of each engine)?", "SELECT COUNT(*) FROM cyc WHERE cycle <= 20")
add("engines_reaching_critical", c, "medium", "How many engines reached the critical action stage?", "SELECT COUNT(DISTINCT engine_id) FROM cyc WHERE rul <= 15")
add("avg_cycles_per_engine_fd002", c, "medium", "On average, how many cycles did each FD002 engine fly?", "SELECT COUNT(*) / COUNT(DISTINCT engine_id) FROM cyc WHERE dataset = 'FD002'")
add("cycles_past_200", c, "medium", "How many cycles were flown at cycle number above 200?", "SELECT COUNT(*) FROM cyc WHERE cycle > 200")
add("cycles_rul_over_100", c, "medium", "How many readings had more than 100 cycles of remaining useful life?", "SELECT COUNT(*) FROM cyc WHERE rul > 100")
add("avg_rul", c, "medium", "What's the average remaining useful life across all readings?", "SELECT AVG(rul) FROM cyc")
add("avg_rul_by_dataset", c, "medium", "Average RUL per dataset across all cycles?", "SELECT dataset, AVG(rul) FROM cyc GROUP BY 1")

# ---------------------------------------------------------------- 8. sensors
c = "sensor"
sensors = {"hpc_outlet_temp": ("HPC outlet temperature", "T30"),
           "lpt_outlet_temp": ("LPT outlet temperature", "T50"),
           "physical_core_speed": ("core speed", "Nc"),
           "static_pressure_hpc_outlet": ("HPC outlet static pressure", "Ps30"),
           "bypass_ratio": ("bypass ratio", "BPR")}
for col, (label, short) in sensors.items():
    add(f"{col}_by_dataset", c, "easy", f"What is the average {label} in each dataset?", f"SELECT dataset, AVG({col}) FROM cyc GROUP BY 1")
    add(f"{col}_fd003", c, "easy", f"Average {short} for FD003?", f"SELECT AVG({col}) FROM cyc WHERE dataset = 'FD003'")
    add(f"{col}_stage_fd001", c, "medium", f"In FD001, how does average {label} change across health stages?", f"SELECT {STAGE}, AVG({col}) FROM cyc WHERE dataset = 'FD001' GROUP BY 1")
    add(f"{col}_by_cond", c, "easy", f"Average {label} by operating conditions?", f"SELECT conditions, AVG({col}) FROM cyc GROUP BY 1")
add("t50_fleet", c, "easy", "What's the fleet-wide average exhaust (LPT outlet) temperature?", "SELECT AVG(lpt_outlet_temp) FROM cyc")
add("t30_fleet", c, "easy", "Average compressor outlet temperature across all readings?", "SELECT AVG(hpc_outlet_temp) FROM cyc")
add("ps30_fd002", c, "easy", "Average Ps30 in FD002?", "SELECT AVG(static_pressure_hpc_outlet) FROM cyc WHERE dataset = 'FD002'")
add("t30_single_by_fault", c, "medium", "Within single-condition fleets, compare average T30 by fault mode.", "SELECT faults, AVG(hpc_outlet_temp) FROM cyc WHERE conditions = 'Single' GROUP BY 1")
add("t50_single_by_stage", c, "medium", "For single sea-level fleets, show average T50 in each health stage.", f"SELECT {STAGE}, AVG(lpt_outlet_temp) FROM cyc WHERE conditions = 'Single' GROUP BY 1")
add("bpr_fd003_by_stage", c, "medium", "How does bypass ratio change by health stage in FD003?", f"SELECT {STAGE}, AVG(bypass_ratio) FROM cyc WHERE dataset = 'FD003' GROUP BY 1")
add("core_fd001_critical", c, "medium", "What's the average core speed in FD001 during the critical action window?", "SELECT AVG(physical_core_speed) FROM cyc WHERE dataset = 'FD001' AND rul <= 15")
add("t30_fd003_healthy", c, "medium", "Average HPC outlet temp for FD003 engines while healthy?", "SELECT AVG(hpc_outlet_temp) FROM cyc WHERE dataset = 'FD003' AND rul > 50")
add("t50_fd001_baseline", c, "medium", "What's FD001's average T50 over the first 20 cycles of each engine?", "SELECT AVG(lpt_outlet_temp) FROM cyc WHERE dataset = 'FD001' AND cycle <= 20")
add("ps30_single_critical", c, "medium", "Average static pressure at HPC outlet in the critical window for single-condition fleets?", "SELECT AVG(static_pressure_hpc_outlet) FROM cyc WHERE conditions = 'Single' AND rul <= 15")
add("t30_fd001_by_engine_top", c, "hard", "Which FD001 engine had the highest average HPC outlet temperature?", "SELECT engine_id FROM cyc WHERE dataset = 'FD001' GROUP BY 1 ORDER BY AVG(hpc_outlet_temp) DESC LIMIT 1", first_row=True)

# ---------------------------------------------------------------- 9. degradation (temperature rise)
c = "degradation"
# The view tells Cortex not to mix Single and Six condition sensor values, so the
# correct fleet-wide answer is split by operating conditions
add("t30_rise_fleet", c, "medium", "What's the HPC temperature rise to failure across the whole fleet?", f"SELECT conditions, {RISE.format(c='hpc_outlet_temp')} FROM cyc GROUP BY 1")
add("t30_rise_by_dataset", c, "medium", "Show T30 degradation for each dataset.", f"SELECT dataset, {RISE.format(c='hpc_outlet_temp')} FROM cyc GROUP BY 1")
add("t30_rise_fd003", c, "easy", "How much does compressor temperature drift before failure in FD003?", f"SELECT {RISE.format(c='hpc_outlet_temp')} FROM cyc WHERE dataset = 'FD003'")
add("t30_rise_fd002", c, "easy", "HPC temperature rise to failure for FD002?", f"SELECT {RISE.format(c='hpc_outlet_temp')} FROM cyc WHERE dataset = 'FD002'")
add("t30_rise_by_fault_single", c, "medium", "For single-condition fleets, compare T30 rise to failure between fault modes.", f"SELECT faults, {RISE.format(c='hpc_outlet_temp')} FROM cyc WHERE conditions = 'Single' GROUP BY 1")
add("t30_rise_by_cond", c, "medium", "HPC temperature rise by operating conditions.", f"SELECT conditions, {RISE.format(c='hpc_outlet_temp')} FROM cyc GROUP BY 1")
add("t50_rise_fleet", c, "medium", "What's the EGT margin loss (T50 rise to failure) fleet-wide?", f"SELECT {RISE.format(c='lpt_outlet_temp')} FROM cyc")
add("t50_rise_fd001", c, "easy", "Turbine temperature drift before failure in FD001?", f"SELECT {RISE.format(c='lpt_outlet_temp')} FROM cyc WHERE dataset = 'FD001'")
add("t50_rise_fd004", c, "easy", "T50 degradation for FD004?", f"SELECT {RISE.format(c='lpt_outlet_temp')} FROM cyc WHERE dataset = 'FD004'")
add("t50_rise_by_fault", c, "medium", "Show LPT temperature rise to failure by fault mode.", f"SELECT faults, {RISE.format(c='lpt_outlet_temp')} FROM cyc GROUP BY 1")
add("t50_rise_single_by_ds", c, "medium", "For the single-condition datasets, what's the T50 rise before failure in each?", f"SELECT dataset, {RISE.format(c='lpt_outlet_temp')} FROM cyc WHERE conditions = 'Single' GROUP BY 1")
add("both_rises_fd001", c, "medium", "For FD001, give both the HPC and LPT temperature rise to failure.", f"SELECT {RISE.format(c='hpc_outlet_temp')}, {RISE.format(c='lpt_outlet_temp')} FROM cyc WHERE dataset = 'FD001'")
add("both_rises_by_ds", c, "hard", "Show T30 and T50 rise to failure side by side for every dataset.", f"SELECT dataset, {RISE.format(c='hpc_outlet_temp')}, {RISE.format(c='lpt_outlet_temp')} FROM cyc GROUP BY 1")
add("t30_baseline_vs_critical_fd003", c, "hard", "In FD003, what's the average T30 in the baseline window versus the critical action window?",
    "SELECT AVG(IFF(cycle <= 20, hpc_outlet_temp, NULL)), AVG(IFF(rul <= 15, hpc_outlet_temp, NULL)) FROM cyc WHERE dataset = 'FD003'")
add("t30_rise_fd001_long", c, "hard", "For FD001 engines that lasted more than 200 cycles, how much did T30 rise before failure?",
    f"SELECT {RISE.format(c='hpc_outlet_temp')} FROM cyc WHERE dataset = 'FD001' AND engine_id IN (SELECT engine_id FROM eng WHERE life > 200)")

# ---------------------------------------------------------------- 10. comparisons / derived
c = "comparison"
add("mttf_diff_fd003_fd001", c, "hard", "How many more cycles do FD003 engines last on average compared to FD001?",
    "SELECT AVG(IFF(dataset = 'FD003', life, NULL)) - AVG(IFF(dataset = 'FD001', life, NULL)) FROM eng")
add("mttf_diff_fault", c, "hard", "What's the difference in MTTF between fan-and-HPC degradation and HPC-only degradation?",
    "SELECT AVG(IFF(faults = 'HPC and fan degradation', life, NULL)) - AVG(IFF(faults = 'HPC degradation', life, NULL)) FROM eng")
add("mttf_diff_cond", c, "hard", "By how many cycles does MTTF differ between six-condition and single-condition fleets (six minus single)?",
    "SELECT AVG(IFF(conditions = 'Six', life, NULL)) - AVG(IFF(conditions = 'Single', life, NULL)) FROM eng")
add("cycles_ratio_fd002_fd001", c, "hard", "How many times more cycles did FD002 fly than FD001?",
    "SELECT COUNT_IF(dataset = 'FD002') / COUNT_IF(dataset = 'FD001') FROM cyc")
add("engine_share_fd004", c, "hard", "What share of all engines belong to FD004?", "SELECT AVG(IFF(dataset = 'FD004', 1, 0)) FROM eng", percent_ok=True)
add("mttf_fd001_fd003", c, "medium", "Compare MTTF of FD001 and FD003.", "SELECT dataset, AVG(life) FROM eng WHERE dataset IN ('FD001', 'FD003') GROUP BY 1")
add("mttf_fd002_fd004", c, "medium", "Mean life in FD002 versus FD004?", "SELECT dataset, AVG(life) FROM eng WHERE dataset IN ('FD002', 'FD004') GROUP BY 1")
add("median_minus_mean", c, "hard", "Is the fleet's median life above or below its mean? Give median minus mean.", "SELECT MEDIAN(life) - AVG(life) FROM eng")
add("t30_rise_diff_fd3_fd1", c, "hard", "How much larger is FD003's T30 rise to failure than FD001's?",
    f"SELECT (SELECT {RISE.format(c='hpc_outlet_temp')} FROM cyc WHERE dataset = 'FD003') - (SELECT {RISE.format(c='hpc_outlet_temp')} FROM cyc WHERE dataset = 'FD001')")
add("cycle_share_by_dataset", c, "hard", "What percentage of all operating cycles came from each dataset?",
    "SELECT dataset, COUNT(*) / SUM(COUNT(*)) OVER () FROM cyc GROUP BY 1", percent_ok=True)
add("crit_vs_healthy_t50_fd001", c, "hard", "In FD001, how much hotter is T50 in the critical window than in the healthy stage?",
    "SELECT AVG(IFF(rul <= 15, lpt_outlet_temp, NULL)) - AVG(IFF(rul > 50, lpt_outlet_temp, NULL)) FROM cyc WHERE dataset = 'FD001'")
add("datasets_mttf_over_200", c, "hard", "Which datasets have an MTTF above 200 cycles?",
    "SELECT dataset FROM eng GROUP BY 1 HAVING AVG(life) > 200")
add("fault_count_and_cycles", c, "hard", "For each fault mode, show the number of engines and the total cycles flown.",
    "SELECT e.faults, e.n, c.n FROM (SELECT faults, COUNT(*) n FROM eng GROUP BY 1) e JOIN (SELECT faults, COUNT(*) n FROM cyc GROUP BY 1) c USING (faults)")
add("longest_vs_fd004_mttf", c, "hard", "How many cycles longer did the longest-lived engine run than FD004's mean life?",
    "SELECT MAX(life) - AVG(IFF(dataset = 'FD004', life, NULL)) FROM eng")
add("mttf_ratio_cond", c, "hard", "What's the ratio of six-condition MTTF to single-condition MTTF?",
    "SELECT AVG(IFF(conditions = 'Six', life, NULL)) / AVG(IFF(conditions = 'Single', life, NULL)) FROM eng")


def plain(v):
    if isinstance(v, Decimal):
        return float(v)
    return v


def main(out_dir):
    ids = [c["id"] for c in CASES]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, dupes
    qs = [c["q"].lower() for c in CASES]
    assert len(set(qs)) == len(qs), "duplicate question text"
    print(f"{len(CASES)} cases")

    conn = snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USER'), private_key_file=os.getenv('SNOWFLAKE_PRIVATE_KEY_FILE'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'), warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
        database=os.getenv('SNOWFLAKE_DATABASE'), role=os.getenv('SNOWFLAKE_ROLE'))
    cur = conn.cursor()
    bad = []
    for case in CASES:
        try:
            cur.execute(TRUTH_CTE + case["truth"])
            rows = [[plain(v) for v in r] for r in cur.fetchall()]
            if not rows or any(v is None for r in rows for v in r):
                bad.append((case["id"], f"empty/null result {rows}"))
            case["expected"] = rows
        except Exception as e:
            bad.append((case["id"], str(e)[:200]))
    conn.close()
    for b in bad:
        print("BAD", *b)

    os.makedirs(out_dir, exist_ok=True)
    out = {
        "generated": date.today().isoformat(),
        "semantic_view": "PREDICTIVE_MAINTENANCE.SEMANTIC.FLEET_RELIABILITY",
        "source_repo": REPO,
        "scoring": "Same rules as 14_eval_cortex_analyst.py: numbers within 0.5% relative; "
                   "first_row=True means the first expected row must appear in the analyst's first row; "
                   "percent_ok=True accepts a share as a fraction or a percentage.",
        "truth_cte": TRUTH_CTE.strip(),
        "count": len(CASES),
        "cases": CASES,
    }
    path = os.path.join(out_dir, "cortex_analyst_questions_200.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote", path, "bad:", len(bad))


if __name__ == "__main__":
    main(sys.argv[1])
