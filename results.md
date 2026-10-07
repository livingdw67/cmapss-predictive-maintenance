# NASA C-MAPSS: Tandem Binary Classifiers for Predictive Maintenance

Dual-model approach using XGBoost classifiers registered in the Snowflake Model Registry:
- **Model A (Early Warning):** Predicts if engine failure is within 50 cycles
- **Model B (Critical Action):** Predicts if engine failure is within 15 cycles

> **v2 correction (October 2026):** Engine numbers restart at 1 in each of the four C-MAPSS training files (FD001–FD004). Earlier versions of this pipeline used the raw number as the engine ID, which merged up to four different engines under one ID (260 "engines" instead of 709) and gave 55% of rows the wrong failure point. Engine IDs now include the dataset (for example `FD002_001`), thresholds are tuned on a separate validation set, and all results below were regenerated. The numbers come from running `08_build_register_classifiers.ipynb` against the same features produced locally by `evaluate_local.py`.

---

## Dataset Summary

| Metric | Value |
|--------|-------|
| Total records | 160,359 |
| Total engines | 709 (FD001: 100, FD002: 260, FD003: 100, FD004: 249) |
| Features | 8 (sensor readings + rolling stats) |
| Split | Engine-level 60/20/20, stratified by lifespan; no engine in two sets |

**Features:** `HPC_OUTLET_TEMP`, `PHYSICAL_CORE_SPEED`, `BYPASS_RATIO`, `HPC_TEMP_MA_5`, `CORE_SPEED_MA_5`, `BYPASS_RATIO_MA_5`, `HPC_TEMP_STD_5`, `CORE_SPEED_STD_5`

| Set | Records | Engines | Used for |
|-----|---------|---------|----------|
| Train | 95,538 | 425 | Fitting models |
| Validation | 32,438 | 142 | Choosing thresholds and resampling strategy |
| Test | 32,383 | 142 | Final scores only |

### Target Distributions

| Target | Train Pos% | Test Pos% | Imbalance Ratio |
|--------|-----------|----------|-----------------|
| EARLY_WARNING | 22.7% | 22.4% | 1:3 |
| CRITICAL_ACTION | 7.1% | 7.0% | 1:13 |

---

## V1 — Baseline Models (threshold = 0.50)

XGBoost with `scale_pos_weight` (Model A: 3.4, Model B: 13.0). Test engines:

| Model | Recall | Precision | F1 |
|-------|--------|-----------|----|
| A: Early Warning (≤ 50 cycles) | 0.88 | 0.74 | 0.81 |
| B: Critical Action (≤ 15 cycles) | 0.94 | 0.58 | 0.71 |

### Tandem Classifier Decision Matrix

| Prediction | Action | Test records |
|-----------|--------|-------------:|
| A=0, B=0 | Normal operations | 23,794 |
| A=1, B=0 | Schedule maintenance | 4,893 |
| A=1, B=1 | Immediate action required | 3,696 |
| A=0, B=1 | Anomaly — investigate | 0 |

---

## V1.1 — F2-Optimized Thresholds

F2 weighs recall 4x more than precision, which suits maintenance, where a missed failure costs far more than a false alarm. Thresholds are chosen on validation engines and applied unchanged to test engines.

| Model | Threshold | Test F2 | Test Recall | Test Precision |
|-------|-----------|---------|-------------|----------------|
| **A: Early Warning** | 0.35 | **0.857** | **0.917** | 0.679 |
| **B: Critical Action** | 0.54 | **0.837** | **0.908** | 0.636 |

(300 trees, depth 7, learning rate 0.08, `scale_pos_weight`.)

---

## Experiment: Resampling Strategy Comparison

Each strategy's threshold is tuned on validation engines; ★ marks the best validation F2.

### Model A: Early Warning (RUL ≤ 50)

| Strategy | Threshold | Val F2 | Test F2 | Recall | Precision |
|----------|-----------|--------|---------|--------|-----------|
| `scale_pos_weight` | 0.35 | 0.8538 | 0.8570 | 0.9171 | 0.6789 |
| SMOTE | 0.35 | 0.8540 | 0.8563 | 0.9189 | 0.6730 |
| Undersampling 6x ★ | 0.17 | 0.8549 | 0.8560 | 0.9089 | 0.6945 |

### Model B: Critical Action (RUL ≤ 15)

| Strategy | Threshold | Val F2 | Test F2 | Recall | Precision |
|----------|-----------|--------|---------|--------|-----------|
| `scale_pos_weight` | 0.54 | 0.8389 | 0.8366 | 0.9080 | 0.6363 |
| SMOTE | 0.65 | 0.8316 | 0.8266 | 0.8926 | 0.6379 |
| Undersampling 6x ★ | 0.20 | 0.8413 | 0.8383 | 0.9371 | 0.5898 |

### Finding

All three strategies land within about 0.01 F2 of each other, and the "winner" changes with the random seed used for SMOTE and undersampling. With roughly 1:3 and 1:13 imbalance, threshold tuning does the real work; resampling adds complexity without a reliable gain. `scale_pos_weight` is the simplest choice and is fully reproducible.

Note: for Model A, positives are already ~23% of rows, so "6x undersampling" keeps every row and is equivalent to training without class weights.

---

## Model Registry

Models are versioned in `PREDICTIVE_MAINTENANCE.ML_MODELS` with computed metrics attached.

| Model | Versions | Purpose |
|-------|----------|---------|
| TURBOFAN_EARLY_WARNING | v1, v1_1, v2_0 | Failure within 50 cycles |
| TURBOFAN_CRITICAL_ACTION | v1, v1_1, v2_0 | Failure within 15 cycles |

## Next Steps

* **Normalize by operating condition and use more sensors:** done in `10_sensor_graph_gnn.py`. With 14 condition-normalized sensors and 30 cycles of history, XGBoost reaches test F2 0.931 (Early Warning) and 0.929 (Critical Action). A graph neural network on the same inputs ties it; see the README.
* **Promote the improved XGBoost** to the Snowflake feature store and registry (per-condition normalization as a SQL layer).
* **Time-to-failure modeling:** done. See `09_weibull_reliability_analysis.ipynb` for the Weibull fleet reliability analysis.
