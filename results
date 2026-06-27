# NASA C-MAPSS: Tandem Binary Classifiers for Predictive Maintenance

Dual-model approach using XGBoost classifiers registered in the Snowflake Model Registry:
- **Model A (Early Warning):** Predicts if engine failure is within 50 cycles
- **Model B (Critical Action):** Predicts if engine failure is within 15 cycles

---

## Dataset Summary

| Metric | Value |
|--------|-------|
| Total records | 160,359 |
| Total engines | 260 |
| Features | 8 (sensor readings + rolling stats) |
| Train/Test split | Engine-stratified (no leakage) |

**Features:** `HPC_OUTLET_TEMP`, `PHYSICAL_CORE_SPEED`, `BYPASS_RATIO`, `HPC_TEMP_MA_5`, `CORE_SPEED_MA_5`, `BYPASS_RATIO_MA_5`, `HPC_TEMP_STD_5`, `CORE_SPEED_STD_5`

### Target Distributions

| Target | Train Pos% | Test Pos% | Imbalance Ratio |
|--------|-----------|----------|-----------------|
| EARLY_WARNING | 10.9% | 10.8% | 1:8 |
| CRITICAL_ACTION | 2.8% | 2.7% | 1:34 |

### Train/Test Split

| Set | Records | Engines |
|-----|---------|---------|
| Train | 125,782 | 208 |
| Test | 34,577 | 52 |

Engine overlap: None (strict engine-level split)

---

## V1 — Baseline Models (threshold=0.50)

XGBoost with `scale_pos_weight` to handle imbalance:
- Model A: `scale_pos_weight = 8.2`
- Model B: `scale_pos_weight = 34.3`

### Model A: Early Warning (Failure within 50 cycles)

```
              precision    recall  f1-score   support

      Normal       0.98      0.90      0.94     30843
     Warning       0.52      0.87      0.65      3734

    accuracy                           0.90     34577
   macro avg       0.75      0.89      0.80     34577
weighted avg       0.93      0.90      0.91     34577
```

### Model B: Critical Action (Failure within 15 cycles)

```
              precision    recall  f1-score   support

      Normal       1.00      0.96      0.98     33648
    Critical       0.39      0.87      0.54       929

    accuracy                           0.96     34577
   macro avg       0.69      0.92      0.76     34577
weighted avg       0.98      0.96      0.97     34577
```

### Tandem Classifier Decision Matrix

| Prediction | Action |
|-----------|--------|
| A=0, B=0 | Normal operations |
| A=1, B=0 | Schedule maintenance |
| A=1, B=1 | Immediate action required |
| A=0, B=1 | Anomaly — investigate |

**Test Set Decision Distribution:**
| Decision | Count |
|----------|-------|
| Normal (A=0, B=0) | 28,263 |
| Schedule maintenance (A=1, B=0) | 4,212 |
| Immediate action (A=1, B=1) | 2,092 |
| Anomaly (A=0, B=1) | 10 |

---

## V1.1 — F2-Optimized Threshold Tuning

F2 score weighs recall 4x more than precision — ideal for maintenance where missing a failure is far costlier than a false alarm.

### Optimal Thresholds Found

| Model | Threshold | F2 |
|-------|-----------|------|
| Model A (Early Warning) | 0.40 | 0.7741 |
| Model B (Critical Action) | 0.47 | 0.7217 |

### V1 vs V1.1 Comparison

**Model A (Early Warning):**

| Metric | V1 (t=0.50) | V1.1 (t=0.40) | Change |
|--------|-------------|---------------|--------|
| F2 | 0.7677 | 0.7741 | +0.0064 |
| Recall | 0.8733 | 0.8913 | +0.0180 |
| Precision | 0.5173 | 0.5074 | -0.0099 |

**Model B (Critical Action):**

| Metric | V1 (t=0.50) | V1.1 (t=0.47) | Change |
|--------|-------------|---------------|--------|
| F2 | 0.6970 | 0.7217 | +0.0247 |
| Recall | 0.8730 | 0.8482 | -0.0248 |
| Precision | 0.3858 | 0.4521 | +0.0663 |

---

## Experiment: Resampling Strategy Comparison

Comparing three class imbalance strategies (F2-optimized):
1. **Baseline** — `scale_pos_weight` only
2. **SMOTE** — Oversample minority with synthetic examples
3. **Undersampling** — Random majority downsample at 6x minority

### Model A: Early Warning (RUL ≤ 50)

| Strategy | Threshold | F2 | Recall | Precision |
|----------|-----------|------|--------|-----------|
| **Baseline** | 0.13 | **0.7744** | 0.8698 | 0.5382 |
| SMOTE | 0.41 | 0.7610 | 0.8873 | 0.4849 |
| Undersampling | 0.16 | 0.7722 | 0.8733 | 0.5278 |

### Model B: Critical Action (RUL ≤ 15)

| Strategy | Threshold | F2 | Recall | Precision |
|----------|-----------|------|--------|-----------|
| **Baseline** | 0.09 | **0.7260** | 0.8493 | 0.4593 |
| SMOTE | 0.64 | 0.7023 | 0.8192 | 0.4471 |
| Undersampling | 0.35 | 0.7203 | 0.8622 | 0.4344 |

**Winner:** Baseline (`scale_pos_weight`) for both models.

---

## Experiment: Undersampling Ratio Sweep (CRITICAL_ACTION)

Testing whether 1:6 is truly optimal by sweeping majority:minority ratios from 1:1 to 1:10.

| Ratio | Samples | Threshold | F2 | Recall | Precision | AUC |
|-------|---------|-----------|------|--------|-----------|------|
| 1:1 | 7,130 | 0.84 | 0.6842 | 0.8611 | 0.3756 | 0.9808 |
| 1:3 | 14,260 | 0.64 | 0.7182 | 0.8364 | 0.4589 | 0.9848 |
| 1:6 | 24,955 | 0.35 | 0.7203 | 0.8622 | 0.4344 | 0.9852 |
| 1:8 | 32,085 | 0.30 | 0.7227 | 0.8471 | 0.4552 | 0.9858 |
| **1:10** | **39,215** | **0.31** | **0.7317** | 0.8385 | **0.4848** | **0.9863** |

### Key Findings

- **Best F2:** 1:10 ratio (0.7317) — outperforms 1:6 (0.7203) by +1.6%
- **Best Recall:** 1:6 ratio (0.8622) — aggressive undersampling favors recall
- **Trend:** F2, precision, and AUC improve monotonically with more majority samples
- **Trade-off:** More data improves overall discrimination (AUC) and precision, but reduces recall slightly

### Recommendation

If **F2 is the optimization target**, 1:10 is superior. However, if the operational priority is **never missing a critical failure** (maximizing recall), 1:6 remains defensible — it catches 86.2% of critical events vs 83.9% at 1:10.

---

## Model Registry

All models are registered in `PREDICTIVE_MAINTENANCE.ML_MODELS` via the Snowflake Model Registry with full metrics attached for comparison and governance.

| Model | Versions | Purpose |
|-------|----------|---------|
| TURBOFAN_EARLY_WARNING | v1, v1.1, ratio sweep | Failure within 50 cycles |
| TURBOFAN_CRITICAL_ACTION | v1, v1.1, ratio sweep (5 versions) | Failure within 15 cycles |
