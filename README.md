# Definition-Driven Predictive Maintenance Pipeline

An end-to-end data engineering and machine learning feature store built on Snowflake. This project demonstrates how to process raw aerospace telemetry data into mathematically transformed, ML-ready features for predictive maintenance.

## Business Objective
Unexpected hardware failure in aviation and heavy industry carries catastrophic costs. This pipeline ingests run-to-failure sensor data from the NASA CMAPSS (Turbofan Engine Degradation) dataset and builds a foundation for **Condition-Based Maintenance**.

By calculating rolling window metrics and defining a Piecewise Remaining Useful Life (RUL) target variable, this architecture allows machine learning models to accurately predict failures before they happen, minimizing unplanned downtime and optimizing supply chain logistics.

## The 5-Schema Snowflake Architecture
This project implements a rigorous, definition-driven data platform decoupled into five distinct layers:

1. **RAW:** Ingestion of flat, space-delimited text logs directly from the internal stage.
2. **STAGING:** Type-casting, standardizing arbitrary column names to physical engine components (e.g., High-Pressure Compressor Temperature), and building a dataset-aware engine key (`FD002_001`), since engine numbers restart in each of the four source files.
3. **CORE:** Relational modeling (Star Schema) splitting static entities (`DIM_ENGINE`) from time-series events (`FCT_TELEMETRY`).
4. **MARTS:** Application of business logic to calculate the target variable: a piecewise-capped Remaining Useful Life (RUL) to prevent overfitting on healthy engines.
5. **FEATURE STORE:** Push-down compute utilizing SQL Window Functions to generate 5-cycle rolling moving averages and standard deviations, dramatically increasing the signal-to-noise ratio for downstream ML models.

## Machine Learning: Tandem Binary Classifiers
To operationalize the feature store, this project includes a complete modeling workflow utilizing Snowpark and XGBoost:

* **Dual-Model Strategy:** Deploys a tandem approach predicting both an *Early Warning* (≤50 cycles) and *Critical Action* (≤15 cycles) state to drive specific maintenance protocols.
* **F2-Optimized Thresholds:** Prioritizes recall to minimize costly missed failures, leveraging the F-beta score (beta=2) to dynamically tune decision thresholds.
* **Honest Evaluation:** Engines are split 60/20/20 into train, validation, and test sets. Thresholds and strategies are chosen on validation engines; test engines are scored once.
* **Class Imbalance Experimentation:** Compares manual SMOTE, 6x undersampling, and `scale_pos_weight`. All three land within about 0.01 F2, so the simplest option (`scale_pos_weight`) is preferred.
* **Model Registry:** Demonstrates MLOps best practices by versioning and logging champion models directly into Snowflake's native ML Registry.

### Results (held-out test engines)

| Model | Recall | Precision | F2 |
|-------|--------|-----------|----|
| Early Warning (≤ 50 cycles) | 0.917 | 0.679 | 0.857 |
| Critical Action (≤ 15 cycles) | 0.908 | 0.636 | 0.837 |

**[Full experiment results](results.md)**

> **v2 correction:** an earlier version merged engines from different source files that shared an ID number, which mislabeled 55% of rows. That is fixed, and all results were regenerated. See [results.md](results.md) for details.

## Run It

```bash
pip install -r requirements.txt
python 01_fetch_data.py       # Download NASA C-MAPSS into data/raw
python evaluate_local.py      # Reproduce the feature store and models locally (no Snowflake needed)
```

To run the full Snowflake pipeline, add Snowflake credentials to a `.env` file and run scripts `02` through `07` in order, then the `08` notebook in Snowflake.

## Technology Stack
* **Data Platform:** Snowflake (SQL, Snowpark ML)
* **Orchestration & Extraction:** Python, `snowflake-connector-python`
* **Data Science & ML:** `xgboost`, `pandas`, `scikit-learn`, `matplotlib`
* **Environment Management:** `python-dotenv`

## Project Structure
```text
├── .env                        # Local Snowflake credentials (git-ignored)
├── .gitignore                  # Security and environment exclusions
├── README.md                   # Project documentation
├── 01_fetch_data.py            # Retrieves the raw NASA CMAPSS dataset
├── 02_upload_to_snowflake.py   # Securely PUTs local data to internal Snowflake stage
├── 03_load_raw_table.py        # Executes COPY INTO commands for the RAW layer
├── 04_build_staging.py         # Type casting and column standardization
├── 05_build_core.py            # Star schema generation (Dim/Fact tables)
├── 06_build_marts.py           # RUL target variable calculation
├── 07_build_feature_store.py   # Rolling window feature engineering via Window Functions
├── 08_build_register_classifiers.ipynb    # XGBoost tandem classifiers & model registration
├── evaluate_local.py           # Snowflake-free reproduction of the pipeline and models
├── results.md                  # Experiment results
└── requirements.txt            # Python dependencies
