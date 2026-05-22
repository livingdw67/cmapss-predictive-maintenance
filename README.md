# Definition-Driven Predictive Maintenance Pipeline

An end-to-end data engineering and machine learning feature store built on Snowflake. This project demonstrates how to process raw aerospace telemetry data into mathematically transformed, ML-ready features for predictive maintenance.

## Business Objective
Unexpected hardware failure in aviation and heavy industry carries catastrophic costs. This pipeline ingests run-to-failure sensor data from the NASA CMAPSS (Turbofan Engine Degradation) dataset and builds a foundation for **Condition-Based Maintenance**. 

By calculating rolling window metrics and defining a Piecewise Remaining Useful Life (RUL) target variable, this architecture allows machine learning models (via MLflow or Snowflake Cortex) to accurately predict failures before they happen, minimizing unplanned downtime and optimizing supply chain logistics.

## The 5-Schema Snowflake Architecture
This project implements a rigorous, definition-driven data platform decoupled into five distinct layers:

1. **RAW:** Ingestion of flat, space-delimited text logs directly from the internal stage.
2. **STAGING:** Type-casting and standardizing arbitrary column names to physical engine components (e.g., High-Pressure Compressor Temperature).
3. **CORE:** Relational modeling (Star Schema) splitting static entities (`DIM_ENGINE`) from time-series events (`FCT_TELEMETRY`).
4. **MARTS:** Application of business logic to calculate the target variable: a piecewise-capped Remaining Useful Life (RUL) to prevent overfitting on healthy engines.
5. **FEATURE STORE:** Push-down compute utilizing SQL Window Functions to generate 5-cycle rolling moving averages and standard deviations, dramatically increasing the signal-to-noise ratio for downstream ML models.

## Technology Stack
* **Data Platform:** Snowflake (SQL)
* **Orchestration & Extraction:** Python, `snowflake-connector-python`
* **Data Science & EDA:** `pandas`, `scikit-learn`, `matplotlib`, `seaborn`
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
└── 08_feature_importance.py    # Generates Pearson Correlation and Random Forest rankings
