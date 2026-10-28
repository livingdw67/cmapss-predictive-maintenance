# C-MAPSS tandem classifiers as a Kedro pipeline

The repo's Databricks job (`../databricks/src/02_transform.py` → `03_train_register.py` →
`04_batch_score.py`) rebuilt as a Kedro 1.7 project that runs locally on pandas, with
[kedro-mlflow](https://github.com/Galileo-Galilei/kedro-mlflow) for tracking and the model registry.
Same features, split, models, thresholds and decision matrix. The difference is structure: each step is
a pure function, the data flow is declared in the catalog, and every setting is in YAML parameters.

## What runs

```
data_processing   raw NASA files ─► staged telemetry ─► model_input (piecewise RUL, 5-cycle rolling features)
                                                    └─► latest_test_features (last cycle of each test engine)
training          model_input ─► split_engines (60/20/20 by engine, stratified by lifespan)
                     ├─► early_warning.train_classifier    (fails within 50 cycles)
                     └─► critical_action.train_classifier  (fails within 15 cycles)
scoring           score_fleet (inference) ─► evaluate_alerts against NASA's true RUL
```

- **One classifier template, two namespaces.** `train_classifier` is written once and instantiated as
  `early_warning` and `critical_action`. Kedro prefixes the outputs and `params:horizon` with the
  namespace, so each copy reads its own horizon and writes its own model, metrics and comparison table.
  Inside each, three imbalance strategies (class weighting, SMOTE, 6x undersampling) get an F2 threshold
  tuned on validation engines; the best validation F2 is kept, and test engines are only reported.
- **Dataset factories.** One catalog entry, `"{split}_{dataset}"`, covers all eight raw files, and
  `"{target}.model"` / `"{target}.metrics"` cover both namespaces.
- **kedro-mlflow.** `kedro run` is one MLflow run: parameters are logged automatically (flattened, e.g.
  `xgboost.max_depth`). Metrics go through `MlflowMetricsHistoryDataset` and CSV reports through
  `MlflowArtifactDataset`. The default pipeline is wrapped with `pipeline_ml_factory`, so at the end of
  the run the inference pipeline and both trained classifiers are logged as **one MLflow model** and
  registered as `cmapss_fleet_scorer`. It takes `latest_test_features` and returns `risk_scores`:

  ```python
  mlflow.pyfunc.load_model("models:/cmapss_fleet_scorer/latest").predict(latest_test_features)
  ```

## Run it

Raw data comes from the repo's `../01_fetch_data.py` (into `../data/raw`, git-ignored).

```
cd kedro
python -m venv .venv
.venv\Scripts\python -m pip install -e .
.venv\Scripts\kedro run                              # everything, ~15 s
.venv\Scripts\kedro run --pipeline data_processing   # or one stage
.venv\Scripts\kedro mlflow ui                        # http://127.0.0.1:5000
.venv\Scripts\python -m pytest -q
```

Tracking is a local SQLite file (`mlflow.db`, git-ignored); the registry needs a database backend.
Point `conf/base/mlflow.yml` (or a `conf/local` override) at a server to share runs.

## Results

Selected model per horizon. Held-out rows are from the 20% of training engines never used to fit or
tune. The engine-level figures score each of NASA's 707 test engines once, at its last observed
cycle, against its true remaining life.

| Model | Strategy, threshold | Held-out rows: recall / precision / F2 | NASA test engines: recall (95% CI) | precision (95% CI) |
|---|---|---|---|---|
| Early Warning (≤ 50 cycles) | undersample, 0.17 | 0.900 / 0.694 / 0.850 | 0.930 (0.890–0.957), 214 of 230 | 0.723 (0.669–0.771) |
| Critical Action (≤ 15 cycles) | undersample, 0.27 | 0.915 / 0.616 / 0.834 | 0.912 (0.841–0.953), 93 of 102 | 0.721 (0.638–0.791) |

The Databricks run of the same job gave 0.952 / 0.720 (Early Warning) and 0.912 / 0.689 (Critical
Action). Both sit inside the intervals above. The small gaps are most likely the XGBoost version
(3.2 here, 2.1.4 there), which also moved the Critical Action threshold from 0.24 to 0.27.

**Why the intervals.** The 707 test engines are a sample, and recall for Critical Action rests on only
the 102 engines that were truly within 15 cycles of failure, so one more or fewer catch moves it by
a full point.
**What they mean.** Engines like these get a Critical Action alert in time somewhere between about
84% and 95% of the time. "91%" is the best single estimate, not a guarantee.
**Method and assumptions.** 95% Wilson score intervals: more accurate than p ± 1.96·SE when a
proportion is near 1 and the count is around 100. They assume each engine is an independent trial
and that NASA's test engines represent the fleet being scored. They cover sampling of engines only,
not the variation from one training seed or split to another.

## Layout

| Path | What |
|---|---|
| `conf/base/catalog.yml` | Every dataset: raw files, parquet layers, pickled models, MLflow metrics and artifacts |
| `conf/base/parameters_*.yml` | RUL cap, split, XGBoost settings, threshold grid, horizons |
| `conf/base/mlflow.yml` | kedro-mlflow: tracking URI, experiment `cmapss-tandem-classifiers` |
| `src/cmapss_kedro/pipelines/` | `data_processing`, `training`, `scoring`: `nodes.py` (functions) and `pipeline.py` (wiring) |
| `src/cmapss_kedro/pipeline_registry.py` | Pipelines, and the `pipeline_ml_factory` wrapper |
| `tests/` | Node tests (rolling windows stay within an engine, no engine in two splits, decision matrix, Wilson interval) |
