"""Project settings. kedro-mlflow registers its hook through its entry point, so none are listed here."""

CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
}
