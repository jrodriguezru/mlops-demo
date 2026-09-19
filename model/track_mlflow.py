from __future__ import annotations

from pathlib import Path


def log_run(
    *,
    experiment_name: str,
    run_name: str,
    tracking_uri: str,
    metric_name: str,
    metric_value: float,
    artifact_path: str | Path,
) -> None:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_metric(metric_name, metric_value)
        mlflow.log_artifact(str(artifact_path))
