from __future__ import annotations

from pathlib import Path

from model.pipeline import (
    load_csv,
    mean_absolute_error,
    save_model,
    train_baseline_model,
    train_validation_split,
)


def run_training(dataset_path: str, target_column: str, output_model_path: str) -> float:
    features, targets = load_csv(dataset_path, target_column=target_column)
    _, train_y, val_x, val_y = train_validation_split(features, targets, ratio=0.8)
    model = train_baseline_model(train_y)
    metric = mean_absolute_error(val_y, model.predict(val_x))
    save_model(model, output_model_path)
    return metric


if __name__ == "__main__":
    run_training(
        dataset_path=str(Path("data") / "dataset.csv"),
        target_column="target",
        output_model_path=str(Path("artifacts") / "model.json"),
    )
