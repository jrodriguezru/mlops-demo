from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def load_csv(path: str | Path, target_column: str) -> tuple[list[float], list[float]]:
    features: list[float] = []
    targets: list[float] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or target_column not in reader.fieldnames:
            raise ValueError(f"Missing target column '{target_column}' in dataset")
        for row in reader:
            targets.append(float(row[target_column]))
            feature_values = [
                float(value) for key, value in row.items() if key != target_column
            ]
            features.append(sum(feature_values) / len(feature_values))
    return features, targets


def train_validation_split(
    features: list[float], targets: list[float], ratio: float = 0.8
) -> tuple[list[float], list[float], list[float], list[float]]:
    if not 0 < ratio < 1:
        raise ValueError("ratio must be between 0 and 1")
    if len(features) != len(targets):
        raise ValueError("features and targets must have the same length")
    split_index = max(1, int(len(features) * ratio))
    return (
        features[:split_index],
        targets[:split_index],
        features[split_index:],
        targets[split_index:],
    )


@dataclass
class MeanRegressor:
    mean_target: float

    def predict(self, values: Iterable[float]) -> list[float]:
        return [self.mean_target for _ in values]


def train_baseline_model(targets: list[float]) -> MeanRegressor:
    if not targets:
        raise ValueError("targets cannot be empty")
    return MeanRegressor(mean_target=sum(targets) / len(targets))


def mean_absolute_error(actual: list[float], predicted: list[float]) -> float:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    if not actual:
        return 0.0
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def save_model(model: MeanRegressor, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"mean_target": model.mean_target}), encoding="utf-8")
    return output
