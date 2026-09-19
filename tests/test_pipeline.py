from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from model.pipeline import (
    load_csv,
    mean_absolute_error,
    save_model,
    train_baseline_model,
    train_validation_split,
)


class PipelineTests(unittest.TestCase):
    def test_load_train_and_save_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset.csv"
            model_path = Path(tmp_dir) / "model.json"

            with dataset_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["f1", "f2", "target"])
                writer.writeheader()
                writer.writerow({"f1": 1, "f2": 2, "target": 2})
                writer.writerow({"f1": 2, "f2": 3, "target": 3})
                writer.writerow({"f1": 3, "f2": 4, "target": 4})

            features, targets = load_csv(dataset_path, target_column="target")
            train_x, train_y, val_x, val_y = train_validation_split(features, targets, 0.67)
            self.assertEqual(len(train_x), 2)
            self.assertEqual(len(val_x), 1)

            model = train_baseline_model(train_y)
            mae = mean_absolute_error(val_y, model.predict(val_x))
            self.assertGreaterEqual(mae, 0.0)

            saved = save_model(model, model_path)
            payload = json.loads(saved.read_text(encoding="utf-8"))
            self.assertIn("mean_target", payload)


if __name__ == "__main__":
    unittest.main()
