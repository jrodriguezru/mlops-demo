# model/

This directory contains the model-side Python code:

- `pipeline.py`:
  - loads tabular CSV data
  - prepares train/validation split
  - trains a lightweight baseline regressor
  - evaluates MAE
  - saves a model artifact
- `track_mlflow.py`:
  - optional MLflow logging utility for metrics/artifacts in a self-hosted MLflow instance

The implementation is intentionally lightweight so the infrastructure flow remains the main focus.
