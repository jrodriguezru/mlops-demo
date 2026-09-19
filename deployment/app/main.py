from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    values: list[float] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    predictions: list[float]


def _load_mean_target() -> float:
    default_path = "/app/model_artifacts/model.json"
    model_path = Path(os.getenv("MODEL_PATH", default_path))
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    return float(payload["mean_target"])


app = FastAPI(title="mlops-demo-inference", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        mean_target = _load_mean_target()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return PredictionResponse(predictions=[mean_target for _ in request.values])
