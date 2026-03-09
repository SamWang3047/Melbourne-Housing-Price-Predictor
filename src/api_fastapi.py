from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
HISTORY_PATH = ARTIFACTS_DIR / "latest_history.csv"

app = FastAPI(title="Melbourne Housing Price Predictor", version="1.0.0")


class PredictRequest(BaseModel):
    locality: str = Field(..., description="Suburb/locality name, e.g. ABBOTSFORD")
    year: int = Field(..., ge=2000, le=2100)
    quarter: Literal[1, 2, 3, 4]
    lag_1: float | None = Field(default=None, gt=0)
    lag_2: float | None = Field(default=None, gt=0)
    rolling_mean_2: float | None = Field(default=None, gt=0)


class PredictResponse(BaseModel):
    locality: str
    year: int
    quarter: int
    predicted_median_price: float
    used_defaults_from_history: bool


def _load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError("Model file not found. Run training first: python src/train.py")
    return joblib.load(MODEL_PATH)


def _load_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        return pd.DataFrame(columns=["locality", "year", "quarter", "median_price"])
    return pd.read_csv(HISTORY_PATH)


def _default_lags(locality: str, history: pd.DataFrame) -> tuple[float | None, float | None, float | None]:
    loc_history = history[history["locality"].str.upper() == locality.upper()].sort_values(["year", "quarter"])
    if loc_history.empty:
        return None, None, None

    latest_prices = loc_history["median_price"].tolist()
    lag_1 = latest_prices[-1] if len(latest_prices) >= 1 else None
    lag_2 = latest_prices[-2] if len(latest_prices) >= 2 else lag_1

    rolling = None
    if len(latest_prices) >= 2:
        rolling = float((latest_prices[-1] + latest_prices[-2]) / 2)
    elif len(latest_prices) == 1:
        rolling = float(latest_prices[-1])

    return lag_1, lag_2, rolling


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    model = _load_model()
    history = _load_history()

    d_lag_1, d_lag_2, d_roll = _default_lags(payload.locality, history)

    lag_1 = payload.lag_1 if payload.lag_1 is not None else d_lag_1
    lag_2 = payload.lag_2 if payload.lag_2 is not None else d_lag_2
    rolling_mean_2 = payload.rolling_mean_2 if payload.rolling_mean_2 is not None else d_roll

    if lag_1 is None or lag_2 is None or rolling_mean_2 is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Not enough history for this locality. Provide lag_1, lag_2, rolling_mean_2 explicitly "
                "or train with richer data."
            ),
        )

    row = pd.DataFrame(
        [
            {
                "locality": payload.locality.upper(),
                "year": payload.year,
                "quarter": payload.quarter,
                "lag_1": lag_1,
                "lag_2": lag_2,
                "rolling_mean_2": rolling_mean_2,
            }
        ]
    )

    pred = float(model.predict(row)[0])

    return PredictResponse(
        locality=payload.locality.upper(),
        year=payload.year,
        quarter=payload.quarter,
        predicted_median_price=round(pred, 2),
        used_defaults_from_history=(payload.lag_1 is None or payload.lag_2 is None or payload.rolling_mean_2 is None),
    )
