from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Resolve project paths relative to this deployment entrypoint.
ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
HISTORY_PATH = ARTIFACTS_DIR / "latest_history.csv"

app = FastAPI(
    title="Melbourne Housing Price Predictor",
    version="1.0.0",
    description="Predict quarterly median house prices for Melbourne localities.",
)


class PredictRequest(BaseModel):
    locality: str = Field(..., description="Locality name such as ABBOTSFORD")
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
    used_history_defaults: bool


def load_model():
    # Fail early with a clear message if the training artifacts are missing.
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. Run the notebook first to generate artifacts."
        )
    return joblib.load(MODEL_PATH)


def load_history() -> pd.DataFrame:
    # Load the latest locality history used to infer lag features for API requests.
    if not HISTORY_PATH.exists():
        return pd.DataFrame(columns=["locality", "year", "quarter", "median_price"])
    return pd.read_csv(HISTORY_PATH)


def default_lags(locality: str, history: pd.DataFrame) -> tuple[float | None, float | None, float | None]:
    # Recover the most recent lag values for a locality from exported artifacts.
    loc_history = history[history["locality"].str.upper() == locality.upper()].sort_values(["year", "quarter"])
    if loc_history.empty:
        return None, None, None

    latest_prices = loc_history["median_price"].tolist()
    lag_1 = latest_prices[-1] if len(latest_prices) >= 1 else None
    lag_2 = latest_prices[-2] if len(latest_prices) >= 2 else lag_1

    if len(latest_prices) >= 2:
        rolling_mean_2 = float((latest_prices[-1] + latest_prices[-2]) / 2)
    elif len(latest_prices) == 1:
        rolling_mean_2 = float(latest_prices[-1])
    else:
        rolling_mean_2 = None

    return lag_1, lag_2, rolling_mean_2


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    model = load_model()
    history = load_history()

    d_lag_1, d_lag_2, d_rolling = default_lags(request.locality, history)
    lag_1 = request.lag_1 if request.lag_1 is not None else d_lag_1
    lag_2 = request.lag_2 if request.lag_2 is not None else d_lag_2
    rolling_mean_2 = request.rolling_mean_2 if request.rolling_mean_2 is not None else d_rolling

    if lag_1 is None or lag_2 is None or rolling_mean_2 is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Not enough history is available for this locality. "
                "Provide lag_1, lag_2, and rolling_mean_2 explicitly."
            ),
        )

    features = pd.DataFrame(
        [
            {
                "locality": request.locality.upper(),
                "year": request.year,
                "quarter": request.quarter,
                "lag_1": lag_1,
                "lag_2": lag_2,
                "rolling_mean_2": rolling_mean_2,
            }
        ]
    )

    predicted_price = float(model.predict(features)[0])
    return PredictResponse(
        locality=request.locality.upper(),
        year=request.year,
        quarter=request.quarter,
        predicted_median_price=round(predicted_price, 2),
        used_history_defaults=(
            request.lag_1 is None or request.lag_2 is None or request.rolling_mean_2 is None
        ),
    )
