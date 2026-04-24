from __future__ import annotations

import io
from pathlib import Path
from typing import Literal

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

matplotlib.use("Agg")

# Resolve project paths relative to this deployment entrypoint.
ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
HISTORY_PATH = ARTIFACTS_DIR / "latest_history.csv"
CLEAN_DATA_PATH = ARTIFACTS_DIR / "clean_long_data.csv"
FRONTEND_PATH = ROOT / "frontend" / "index.html"
GUIDE_PATH = ROOT / "frontend" / "guide.html"

app = FastAPI(
    title="Melbourne Housing Price Predictor",
    version="1.0.0",
    description="Predict quarterly median house prices for Melbourne localities.",
)


class PredictRequest(BaseModel):
    # Optional lag inputs let callers override the default history-based features.
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


class ForecastResponse(BaseModel):
    # The frontend uses this payload to fill summary cards and load the chart image.
    locality: str
    prediction_year: int
    prediction_quarter: int
    predicted_median_price: float
    latest_actual_price: float
    chart_url: str


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


def load_clean_data() -> pd.DataFrame:
    # Load the full cleaned historical dataset for charting and locality search.
    if not CLEAN_DATA_PATH.exists():
        raise RuntimeError(
            f"Clean dataset artifact not found at {CLEAN_DATA_PATH}. Run the notebook first to generate artifacts."
        )
    df = pd.read_csv(CLEAN_DATA_PATH)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


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


def next_quarter(year: int, quarter: int) -> tuple[int, int]:
    # Move one step forward in quarterly time.
    if quarter == 4:
        return year + 1, 1
    return year, quarter + 1


def get_locality_history(locality: str) -> pd.DataFrame:
    # Return the full historical series for one locality.
    clean_df = load_clean_data()
    locality_history = clean_df[clean_df["locality"].str.upper() == locality.upper()].copy()
    if locality_history.empty:
        raise HTTPException(status_code=404, detail=f"Locality '{locality}' was not found.")
    return locality_history.sort_values(["year", "quarter"])


def build_forecast(locality: str) -> ForecastResponse:
    # Predict the next quarter based on the most recent available locality history.
    locality_history = get_locality_history(locality)
    latest_row = locality_history.iloc[-1]
    prediction_year, prediction_quarter = next_quarter(int(latest_row["year"]), int(latest_row["quarter"]))

    history = load_history()
    d_lag_1, d_lag_2, d_rolling = default_lags(locality, history)
    if d_lag_1 is None or d_lag_2 is None or d_rolling is None:
        raise HTTPException(
            status_code=400,
            detail="Not enough history is available for this locality to build an automatic forecast.",
        )

    prediction = predict(
        PredictRequest(
            locality=locality.upper(),
            year=prediction_year,
            quarter=prediction_quarter,
            lag_1=d_lag_1,
            lag_2=d_lag_2,
            rolling_mean_2=d_rolling,
        )
    )

    return ForecastResponse(
        locality=locality.upper(),
        prediction_year=prediction_year,
        prediction_quarter=prediction_quarter,
        predicted_median_price=prediction.predicted_median_price,
        latest_actual_price=float(latest_row["median_price"]),
        chart_url=f"/forecast-chart?locality={locality.upper()}",
    )


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    # Serve the lightweight search UI.
    if not FRONTEND_PATH.exists():
        raise HTTPException(status_code=500, detail="Frontend file is missing.")
    return HTMLResponse(FRONTEND_PATH.read_text(encoding="utf-8"))


@app.get("/guide", response_class=HTMLResponse)
def guide() -> HTMLResponse:
    # Serve a simple usage guide for the frontend page.
    if not GUIDE_PATH.exists():
        raise HTTPException(status_code=500, detail="Guide file is missing.")
    return HTMLResponse(GUIDE_PATH.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/localities")
def localities(query: str = "") -> dict[str, list[str]]:
    # Provide case-insensitive locality autocomplete suggestions.
    clean_df = load_clean_data()
    names = sorted(clean_df["locality"].dropna().astype(str).str.upper().unique().tolist())
    if query.strip():
        query_upper = query.strip().upper()
        names = [name for name in names if query_upper in name]
    return {"items": names[:10]}


@app.get("/forecast", response_model=ForecastResponse)
def forecast(locality: str) -> ForecastResponse:
    # Return summary data for the next-quarter forecast.
    return build_forecast(locality)


@app.get("/forecast-chart")
def forecast_chart(locality: str) -> StreamingResponse:
    # Render a chart where the forecast segment is visually distinct from history.
    forecast_result = build_forecast(locality)
    locality_history = get_locality_history(locality)

    history_labels = [f"{int(row.year)} Q{int(row.quarter)}" for row in locality_history.itertuples()]
    history_values = locality_history["median_price"].tolist()
    forecast_label = f"{forecast_result.prediction_year} Q{forecast_result.prediction_quarter}"

    # Plot the historical series and then extend it with a visually distinct forecast segment.
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(history_labels, history_values, marker="o", linewidth=2.5, color="#1f77b4", label="Historical price")
    ax.plot(
        [history_labels[-1], forecast_label],
        [history_values[-1], forecast_result.predicted_median_price],
        marker="o",
        linewidth=3,
        linestyle="--",
        color="#d62728",
        label="Forecast segment",
    )

    # Label every historical node so users can read each quarter directly from the chart.
    for label, value in zip(history_labels, history_values):
        ax.annotate(
            f"${value:,.0f}",
            (label, value),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color="#1f77b4",
            fontsize=9,
        )

    ax.annotate(
        f"${forecast_result.predicted_median_price:,.0f}",
        (forecast_label, forecast_result.predicted_median_price),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        color="#d62728",
        fontsize=10,
        fontweight="bold",
    )

    ax.set_title(f"{forecast_result.locality}: Quarterly Price Trend and Next-Quarter Forecast")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Median house price")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    # Use explicit lag values when provided; otherwise fall back to the latest exported history.
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

    # The saved model expects one row with the engineered features used during training.
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
