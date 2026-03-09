from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_pipeline import load_and_clean_dataset

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None


def build_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    df = clean_df.copy().sort_values(["locality", "year", "quarter"])

    by_loc = df.groupby("locality")
    df["lag_1"] = by_loc["median_price"].shift(1)
    df["lag_2"] = by_loc["median_price"].shift(2)
    df["rolling_mean_2"] = by_loc["median_price"].shift(1).rolling(2).mean().reset_index(level=0, drop=True)

    # Use median fallbacks so sparse localities can still be scored.
    for col in ["lag_1", "lag_2", "rolling_mean_2"]:
        df[col] = df[col].fillna(df["median_price"].median())

    return df


def time_split(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_periods = (
        df[["year", "quarter"]]
        .drop_duplicates()
        .sort_values(["year", "quarter"])
        .reset_index(drop=True)
    )

    cutoff_idx = max(1, int(len(unique_periods) * (1 - test_ratio)))
    cutoff = unique_periods.iloc[cutoff_idx - 1]

    is_train = (df["year"] < cutoff["year"]) | ((df["year"] == cutoff["year"]) & (df["quarter"] <= cutoff["quarter"]))
    train_df = df[is_train].copy()
    test_df = df[~is_train].copy()

    if test_df.empty:
        train_df = df.iloc[:-1].copy()
        test_df = df.iloc[-1:].copy()

    return train_df, test_df


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_preprocessor() -> ColumnTransformer:
    numeric_features = ["year", "quarter", "lag_1", "lag_2", "rolling_mean_2"]
    categorical_features = ["locality"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def train_and_compare(dataset_dir: str | Path, artifacts_dir: str | Path) -> dict[str, dict[str, float]]:
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    clean_df = load_and_clean_dataset(dataset_dir)
    feat_df = build_features(clean_df)

    feat_df.to_csv(artifacts_path / "clean_long_data.csv", index=False)

    train_df, test_df = time_split(feat_df)

    feature_cols = ["locality", "year", "quarter", "lag_1", "lag_2", "rolling_mean_2"]
    target_col = "median_price"

    x_train, y_train = train_df[feature_cols], train_df[target_col]
    x_test, y_test = test_df[feature_cols], test_df[target_col]

    models: dict[str, object] = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }

    skipped_models: dict[str, str] = {}
    if XGBRegressor is not None:
        models["xgboost"] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="reg:squarederror",
        )
    else:
        skipped_models["xgboost"] = "xgboost package not installed"

    if LGBMRegressor is not None:
        models["lightgbm"] = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="regression",
        )
    else:
        skipped_models["lightgbm"] = "lightgbm package not installed"

    metrics: dict[str, dict[str, float]] = {}
    trained_pipelines: dict[str, Pipeline] = {}

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", model),
            ]
        )
        pipeline.fit(x_train, y_train)
        preds = pipeline.predict(x_test)

        model_metrics = evaluate(y_test.to_numpy(), preds)
        metrics[name] = model_metrics
        trained_pipelines[name] = pipeline

    best_model_name = min(metrics.keys(), key=lambda m: metrics[m]["rmse"])
    best_pipeline = trained_pipelines[best_model_name]

    joblib.dump(best_pipeline, artifacts_path / "best_model.joblib")

    meta = {
        "best_model": best_model_name,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "locality_count": int(clean_df["locality"].nunique()),
    }
    if skipped_models:
        meta["skipped_models"] = skipped_models

    with open(artifacts_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "meta": meta}, f, ensure_ascii=False, indent=2)

    # Keep latest per-locality history for API lag feature defaults.
    latest_history = (
        clean_df.sort_values(["locality", "year", "quarter"]).groupby("locality", as_index=False).tail(2)
    )
    latest_history.to_csv(artifacts_path / "latest_history.csv", index=False)

    return metrics


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    dataset_dir = root / "dataset"
    artifacts_dir = root / "artifacts"

    result = train_and_compare(dataset_dir=dataset_dir, artifacts_dir=artifacts_dir)
    print("Training done. Metrics:")
    for model_name, vals in result.items():
        print(model_name, vals)
