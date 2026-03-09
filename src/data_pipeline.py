from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

QUARTER_NAME_TO_NUM = {
    "jan-mar": 1,
    "apr-jun": 2,
    "jul-sep": 3,
    "oct-dec": 4,
}

PRICE_EXCLUDE_KEYWORDS = ("sale", "change", "ytd", "no.", "no of", "no.", "%")
LOCALITY_KEYWORDS = ("locality", "suburb")


@dataclass
class QuarterColumn:
    column_name: str
    quarter_num: int
    year: int


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _parse_numeric(value: object) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    cleaned = re.sub(r"[^0-9\.-]", "", text)
    if cleaned in {"", "-", ".", "-."}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_year_from_series(values: Iterable[object]) -> int | None:
    for value in values:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        match = re.search(r"(19|20)\d{2}", str(value))
        if match:
            return int(match.group(0))
    return None


def _extract_quarter_and_year(raw_col_name: str, sample_values: pd.Series) -> tuple[int, int] | None:
    col = _normalize_text(raw_col_name.replace("\n", " "))
    if any(keyword in col for keyword in PRICE_EXCLUDE_KEYWORDS):
        return None

    quarter_num = None
    for q_name, q_num in QUARTER_NAME_TO_NUM.items():
        if q_name in col:
            quarter_num = q_num
            break

    if quarter_num is None:
        return None

    year_match = re.search(r"(19|20)\d{2}", col)
    year = int(year_match.group(0)) if year_match else _extract_year_from_series(sample_values.head(5).tolist())

    if year is None:
        return None

    return quarter_num, year


def _get_locality_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        normalized = _normalize_text(col)
        if any(keyword in normalized for keyword in LOCALITY_KEYWORDS):
            return col
    return df.columns[0]


def _read_one_file(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0, dtype=object)
    if df.empty:
        return pd.DataFrame(columns=["locality", "year", "quarter", "median_price", "source_file"])

    locality_col = _get_locality_column(df)
    working = df.copy()
    working[locality_col] = working[locality_col].astype(str).str.strip()

    records: list[dict[str, object]] = []

    for col in working.columns:
        if col == locality_col:
            continue
        parsed = _extract_quarter_and_year(str(col), working[col])
        if not parsed:
            continue

        quarter_num, year = parsed
        for _, row in working[[locality_col, col]].iterrows():
            locality = str(row[locality_col]).strip()
            if not locality or locality.lower() == "nan":
                continue

            if re.search(r"^(total|grand total|median house prices)", locality.lower()):
                continue

            price = _parse_numeric(row[col])
            if price is None or price <= 0:
                continue

            records.append(
                {
                    "locality": locality.upper(),
                    "year": int(year),
                    "quarter": int(quarter_num),
                    "median_price": float(price),
                    "source_file": path.name,
                }
            )

    if not records:
        return pd.DataFrame(columns=["locality", "year", "quarter", "median_price", "source_file"])
    return pd.DataFrame.from_records(records)


def load_and_clean_dataset(dataset_dir: str | Path) -> pd.DataFrame:
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    files = sorted(dataset_path.glob("*.xls"))
    if not files:
        raise FileNotFoundError(f"No .xls files found in: {dataset_path}")

    all_parts = [_read_one_file(path) for path in files]
    non_empty_parts = [part for part in all_parts if not part.empty]
    if not non_empty_parts:
        raise ValueError("No valid price data extracted from VPSR files.")
    df = pd.concat(non_empty_parts, ignore_index=True)

    # Deduplicate same locality-quarter from multiple reports using average price.
    deduped = (
        df.groupby(["locality", "year", "quarter"], as_index=False)["median_price"]
        .mean()
        .sort_values(["locality", "year", "quarter"])
    )

    deduped["date"] = pd.PeriodIndex.from_fields(
        year=deduped["year"], quarter=deduped["quarter"], freq="Q"
    ).to_timestamp()
    return deduped


def save_clean_data(dataset_dir: str | Path, output_csv: str | Path) -> pd.DataFrame:
    cleaned = load_and_clean_dataset(dataset_dir)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    return cleaned
