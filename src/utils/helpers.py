"""General utility helpers."""
import pandas as pd
import numpy as np


def risk_band_color(band: str) -> str:
    return {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}.get(band, "#6b7280")


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def dataframe_to_records(df: pd.DataFrame, max_rows: int = 500) -> list:
    return df.head(max_rows).replace({np.nan: None}).to_dict(orient="records")


def safe_json(df: pd.DataFrame, max_rows: int = 20) -> str:
    return df.head(max_rows).replace({np.nan: None}).to_json(orient="records", indent=2)
