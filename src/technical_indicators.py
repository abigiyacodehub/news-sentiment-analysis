"""Helpers for loading stock price data used in Task 2."""

from __future__ import annotations

import pandas as pd


PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def load_price_data(path: str) -> pd.DataFrame:
    """Load historical OHLCV data and return a date-indexed DataFrame."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    for column in PRICE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def fill_missing_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fill isolated missing values using forward fill then backward fill."""
    return df.ffill().bfill()
