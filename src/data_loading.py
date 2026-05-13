from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from .config import NEWS_ZIP_CANDIDATES, STOCK_PRICE_DIR


REQUIRED_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def find_news_zip(candidates: list[Path] | None = None) -> Path:
    """Return the first available newsData.zip path."""
    for path in candidates or NEWS_ZIP_CANDIDATES:
        if path.exists():
            return path
    searched = ", ".join(str(path) for path in candidates or NEWS_ZIP_CANDIDATES)
    raise FileNotFoundError(f"newsData.zip not found. Searched: {searched}")


def load_news_data(path: str | Path | None = None, nrows: int | None = None) -> pd.DataFrame:
    """Load Benzinga analyst ratings from the provided zip file."""
    zip_path = Path(path) if path else find_news_zip()
    with ZipFile(zip_path) as archive:
        csv_name = next(name for name in archive.namelist() if name.endswith(".csv"))
        df = pd.read_csv(archive.open(csv_name), nrows=nrows)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    if "" in df.columns:
        df = df.drop(columns=[""])
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["headline"] = df["headline"].fillna("").astype(str)
    df["publisher"] = df["publisher"].fillna("Unknown").astype(str)
    df["stock"] = df["stock"].fillna("").astype(str)
    return df


def load_stock_prices(ticker: str, price_dir: str | Path = STOCK_PRICE_DIR) -> pd.DataFrame:
    """Load local OHLCV prices for a ticker and normalize column types."""
    path = Path(price_dir) / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing stock price file for {ticker}: {path}. "
            "Add a local CSV or use yfinance in the notebook."
        )

    df = pd.read_csv(path)
    missing = [col for col in ["Date", *REQUIRED_PRICE_COLUMNS] if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in REQUIRED_PRICE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    return df


def clean_stock_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing OHLCV values with time-series-friendly forward/back fill."""
    cleaned = df.copy()
    cleaned[REQUIRED_PRICE_COLUMNS] = cleaned[REQUIRED_PRICE_COLUMNS].ffill().bfill()
    cleaned["Volume"] = cleaned["Volume"].fillna(0)
    return cleaned
