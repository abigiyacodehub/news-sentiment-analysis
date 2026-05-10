"""Validate that the bundled sample datasets support the notebooks."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.eda_utils import add_headline_features
from src.technical_indicators import fill_missing_price_data, load_price_data


def main() -> None:
    news_path = PROJECT_ROOT / "data" / "raw" / "financial_news_sample.csv"
    price_path = PROJECT_ROOT / "data" / "raw" / "aapl_historical_prices_sample.csv"
    multi_price_path = PROJECT_ROOT / "data" / "raw" / "multi_stock_prices_sample.csv"

    news = add_headline_features(pd.read_csv(news_path))
    prices = fill_missing_price_data(load_price_data(price_path))
    multi_prices = pd.read_csv(multi_price_path)

    print(f"News rows: {len(news)}")
    print(f"Price rows: {len(prices)}")
    print(f"Multi-stock price rows: {len(multi_prices)}")
    print("Datasets validated successfully.")


if __name__ == "__main__":
    main()
