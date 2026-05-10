from pathlib import Path

from src.technical_indicators import fill_missing_price_data, load_price_data


def test_load_price_data_returns_date_indexed_frame():
    path = Path("data/raw/aapl_historical_prices_sample.csv")
    df = load_price_data(path)

    assert df.index.name == "Date"
    assert {"Open", "High", "Low", "Close", "Adj Close", "Volume"}.issubset(df.columns)
    assert len(df) >= 50


def test_fill_missing_price_data_removes_sample_missing_value():
    path = Path("data/raw/aapl_historical_prices_sample.csv")
    df = load_price_data(path)

    assert df.isna().sum().sum() > 0

    clean = fill_missing_price_data(df)

    assert clean.isna().sum().sum() == 0
