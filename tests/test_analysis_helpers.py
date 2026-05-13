import pandas as pd

from src.correlation import map_to_next_trading_day, pearson_correlation
from src.indicators import daily_returns, ema, macd, sma
from src.news_analysis import publisher_domain
from src.sentiment import sentiment_category


def test_publisher_domain_extracts_email_domain():
    assert publisher_domain("writer@example.com") == "example.com"
    assert publisher_domain("Benzinga Newsdesk") is None


def test_moving_average_helpers():
    values = pd.Series([1, 2, 3, 4, 5], dtype=float)
    assert sma(values, 3).iloc[-1] == 4
    assert ema(values, 3).iloc[-1] > ema(values, 3).iloc[-2]


def test_macd_has_expected_columns():
    values = pd.Series(range(1, 40), dtype=float)
    result = macd(values)
    assert {"MACD", "Signal", "Histogram"} <= set(result.columns)


def test_daily_returns_percentage_formula():
    prices = pd.Series([100, 110, 99], dtype=float)
    returns = daily_returns(prices)
    assert round(returns.iloc[1], 2) == 10
    assert round(returns.iloc[2], 2) == -10


def test_weekend_maps_to_next_trading_day():
    dates = pd.Series(pd.to_datetime(["2024-01-06", "2024-01-08"], utc=True))
    trading_days = pd.DatetimeIndex(["2024-01-05", "2024-01-08", "2024-01-09"])
    mapped = map_to_next_trading_day(dates, trading_days)
    assert mapped.iloc[0] == pd.Timestamp("2024-01-08")
    assert mapped.iloc[1] == pd.Timestamp("2024-01-08")


def test_sentiment_category_thresholds():
    assert sentiment_category(0.10) == "positive"
    assert sentiment_category(-0.10) == "negative"
    assert sentiment_category(0.0) == "neutral"


def test_pearson_correlation_returns_float():
    frame = pd.DataFrame({"avg_sentiment": [0.1, 0.2, -0.1], "daily_return_pct": [1, 2, -1]})
    assert pearson_correlation(frame) > 0.9
