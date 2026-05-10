import math

import pandas as pd

from src.sentiment_correlation import (
    aggregate_daily_sentiment,
    align_sentiment_with_returns,
    compute_daily_returns,
    pearson_correlation,
    score_headline_sentiment,
)


def test_score_headline_sentiment_detects_direction():
    assert score_headline_sentiment("Revenue beats expectations as shares rally") > 0
    assert score_headline_sentiment("Shares fall after antitrust investigation") < 0
    assert score_headline_sentiment("Company schedules investor meeting") == 0


def test_aggregate_daily_sentiment_groups_by_stock_and_date():
    news = pd.DataFrame(
        {
            "date": ["2024-01-01 09:30:00", "2024-01-01 14:00:00", "2024-01-02"],
            "stock": ["AAPL", "AAPL", "MSFT"],
            "headline": ["Shares rally", "Revenue beats estimates", "Shares fall"],
        }
    )

    daily = aggregate_daily_sentiment(news)

    row = daily[(daily["stock"] == "AAPL") & (daily["date"] == pd.Timestamp("2024-01-01"))].iloc[0]
    assert row["article_count"] == 2
    assert row["avg_sentiment"] > 0


def test_compute_daily_returns_and_aligns_with_sentiment():
    prices = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "stock": ["AAPL", "AAPL", "AAPL"],
            "Close": [100.0, 110.0, 99.0],
        }
    )
    sentiment = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "stock": ["AAPL", "AAPL"],
            "avg_sentiment": [1.0, -1.0],
            "article_count": [1, 1],
        }
    )

    returns = compute_daily_returns(prices)
    aligned = align_sentiment_with_returns(sentiment, returns)

    assert len(aligned) == 2
    assert math.isclose(aligned.iloc[0]["daily_return"], 0.10)
    assert pearson_correlation(aligned) > 0
