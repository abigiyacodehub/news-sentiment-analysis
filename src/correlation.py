from __future__ import annotations

import pandas as pd


def map_to_next_trading_day(article_dates: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """Map each article date to the same or next available trading day."""
    trading_days = pd.DatetimeIndex(pd.to_datetime(trading_days).normalize()).sort_values()
    normalized = pd.to_datetime(article_dates, utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    positions = trading_days.searchsorted(normalized, side="left")
    mapped = [
        trading_days[pos] if pd.notna(date) and pos < len(trading_days) else pd.NaT
        for date, pos in zip(normalized, positions)
    ]
    return pd.Series(mapped, index=article_dates.index, name="trading_day")


def aggregate_daily_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        news.dropna(subset=["trading_day"])
        .groupby(["stock", "trading_day"], as_index=False)
        .agg(
            avg_sentiment=("sentiment_score", "mean"),
            article_count=("headline", "size"),
            dominant_category=("sentiment_category", lambda s: s.value_counts().idxmax()),
        )
    )
    return grouped


def prepare_correlation_frame(
    daily_sentiment: pd.DataFrame, returns: pd.DataFrame, ticker: str
) -> pd.DataFrame:
    ticker_sentiment = daily_sentiment[daily_sentiment["stock"] == ticker].copy()
    ticker_sentiment["trading_day"] = pd.to_datetime(ticker_sentiment["trading_day"]).dt.normalize()
    returns = returns.copy()
    returns["trading_day"] = pd.to_datetime(returns["trading_day"]).dt.normalize()
    return ticker_sentiment.merge(returns, on="trading_day", how="inner")


def pearson_correlation(frame: pd.DataFrame) -> float:
    if len(frame) < 2:
        return float("nan")
    return float(frame["avg_sentiment"].corr(frame["daily_return_pct"], method="pearson"))
