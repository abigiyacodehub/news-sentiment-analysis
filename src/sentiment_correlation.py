"""Utilities for Task 3 sentiment and stock-return correlation analysis."""

from __future__ import annotations

import pandas as pd


POSITIVE_WORDS = {
    "accelerates",
    "adoption",
    "beat",
    "beats",
    "boosts",
    "buyback",
    "climb",
    "climbs",
    "expands",
    "growth",
    "improve",
    "improves",
    "jumps",
    "largest",
    "record",
    "rally",
    "rises",
    "stronger",
    "surges",
    "tops",
}

NEGATIVE_WORDS = {
    "abuse",
    "antitrust",
    "complaints",
    "concern",
    "concerns",
    "cuts",
    "decline",
    "declines",
    "fine",
    "falls",
    "investigation",
    "misses",
    "narrows",
    "penalty",
    "pressure",
    "risk",
    "slips",
    "slows",
    "under",
    "weaker",
}


def normalize_news_dates(news: pd.DataFrame) -> pd.DataFrame:
    """Normalize publication dates to day-level timestamps."""
    dated = news.copy()
    dated["date"] = pd.to_datetime(dated["date"], format="mixed").dt.normalize()
    return dated


def score_headline_sentiment(headline: str) -> float:
    """Score headline sentiment with a tiny deterministic finance lexicon."""
    words = {
        token.strip(".,:;!?()[]{}\"'").lower()
        for token in str(headline).split()
    }
    positive_hits = len(words.intersection(POSITIVE_WORDS))
    negative_hits = len(words.intersection(NEGATIVE_WORDS))
    total_hits = positive_hits + negative_hits
    if total_hits == 0:
        return 0.0
    return (positive_hits - negative_hits) / total_hits


def aggregate_daily_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    """Average sentiment by stock and publication date."""
    dated = normalize_news_dates(news)
    if "sentiment_score" not in dated.columns:
        dated["sentiment_score"] = dated["headline"].apply(score_headline_sentiment)
    return (
        dated.groupby(["stock", "date"], as_index=False)
        .agg(
            avg_sentiment=("sentiment_score", "mean"),
            article_count=("headline", "size"),
        )
    )


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily percentage returns by stock from close prices."""
    required = {"date", "stock", "Close"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    returns = prices.copy()
    returns["date"] = pd.to_datetime(returns["date"], format="mixed").dt.normalize()
    returns = returns.sort_values(["stock", "date"])
    returns["daily_return"] = returns.groupby("stock")["Close"].pct_change()
    return returns.dropna(subset=["daily_return"])


def align_sentiment_with_returns(sentiment: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Join daily sentiment with same-day stock returns."""
    return sentiment.merge(
        returns[["stock", "date", "Close", "daily_return"]],
        on=["stock", "date"],
        how="inner",
    )


def pearson_correlation(aligned: pd.DataFrame) -> float:
    """Return Pearson correlation between average sentiment and daily returns."""
    if len(aligned) < 2:
        return float("nan")
    return float(aligned["avg_sentiment"].corr(aligned["daily_return"], method="pearson"))
