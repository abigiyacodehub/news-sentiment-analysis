"""Utility helpers for Task 1 exploratory news analysis."""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {"headline", "publisher", "date"}


def validate_news_schema(df: pd.DataFrame) -> None:
    """Raise a clear error when the news dataset cannot support Task 1 EDA."""
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_list}")


def add_headline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with headline character and word counts."""
    validate_news_schema(df)
    enriched = df.copy()
    enriched["date"] = pd.to_datetime(enriched["date"])
    enriched["headline_char_count"] = enriched["headline"].astype(str).str.len()
    enriched["headline_word_count"] = enriched["headline"].astype(str).str.split().str.len()
    return enriched


def publisher_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Count articles by publisher, sorted from most to least active."""
    validate_news_schema(df)
    return (
        df.groupby("publisher", as_index=False)
        .size()
        .rename(columns={"size": "article_count"})
        .sort_values("article_count", ascending=False)
        .reset_index(drop=True)
    )


def daily_news_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Return daily publication counts for time-series news-volume analysis."""
    validate_news_schema(df)
    dated = df.copy()
    dated["date"] = pd.to_datetime(dated["date"])
    return (
        dated.set_index("date")
        .resample("D")
        .size()
        .rename("article_count")
        .reset_index()
    )
