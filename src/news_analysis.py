from __future__ import annotations

import re

import pandas as pd


EMAIL_RE = re.compile(r"^[^@\s]+@(?P<domain>[^@\s]+)$")


def add_headline_features(news: pd.DataFrame) -> pd.DataFrame:
    df = news.copy()
    df["headline_char_count"] = df["headline"].fillna("").astype(str).str.len()
    df["headline_word_count"] = df["headline"].fillna("").astype(str).str.split().str.len()
    df["publish_date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.date
    return df


def publisher_domain(publisher: str) -> str | None:
    """Extract a domain from publishers stored as email addresses."""
    match = EMAIL_RE.match(str(publisher).strip())
    return match.group("domain").lower() if match else None


def add_publisher_domain(news: pd.DataFrame) -> pd.DataFrame:
    df = news.copy()
    df["publisher_domain"] = df["publisher"].map(publisher_domain)
    return df


def daily_publication_counts(news: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(news["date"], utc=True, errors="coerce").dt.date
    return dates.value_counts().sort_index()


def spike_days(daily_counts: pd.Series, z_threshold: float = 2.0) -> pd.Series:
    """Identify unusually high publication days using a simple z-score rule."""
    if daily_counts.empty or daily_counts.std(ddof=0) == 0:
        return daily_counts.iloc[0:0]
    z_scores = (daily_counts - daily_counts.mean()) / daily_counts.std(ddof=0)
    return daily_counts[z_scores >= z_threshold].sort_values(ascending=False)
