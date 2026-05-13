from __future__ import annotations

import pandas as pd


POSITIVE_WORDS = {
    "beat",
    "beats",
    "bullish",
    "buy",
    "gain",
    "gains",
    "higher",
    "outperform",
    "positive",
    "raise",
    "raises",
    "rally",
    "surge",
    "upgrade",
    "upgraded",
}
NEGATIVE_WORDS = {
    "bearish",
    "cut",
    "downgrade",
    "downgraded",
    "drop",
    "falls",
    "loss",
    "lower",
    "miss",
    "negative",
    "plunge",
    "sell",
    "slump",
    "weak",
}


def lexicon_sentiment(text: str) -> float:
    """Small fallback sentiment scorer used when TextBlob/VADER is unavailable."""
    words = {word.strip(".,:;!?()[]{}'\"").lower() for word in str(text).split()}
    score = len(words & POSITIVE_WORDS) - len(words & NEGATIVE_WORDS)
    return max(-1.0, min(1.0, score / 3))


def sentiment_category(score: float, threshold: float = 0.05) -> str:
    if score > threshold:
        return "positive"
    if score < -threshold:
        return "negative"
    return "neutral"


def add_sentiment_columns(news: pd.DataFrame, score_column: str = "sentiment_score") -> pd.DataFrame:
    df = news.copy()
    df[score_column] = df["headline"].map(lexicon_sentiment)
    df["sentiment_category"] = df[score_column].map(sentiment_category)
    return df
