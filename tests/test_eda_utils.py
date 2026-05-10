import pandas as pd
import pytest

from src.eda_utils import add_headline_features, daily_news_volume, publisher_summary


def sample_news_frame():
    return pd.DataFrame(
        {
            "headline": [
                "Stocks rally as inflation cools",
                "Apple shares fall after guidance",
                "Stocks rally as inflation cools again",
            ],
            "publisher": ["Reuters", "Bloomberg", "Reuters"],
            "date": ["2024-01-10", "2024-01-10", "2024-01-11"],
        }
    )


def test_add_headline_features_creates_counts():
    enriched = add_headline_features(sample_news_frame())

    assert "headline_char_count" in enriched.columns
    assert "headline_word_count" in enriched.columns
    assert enriched.loc[0, "headline_word_count"] == 5


def test_publisher_summary_orders_most_active_first():
    summary = publisher_summary(sample_news_frame())

    assert summary.iloc[0]["publisher"] == "Reuters"
    assert summary.iloc[0]["article_count"] == 2


def test_daily_news_volume_counts_articles_per_day():
    volume = daily_news_volume(sample_news_frame())

    assert volume.loc[volume["date"] == pd.Timestamp("2024-01-10"), "article_count"].iloc[0] == 2
    assert volume.loc[volume["date"] == pd.Timestamp("2024-01-11"), "article_count"].iloc[0] == 1


def test_missing_required_column_raises_clear_error():
    with pytest.raises(ValueError, match="publisher"):
        publisher_summary(pd.DataFrame({"headline": ["Missing source"], "date": ["2024-01-01"]}))
