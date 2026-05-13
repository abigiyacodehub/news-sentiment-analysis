from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.correlation import aggregate_daily_sentiment, map_to_next_trading_day, pearson_correlation
from src.data_loading import REQUIRED_PRICE_COLUMNS, clean_stock_prices, load_news_data
from src.indicators import add_indicator_columns, daily_returns
from src.news_analysis import (
    add_headline_features,
    add_publisher_domain,
    daily_publication_counts,
    spike_days,
)
from src.sentiment import lexicon_sentiment, sentiment_category


PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_NEWS_PATH = PROJECT_ROOT / "data" / "raw" / "financial_news_sample.csv"
SAMPLE_PRICE_PATH = PROJECT_ROOT / "data" / "raw" / "aapl_historical_prices_sample.csv"

st.set_page_config(page_title="News Sentiment Price Moves", layout="wide")
plt.style.use("seaborn-v0_8-whitegrid")

STOP_WORDS = {
    "a",
    "after",
    "and",
    "as",
    "at",
    "for",
    "from",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
}


@st.cache_data(show_spinner=False)
def load_news(sample_rows: int | None, use_sample_file: bool) -> pd.DataFrame:
    if use_sample_file and SAMPLE_NEWS_PATH.exists():
        df = pd.read_csv(SAMPLE_NEWS_PATH)
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        return df
    return load_news_data(nrows=sample_rows)


@st.cache_data(show_spinner=False)
def load_prices_from_path(path: str) -> pd.DataFrame:
    return normalize_prices(pd.read_csv(path))


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in ["Date", *REQUIRED_PRICE_COLUMNS] if col not in df.columns]
    if missing:
        raise ValueError(f"Price data is missing required columns: {missing}")
    prices = df.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    for col in REQUIRED_PRICE_COLUMNS:
        prices[col] = pd.to_numeric(prices[col], errors="coerce")
    prices = prices.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    return clean_stock_prices(prices[REQUIRED_PRICE_COLUMNS])


def score_sentiment(headlines: pd.Series) -> pd.Series:
    try:
        from textblob import TextBlob

        return headlines.fillna("").astype(str).map(lambda text: TextBlob(text).sentiment.polarity)
    except Exception:
        return headlines.fillna("").astype(str).map(lexicon_sentiment)


def plot_to_streamlit(fig) -> None:
    st.pyplot(fig, clear_figure=True)


def extract_keyword_counts(headlines: pd.Series, top_n: int = 20) -> pd.DataFrame:
    words: Counter[str] = Counter()
    for headline in headlines.dropna().astype(str):
        tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", headline.lower())
        words.update(token for token in tokens if token not in STOP_WORDS)
    return pd.DataFrame(words.most_common(top_n), columns=["term", "count"])


def show_overview() -> None:
    st.title("News Sentiment Price Moves")
    st.caption("Interactive dashboard for the EDA, technical analysis, and sentiment-correlation rubric.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Task 1", "EDA + NLP", "complete")
    col2.metric("Task 2", "Indicators", "complete")
    col3.metric("Task 3", "Correlation", "complete")

    st.write(
        "Use the sidebar to switch between sections. The app reads the provided "
        "`newsData.zip` when available, or the sample data committed under `data/raw/`."
    )

    st.subheader("Project Artifacts")
    st.dataframe(
        pd.DataFrame(
            {
                "Area": ["EDA", "Quantitative", "Correlation", "CI"],
                "Primary artifact": [
                    "notebooks/01_eda_news.ipynb",
                    "notebooks/02_quantitative_analysis.ipynb",
                    "notebooks/03_sentiment_correlation.ipynb",
                    ".github/workflows/unittests.yml",
                ],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def show_news_eda(news: pd.DataFrame) -> None:
    st.header("Task 1: News EDA")
    news = add_headline_features(add_publisher_domain(news))

    metric_cols = st.columns(4)
    metric_cols[0].metric("Articles", f"{len(news):,}")
    metric_cols[1].metric("Publishers", f"{news['publisher'].nunique():,}")
    metric_cols[2].metric("Stocks", f"{news['stock'].nunique():,}")
    metric_cols[3].metric("Avg headline chars", f"{news['headline_char_count'].mean():.1f}")

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Headline Length Summary")
        st.dataframe(news[["headline_char_count", "headline_word_count"]].describe().T, use_container_width=True)
    with right:
        st.subheader("Email Publisher Domains")
        domain_counts = (
            news.dropna(subset=["publisher_domain"])["publisher_domain"]
            .value_counts()
            .head(10)
            .rename_axis("domain")
            .reset_index(name="articles")
        )
        st.dataframe(domain_counts, use_container_width=True, hide_index=True)

    st.subheader("Most Active Publishers")
    top_publishers = news["publisher"].value_counts().head(15).rename_axis("publisher").reset_index(name="articles")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(top_publishers["publisher"], top_publishers["articles"], color="#33658A")
    ax.invert_yaxis()
    ax.set_title("Top Publishers by Article Count")
    ax.set_xlabel("Article Count")
    ax.set_ylabel("Publisher")
    plot_to_streamlit(fig)

    st.subheader("Publication Frequency")
    daily_counts = daily_publication_counts(news)
    spikes = spike_days(daily_counts)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pd.to_datetime(daily_counts.index), daily_counts.values, color="#2F4858", label="Daily articles")
    if not spikes.empty:
        ax.scatter(pd.to_datetime(spikes.index), spikes.values, color="#F26419", label="Spike days")
    ax.set_title("Daily Publication Frequency")
    ax.set_xlabel("Date")
    ax.set_ylabel("Article Count")
    ax.legend()
    plot_to_streamlit(fig)

    st.subheader("Keyword Themes")
    headlines = news["headline"].dropna().astype(str)
    if len(headlines) > 100_000:
        headlines = headlines.sample(100_000, random_state=42)
    keywords = extract_keyword_counts(headlines)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(keywords["term"], keywords["count"], color="#6A994E")
    ax.invert_yaxis()
    ax.set_title("Top Financial News Keywords")
    ax.set_xlabel("Count")
    ax.set_ylabel("Keyword")
    plot_to_streamlit(fig)


def show_technical_analysis(prices: pd.DataFrame) -> pd.DataFrame:
    st.header("Task 2: Technical Indicators")
    indicators = add_indicator_columns(prices, windows=(20, 50))
    indicators["daily_return_pct"] = daily_returns(indicators["Adj Close"])

    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows", f"{len(indicators):,}")
    metric_cols[1].metric("Latest close", f"{indicators['Close'].iloc[-1]:,.2f}")
    metric_cols[2].metric("Mean return", f"{indicators['daily_return_pct'].mean():.2f}%")
    metric_cols[3].metric("Volatility", f"{indicators['daily_return_pct'].std():.2f}%")

    st.subheader("Close with SMA and EMA")
    fig, ax = plt.subplots(figsize=(12, 5))
    indicators[["Close", "SMA_20", "SMA_50", "EMA_20", "EMA_50"]].plot(ax=ax)
    ax.set_title("Closing Price with Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plot_to_streamlit(fig)

    st.subheader("RSI")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(indicators.index, indicators["RSI_14"], color="#33658A", label="RSI 14")
    ax.axhline(70, color="#BC4749", linestyle="--", label="Overbought")
    ax.axhline(30, color="#6A994E", linestyle="--", label="Oversold")
    ax.set_title("Relative Strength Index")
    ax.set_xlabel("Date")
    ax.set_ylabel("RSI")
    ax.legend()
    plot_to_streamlit(fig)

    st.subheader("MACD")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(indicators.index, indicators["MACD"], label="MACD", color="#2F4858")
    ax.plot(indicators.index, indicators["Signal"], label="Signal", color="#F26419")
    ax.bar(indicators.index, indicators["Histogram"], label="Histogram", color="#86BBD8", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("MACD, Signal Line, and Divergence/Convergence")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    plot_to_streamlit(fig)

    returns = indicators["daily_return_pct"].dropna() / 100
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    st.subheader("PyNance-Style Financial Metrics")
    st.dataframe(
        pd.DataFrame(
            {
                "metric": [
                    "Cumulative return",
                    "Annualized volatility",
                    "Sharpe ratio, zero risk-free rate",
                    "Maximum drawdown",
                ],
                "value": [
                    f"{(wealth.iloc[-1] - 1) * 100:.2f}%",
                    f"{returns.std() * (252 ** 0.5) * 100:.2f}%",
                    f"{(returns.mean() / returns.std()) * (252 ** 0.5):.2f}",
                    f"{drawdown.min() * 100:.2f}%",
                ],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    return indicators


def show_sentiment_correlation(news: pd.DataFrame, prices: pd.DataFrame) -> None:
    st.header("Task 3: Sentiment and Stock Movement")
    available = sorted(news["stock"].dropna().astype(str).unique())
    default_index = available.index("AAPL") if "AAPL" in available else 0
    ticker = st.selectbox("Stock ticker for correlation", available, index=default_index)

    stock_news = news[news["stock"].astype(str).eq(ticker)].copy()
    stock_news["sentiment_score"] = score_sentiment(stock_news["headline"])
    stock_news["sentiment_category"] = stock_news["sentiment_score"].map(sentiment_category)
    stock_news["trading_day"] = map_to_next_trading_day(stock_news["date"], prices.index)

    returns = pd.DataFrame(
        {"trading_day": prices.index, "daily_return_pct": daily_returns(prices["Adj Close"]).values}
    ).dropna()
    daily_sentiment = aggregate_daily_sentiment(stock_news)
    aligned = daily_sentiment.merge(returns, on="trading_day", how="inner")

    if aligned.empty:
        st.warning(
            "No overlapping dates between the selected news and price data. "
            "Upload a price CSV covering the same period as the selected stock news."
        )
        return

    corr = pearson_correlation(aligned)
    metric_cols = st.columns(3)
    metric_cols[0].metric("Aligned trading days", f"{len(aligned):,}")
    metric_cols[1].metric("Average sentiment", f"{aligned['avg_sentiment'].mean():.3f}")
    metric_cols[2].metric("Pearson r", f"{corr:.3f}")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(aligned["avg_sentiment"], aligned["daily_return_pct"], alpha=0.7, color="#33658A")
    if len(aligned) > 1:
        slope, intercept = np.polyfit(aligned["avg_sentiment"], aligned["daily_return_pct"], 1)
        x_values = np.linspace(aligned["avg_sentiment"].min(), aligned["avg_sentiment"].max(), 100)
        ax.plot(x_values, slope * x_values + intercept, color="#F26419", label="Trend")
        ax.legend()
    ax.set_title(f"{ticker}: Average Daily Sentiment vs Daily Return")
    ax.set_xlabel("Average Daily Sentiment")
    ax.set_ylabel("Daily Return (%)")
    ax.annotate(f"Pearson r = {corr:.3f}", xy=(0.05, 0.95), xycoords="axes fraction", va="top")
    plot_to_streamlit(fig)

    category_returns = (
        aligned.groupby("dominant_category", as_index=False)["daily_return_pct"]
        .mean()
        .rename(columns={"daily_return_pct": "avg_daily_return_pct"})
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(category_returns["dominant_category"], category_returns["avg_daily_return_pct"], color="#6A994E")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{ticker}: Average Daily Return by Sentiment Category")
    ax.set_xlabel("Sentiment Category")
    ax.set_ylabel("Average Daily Return (%)")
    plot_to_streamlit(fig)

    st.info(
        "Interpretation: the Pearson value shows linear same-day association, not causation. "
        "Lag effects, publication time, earnings, macro shocks, and market beta can all confound this relationship."
    )


def sidebar_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    st.sidebar.title("Controls")
    use_sample_news = st.sidebar.checkbox("Use committed sample news", value=True)
    sample_rows = st.sidebar.select_slider(
        "Rows to load from newsData.zip",
        options=[10_000, 25_000, 50_000, 100_000, None],
        value=25_000,
        format_func=lambda value: "All rows" if value is None else f"{value:,}",
        disabled=use_sample_news,
    )

    with st.spinner("Loading news data..."):
        news = load_news(sample_rows, use_sample_news)
    st.sidebar.caption(f"Loaded {len(news):,} news rows.")

    uploaded_prices = st.sidebar.file_uploader("Upload OHLCV stock price CSV", type=["csv"])
    if uploaded_prices is not None:
        prices = normalize_prices(pd.read_csv(uploaded_prices))
    elif SAMPLE_PRICE_PATH.exists():
        prices = load_prices_from_path(str(SAMPLE_PRICE_PATH))
    else:
        st.sidebar.error("No sample price CSV found. Upload a price file to use Task 2 and Task 3.")
        prices = pd.DataFrame()
    return news, prices


def main() -> None:
    news, prices = sidebar_inputs()
    section = st.sidebar.radio(
        "Section",
        ["Overview", "News EDA", "Technical Analysis", "Sentiment Correlation"],
    )

    if section == "Overview":
        show_overview()
    elif section == "News EDA":
        show_news_eda(news)
    elif section == "Technical Analysis":
        if prices.empty:
            st.warning("Upload a stock price CSV to view technical indicators.")
        else:
            show_technical_analysis(prices)
    elif section == "Sentiment Correlation":
        if prices.empty:
            st.warning("Upload a stock price CSV to view sentiment correlation.")
        else:
            show_sentiment_correlation(news, prices)


if __name__ == "__main__":
    main()
