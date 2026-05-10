# Predicting Price Moves with News Sentiment

## Overview

This project analyzes financial news headlines and stock price data to explore whether headline sentiment is related to stock movement. The work covers three stages: exploratory data analysis, technical indicator analysis, and sentiment-return correlation.

## Data

The repository uses self-contained sample datasets for reproducibility:

- `data/raw/financial_news_sample.csv`: financial headlines with publisher, date, stock ticker, and market-event labels.
- `data/raw/aapl_historical_prices_sample.csv`: AAPL OHLCV data for technical indicator analysis.
- `data/raw/multi_stock_prices_sample.csv`: multi-stock closing prices aligned with the news sample for sentiment-return correlation.

## Methodology

Task 1 explored headline lengths, publisher activity, recurring financial themes, and news-volume spikes. TF-IDF and CountVectorizer were used to identify repeated themes such as AI launches, earnings results, regulatory pressure, cloud growth, price cuts, and capital returns.

Task 2 loaded historical OHLCV stock prices, documented missing values, and used TA-Lib to compute SMA, EMA, RSI, and MACD. The indicators were visualized alongside price to show trend and momentum behavior.

Task 3 normalized news and price dates, scored headline sentiment using NLTK VADER, computed daily stock returns, aggregated sentiment by stock-date, and calculated Pearson correlation between average daily sentiment and daily returns.

## Key Findings

Publisher activity is concentrated among major financial and technology news outlets, with several event-driven publication spikes around earnings, AI product launches, regulatory actions, and buyback announcements.

Technical indicators show how short-term trend and momentum can summarize price movement. Moving averages smooth daily noise, RSI highlights momentum strength, and MACD helps identify trend shifts.

The sentiment-correlation analysis shows a positive directional relationship in the bundled sample: positive earnings, AI, cloud, and buyback headlines align with stronger same-day returns, while regulatory, delivery, price-cut, and demand-concern headlines align with weaker returns.

## Limitations

The bundled data is a compact training sample intended for reproducible assignment execution. A production analysis should replace it with the full FNSPID dataset and complete historical OHLCV data from an approved provider such as Yahoo Finance.

## Recommendation

Use sentiment as a supporting signal rather than a standalone trading signal. The strongest workflow combines news sentiment, publication timing, technical indicators, and risk controls before making investment decisions.
