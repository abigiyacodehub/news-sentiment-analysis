# News Sentiment Analysis

This repository contains the Week 1 financial news sentiment analysis work. It covers exploratory data analysis of market news headlines, initial technical-indicator analysis for stock prices, and a basic CI workflow for automated tests.

## Contents

- `notebooks/task_1_eda.ipynb`: EDA notebook covering descriptive statistics, publisher activity, NLP keyword analysis, and news-volume time series.
- `notebooks/task_2_technical_indicators.ipynb`: initial technical-indicator notebook loading historical stock prices, handling missing values, and visualizing TA-Lib indicators.
- `data/raw/financial_news_sample.csv`: self-contained financial news sample used by the notebook.
- `data/raw/aapl_historical_prices_sample.csv`: historical AAPL OHLCV sample used by the Task 2 notebook.
- `src/eda_utils.py`: small reusable helpers for schema validation and EDA summaries.
- `src/technical_indicators.py`: reusable helpers for loading and cleaning OHLCV price data.
- `tests/`: unit tests for helper functions.
- `scripts/validate_datasets.py`: quick validation script for the bundled sample datasets.
- `.github/workflows/unittests.yml`: GitHub Actions workflow for CI.

## Repository Structure

```text
.github/workflows/      GitHub Actions CI configuration
data/raw/               Raw sample datasets
notebooks/              Jupyter notebooks for Task 1 and Task 2
scripts/                Utility scripts
src/                    Reusable project helper modules
tests/                  Unit tests
```

## Data Sources

The current repository uses small, self-contained sample datasets so the notebooks can run reproducibly during training:

- `financial_news_sample.csv`: curated financial-news headline sample with `date`, `publisher`, `stock`, `headline`, and `market_event` fields.
- `aapl_historical_prices_sample.csv`: AAPL OHLCV historical-price sample with one intentional missing value to demonstrate data-quality handling.

For a production version, replace these files with the full Financial News and Stock Price Integration Dataset or another approved financial news dataset, plus historical OHLCV data from a provider such as Yahoo Finance.

## Run Locally

```bash
pip install -r requirements.txt
pytest -q
jupyter notebook notebooks/task_1_eda.ipynb
```

For Task 2 technical indicators:

```bash
jupyter notebook notebooks/task_2_technical_indicators.ipynb
```

To validate the bundled datasets:

```bash
python scripts/validate_datasets.py
```
