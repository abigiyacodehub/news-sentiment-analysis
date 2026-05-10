# News Sentiment Analysis

Task 1 implementation for Git, GitHub, and Exploratory Data Analysis.

## Contents

- `notebooks/task_1_eda.ipynb`: EDA notebook covering descriptive statistics, publisher activity, NLP keyword analysis, and news-volume time series.
- `notebooks/task_2_technical_indicators.ipynb`: initial technical-indicator notebook loading historical stock prices, handling missing values, and visualizing TA-Lib indicators.
- `data/raw/financial_news_sample.csv`: self-contained financial news sample used by the notebook.
- `data/raw/aapl_historical_prices_sample.csv`: historical AAPL OHLCV sample used by the Task 2 notebook.
- `src/eda_utils.py`: small reusable helpers for schema validation and EDA summaries.
- `tests/test_eda_utils.py`: unit tests for the helper functions.
- `.github/workflows/unittests.yml`: GitHub Actions workflow for CI.

## Run Locally

```bash
pip install -r requirements.txt
pytest -q
jupyter notebook notebooks/task_1_eda.ipynb
```

For Task 2 technical indicators, install the optional TA-Lib dependency:

```bash
pip install -r requirements-task2.txt
jupyter notebook notebooks/task_2_technical_indicators.ipynb
```
