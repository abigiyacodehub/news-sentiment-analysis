# News Sentiment Analysis

Task 1 implementation for Git, GitHub, and Exploratory Data Analysis.

## Contents

- `notebooks/task_1_eda.ipynb`: EDA notebook covering descriptive statistics, publisher activity, NLP keyword analysis, and news-volume time series.
- `data/raw/financial_news_sample.csv`: self-contained financial news sample used by the notebook.
- `src/eda_utils.py`: small reusable helpers for schema validation and EDA summaries.
- `tests/test_eda_utils.py`: unit tests for the helper functions.
- `.github/workflows/unittests.yml`: GitHub Actions workflow for CI.

## Run Locally

```bash
pip install -r requirements.txt
pytest -q
jupyter notebook notebooks/task_1_eda.ipynb
```
