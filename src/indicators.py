from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({"MACD": macd_line, "Signal": signal_line, "Histogram": histogram})


def daily_returns(adj_close: pd.Series) -> pd.Series:
    return adj_close.pct_change() * 100


def add_indicator_columns(prices: pd.DataFrame, windows: tuple[int, ...] = (20, 50)) -> pd.DataFrame:
    df = prices.copy()
    for window in windows:
        df[f"SMA_{window}"] = sma(df["Close"], window)
        df[f"EMA_{window}"] = ema(df["Close"], window)
    df["RSI_14"] = rsi(df["Close"], 14)
    return df.join(macd(df["Close"]))
