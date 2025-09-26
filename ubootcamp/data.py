import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Dict

def download_prices(ticker: str, start: str = "2018-01-01", end: Optional[str] = None) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"No data downloaded for {ticker}.")
    df = df.dropna()
    if "Close" not in df.columns:
        # yfinance set should include Close; but guard anyway
        if "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close":"Close"})
        else:
            raise ValueError("Close column not found in downloaded data.")
    return df

def load_from_csv(path: str, date_col: Optional[str] = None, close_col: str = "Close") -> pd.DataFrame:
    """Load OHLCV-like CSV for Kaggle.
    - path: CSV path in Kaggle like ../input/dataset/file.csv
    - date_col: optional date column name; if provided, parsed to datetime and sorted.
    - close_col: which column to treat as Close (default 'Close', can be 'Adj Close').
    The function returns a DataFrame indexed by date (if date_col given) or row index.
    It ensures a 'Close' column exists for feature engineering.
    """
    df = pd.read_csv(path)
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    # normalize close column
    if close_col not in df.columns:
        raise ValueError(f"close_col='{close_col}' not found in CSV columns: {list(df.columns)[:10]}...")
    if close_col != "Close":
        df = df.rename(columns={close_col: "Close"})
    # minimal clean
    df = df.dropna()
    # keep at least 'Close' for features; other cols optional
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Close" not in df.columns:
        raise ValueError("Expected 'Close' column in DataFrame for feature engineering.")
    df["ret"]   = df["Close"].pct_change()
    df["lag1"]  = df["Close"].shift(1)
    df["lag5"]  = df["Close"].shift(5)
    df["lag10"] = df["Close"].shift(10)
    df["sma5"]  = df["Close"].rolling(5).mean()
    df["sma10"] = df["Close"].rolling(10).mean()
    df["vol5"]  = df["ret"].rolling(5).std()
    # RSI (14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))
    return df

def make_supervised(df: pd.DataFrame, horizon: int = 1):
    data = df.copy()
    data["y"] = data["Close"].shift(-horizon)
    features = ["lag1","lag5","lag10","sma5","sma10","vol5","rsi14"]
    data = data.dropna().copy()
    X, y = data[features], data["y"]
    return X, y, features, data
