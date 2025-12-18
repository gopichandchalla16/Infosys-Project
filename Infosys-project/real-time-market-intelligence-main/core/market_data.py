from __future__ import annotations

import yfinance as yf
import pandas as pd
from core.utils import get_ticker


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance can return MultiIndex columns (especially with auto_adjust).
    We normalize to simple column names: Open, High, Low, Close, Volume.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Often levels like: ('Close', 'AAPL') or ('AAPL', 'Close')
        # We try to pick the "price field" level.
        # Most common: first level is field name.
        df.columns = [c[0] for c in df.columns]
    return df


def fetch_market_data(company: str):
    ticker = get_ticker(company)
    if not ticker:
        return None

    df = yf.download(
        ticker,
        period="1y",
        progress=False,
        auto_adjust=True,
        threads=True,
    )

    if df is None or df.empty:
        return None

    df = _flatten_columns(df)

    # Ensure standard columns exist
    needed = {"Open", "High", "Low", "Close"}
    if not needed.issubset(set(df.columns)):
        # If auto_adjust removed OHLC unexpectedly, fail gracefully
        if "Close" not in df.columns:
            return None

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    return df.sort_index()
