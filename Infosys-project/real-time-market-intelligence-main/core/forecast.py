from __future__ import annotations

from prophet import Prophet
import pandas as pd


def run_prophet(df: pd.DataFrame, days: int = 7):
    """
    Runs Prophet forecast.
    - Robust Date handling (index or column).
    - Returns only last `days` rows of forecast columns.
    """

    if df is None or len(df) < 60 or "Close" not in df.columns:
        return None

    data = df.copy()

    # Ensure datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        if "Date" in data.columns:
            data["Date"] = pd.to_datetime(data["Date"])
            data = data.set_index("Date")
        else:
            data.index = pd.to_datetime(data.index, errors="coerce")

    data = data.dropna(subset=["Close"]).sort_index()

    if len(data) < 60:
        return None

    prophet_df = data.reset_index()
    # The reset index column name can be "Date" or something else.
    date_col = "Date" if "Date" in prophet_df.columns else prophet_df.columns[0]

    prophet_df = prophet_df[[date_col, "Close"]].rename(columns={date_col: "ds", "Close": "y"})
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], errors="coerce")
    prophet_df = prophet_df.dropna(subset=["ds", "y"])

    if len(prophet_df) < 60:
        return None

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
    )
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days)
