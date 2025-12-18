# core/strategy.py

from __future__ import annotations
import numpy as np
import pandas as pd


# =========================================================
# COMPETITIVE POSITIONING INDEX
# =========================================================
def compute_competitive_index(
    market_df: pd.DataFrame,
    sentiment_counts: dict,
    forecast_df: pd.DataFrame | None,
) -> float:
    """
    Computes a 0–100 Competitive Positioning Index.

    Components:
    - Market Momentum (35%)
    - Sentiment Strength (30%)
    - Forecast Direction (20%)
    - News Intensity (15%)

    Output:
    - Float value in range [0, 100]
    """

    if market_df is None or market_df.empty or "Close" not in market_df.columns:
        return 50.0  # neutral fallback

    market_df = market_df.dropna(subset=["Close"])

    # -----------------------------------------------------
    # 1. Market Momentum (35%)
    # -----------------------------------------------------
    price_momentum_score = 0.0
    if len(market_df) >= 2:
        start_price = market_df["Close"].iloc[0]
        end_price = market_df["Close"].iloc[-1]
        if start_price > 0:
            pct_change = (end_price - start_price) / start_price
            price_momentum_score = np.clip(pct_change * 100, -20, 20)

    # -----------------------------------------------------
    # 2. Sentiment Strength (30%)
    # -----------------------------------------------------
    pos = sentiment_counts.get("positive", 0)
    neg = sentiment_counts.get("negative", 0)
    neu = sentiment_counts.get("neutral", 0)
    total = pos + neg + neu

    if total == 0:
        sentiment_score = 0.0
    else:
        sentiment_score = ((pos - neg) / total) * 100
        sentiment_score = np.clip(sentiment_score, -30, 30)

    # -----------------------------------------------------
    # 3. Forecast Direction (20%)
    # -----------------------------------------------------
    forecast_score = 0.0
    if forecast_df is not None and not forecast_df.empty:
        try:
            y0 = forecast_df["yhat"].iloc[0]
            y1 = forecast_df["yhat"].iloc[-1]
            if y0 != 0:
                trend_pct = (y1 - y0) / y0
                forecast_score = np.clip(trend_pct * 100, -20, 20)
        except Exception:
            forecast_score = 0.0

    # -----------------------------------------------------
    # 4. News Intensity (15%)
    # -----------------------------------------------------
    news_intensity_score = np.clip(total * 2, 0, 30)

    # -----------------------------------------------------
    # Weighted Composite Index
    # -----------------------------------------------------
    raw_index = (
        (price_momentum_score * 0.35)
        + (sentiment_score * 0.30)
        + (forecast_score * 0.20)
        + (news_intensity_score * 0.15)
    )

    # Normalize to 0–100 scale with 50 as neutral
    competitive_index = np.clip(raw_index + 50, 0, 100)

    return round(float(competitive_index), 2)


# =========================================================
# STRATEGIC SIGNAL CLASSIFICATION
# =========================================================
def classify_strategic_signal(
    market_df: pd.DataFrame,
    sentiment_counts: dict,
    forecast_df: pd.DataFrame | None,
) -> dict:
    """
    Classifies current strategic condition into:
    - OPPORTUNITY
    - THREAT
    - MONITOR

    Returns a dict suitable for UI + alerts.
    """

    # Safety checks
    if market_df is None or market_df.empty or "Close" not in market_df.columns:
        return {
            "signal": "MONITOR",
            "confidence": "Low",
            "strength": 0,
            "reason": ["Insufficient market data"],
        }

    pos = sentiment_counts.get("positive", 0)
    neg = sentiment_counts.get("negative", 0)

    # -----------------------------------------------------
    # Recent Market Momentum (7-day)
    # -----------------------------------------------------
    recent_return = 0.0
    if len(market_df) >= 7:
        p0 = market_df["Close"].iloc[-7]
        p1 = market_df["Close"].iloc[-1]
        if p0 != 0:
            recent_return = (p1 - p0) / p0

    # -----------------------------------------------------
    # Forecast Trend
    # -----------------------------------------------------
    forecast_trend = 0.0
    if forecast_df is not None and not forecast_df.empty:
        try:
            forecast_trend = forecast_df["yhat"].iloc[-1] - forecast_df["yhat"].iloc[0]
        except Exception:
            forecast_trend = 0.0

    # -----------------------------------------------------
    # Decision Logic
    # -----------------------------------------------------
    if pos > neg and recent_return > 0 and forecast_trend > 0:
        return {
            "signal": "OPPORTUNITY",
            "confidence": "High",
            "strength": round(min((pos - neg) * 10 + recent_return * 100, 100), 1),
            "reason": [
                "Positive price momentum observed",
                "Bullish sentiment dominates news flow",
                "Forecast indicates upward trend",
            ],
        }

    if neg > pos and recent_return < 0:
        return {
            "signal": "THREAT",
            "confidence": "High",
            "strength": round(min((neg - pos) * 10 + abs(recent_return) * 100, 100), 1),
            "reason": [
                "Negative sentiment outweighs positive coverage",
                "Recent price decline detected",
                "Downside risk present in market behavior",
            ],
        }

    return {
        "signal": "MONITOR",
        "confidence": "Medium",
        "strength": round(abs(recent_return) * 100, 1),
        "reason": [
            "Conflicting or weak market signals",
            "No dominant sentiment direction",
            "Strategic patience recommended",
        ],
    }
