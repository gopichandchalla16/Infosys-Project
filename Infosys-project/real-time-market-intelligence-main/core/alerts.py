from __future__ import annotations

from datetime import datetime
import os
import requests


def build_alert(company: str, ticker: str, counts: dict, *, strategic: dict | None = None):
    """
    Builds a richer alert payload.
    Optionally accepts `strategic` dict: {"competitive_index": float, "strategic_signal": {...}}
    """
    counts = counts or {"positive": 0, "neutral": 0, "negative": 0}

    total = sum(counts.values()) or 1
    pos = counts.get("positive", 0) / total
    neg = counts.get("negative", 0) / total

    # Sentiment label logic
    if pos >= 0.60:
        alert_type = "📈 Bullish Sentiment"
        sentiment_score = round(pos, 2)
        strategic_action = "Consider opportunity: monitor momentum and confirm with forecast."
    elif neg >= 0.60:
        alert_type = "📉 Bearish Sentiment"
        sentiment_score = round(-neg, 2)
        strategic_action = "Risk alert: monitor downside factors and tighten watch on negatives."
    else:
        alert_type = "⚖️ Neutral Sentiment"
        sentiment_score = 0.0
        strategic_action = "Monitor: signals are mixed; wait for confirmation."

    payload = {
        "alert_type": alert_type,
        "company_name": company,
        "company_ticker": ticker,
        "sentiment_score": sentiment_score,
        "sentiment_breakdown": {
            "positive": counts.get("positive", 0),
            "neutral": counts.get("neutral", 0),
            "negative": counts.get("negative", 0),
        },
        "volatility_metric": "Medium",
        "signal_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "strategic_action": strategic_action,
    }

    # Optional strategic layer
    if strategic:
        payload["competitive_index"] = strategic.get("competitive_index")
        payload["strategic_signal"] = strategic.get("strategic_signal")

    return payload


def send_slack(alert: dict) -> bool:
    """
    Sends a structured Slack message via Incoming Webhook.
    Uses env var: SLACK_WEBHOOK_URL
    """
    webhook = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook:
        return False

    # Human-friendly summary text + structured fields
    text = (
        f"{alert.get('alert_type','Alert')} — {alert.get('company_name','')}"
        f" ({alert.get('company_ticker','')})"
    )

    try:
        r = requests.post(
            webhook,
            json={"text": text, "alert": alert},
            timeout=10,
        )
        return r.status_code == 200
    except Exception:
        return False
