# core/alerts.py

import json
import requests
import streamlit as st
from datetime import datetime


# --------------------------------------------------
# BUILD ALERT OBJECT (USED BY app.py UI + LOGIC)
# --------------------------------------------------
def build_alert(
    company: str,
    ticker: str,
    sentiment_counts: dict,
    strategic: dict,
) -> dict:
    """
    Builds a normalized alert object for UI + Slack.
    """

    signal = strategic.get("strategic_signal", {}).get("signal", "NEUTRAL")

    alert_type = (
        "OPPORTUNITY" if signal == "OPPORTUNITY"
        else "THREAT" if signal == "THREAT"
        else "NEUTRAL"
    )

    return {
        "alert_type": alert_type,
        "company": company,
        "ticker": ticker,
        "timestamp": datetime.utcnow().isoformat(),
        "sentiment": sentiment_counts,
        "strategic_action": signal,
        "message": f"Strategic signal generated for {company} ({ticker})",
    }


# --------------------------------------------------
# SEND SLACK MESSAGE (INCOMING WEBHOOK)
# --------------------------------------------------
def send_slack(payload: dict) -> bool:
    """
    Sends a Slack message via Incoming Webhook.
    Expects payload = {"text": "..."} or Block Kit payload.
    """

    webhook_url = st.secrets.get("SLACK_WEBHOOK_URL")

    if not webhook_url:
        return False

    try:
        response = requests.post(
            webhook_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=10,
        )

        return response.status_code == 200

    except Exception:
        return False
