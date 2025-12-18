# core/alerts.py

import json
import requests
import streamlit as st
from datetime import datetime


# ---------------------------------------------------------
# BUILD ALERT PAYLOAD (USED INTERNALLY BY app.py)
# ---------------------------------------------------------
def build_alert(
    company: str,
    ticker: str,
    sentiment_counts: dict,
    strategic: dict,
) -> dict:
    """
    Builds a normalized alert object consumed by UI + Slack.
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


# ---------------------------------------------------------
# SEND SLACK ALERT — BLOCK KIT (ENTERPRISE GRADE)
# ---------------------------------------------------------
def send_slack(payload: dict) -> bool:
    """
    Sends a Block Kit Slack message.
    Expects payload = { "blocks": [...] } OR { "text": "..."}
    """

    webhook_url = st.secrets.get("SLACK_WEBHOOK_URL")

    if not webhook_url:
        return False

    try:
        # If plain text is passed, wrap it safely
        if "blocks" not in payload and "text" in payload:
            payload = {
                "text": payload["text"]
            }

        response = requests.post(
            webhook_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        return response.status_code == 200

    except Exception:
        return False
