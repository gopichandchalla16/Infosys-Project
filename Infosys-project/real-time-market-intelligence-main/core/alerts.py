# core/alerts.py

import json
import requests
import streamlit as st
from datetime import datetime


def build_alert(
    company: str,
    ticker: str,
    sentiment_counts: dict,
    strategic: dict,
) -> dict:
    """
    Builds the alert object used by the app UI.
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


def send_slack(payload: dict) -> bool:
    """
    Sends a Slack message via Incoming Webhook.
    """

    webhook_url = st.secrets.get("SLACK_WEBHOOK_URL")

    if not webhook_url:
        st.error("SLACK_WEBHOOK_URL not found in Streamlit secrets")
        return False

    try:
        response = requests.post(
            webhook_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=10,
        )

        if response.status_code != 200:
            st.error(f"Slack error {response.status_code}: {response.text}")
            return False

        return True

    except Exception as e:
        st.error(f"Slack request failed: {e}")
        return False
