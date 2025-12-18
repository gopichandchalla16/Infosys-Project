import json
import requests
import streamlit as st


def send_slack(payload: dict) -> bool:
    """
    Sends a Slack message using Incoming Webhook.
    """

    webhook_url = st.secrets.get("SLACK_WEBHOOK_URL")

    if not webhook_url:
        st.error("SLACK_WEBHOOK_URL not found in Streamlit secrets.")
        return False

    try:
        response = requests.post(
            webhook_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=10,
        )

        if response.status_code != 200:
            st.error(f"Slack API error {response.status_code}: {response.text}")
            return False

        return True

    except Exception as e:
        st.error(f"Slack request failed: {e}")
        return False
