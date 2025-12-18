from __future__ import annotations

import streamlit as st
from transformers import pipeline


@st.cache_resource(show_spinner=False)
def load_finbert():
    return pipeline(
        "sentiment-analysis",
        model="yiyanghkust/finbert-tone",
        tokenizer="yiyanghkust/finbert-tone",
    )


def analyze_sentiment(news_items: list[dict]):
    """
    Returns:
      detailed: [{title, sentiment, confidence}]
      counts: {"positive": int, "neutral": int, "negative": int}
    """
    if not news_items:
        return [], {"positive": 0, "neutral": 0, "negative": 0}

    finbert = load_finbert()

    texts = [n.get("title", "") for n in news_items if n.get("title")]
    if not texts:
        return [], {"positive": 0, "neutral": 0, "negative": 0}

    results = finbert(texts)

    counts = {"positive": 0, "neutral": 0, "negative": 0}
    detailed = []

    for title, r in zip(texts, results):
        label = (r.get("label", "") or "neutral").lower()
        if label not in counts:
            label = "neutral"
        counts[label] += 1

        detailed.append(
            {
                "title": title,
                "sentiment": label,
                "confidence": round(float(r.get("score", 0.0)), 3),
            }
        )

    return detailed, counts
