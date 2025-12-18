from __future__ import annotations

import feedparser
import re


def clean(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<.*?>", "", text)).strip()


def fetch_news(company: str, limit: int = 12):
    """
    Google News RSS fetcher with richer fields.
    Returns: title, link, source, published, description
    """
    url = f"https://news.google.com/rss/search?q={company}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)

    news = []
    for e in feed.entries[:limit]:
        title = clean(getattr(e, "title", ""))
        link = getattr(e, "link", "#")
        published = clean(getattr(e, "published", "") or getattr(e, "updated", ""))
        summary = clean(getattr(e, "summary", ""))

        # feedparser often includes source in e.source.title
        source = ""
        try:
            source = clean(getattr(getattr(e, "source", None), "title", ""))  # type: ignore
        except Exception:
            source = ""

        news.append(
            {
                "title": title,
                "link": link,
                "published": published,
                "source": source,
                "description": summary,
            }
        )

    return news
