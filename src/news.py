\
from __future__ import annotations
import datetime as dt
import json
from typing import Dict, Any, List
import pandas as pd
import requests
import feedparser

def utc_now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def gdelt_search(query: str, max_records: int = 50, lookback_days: int = 2) -> List[Dict[str, Any]]:
    """
    GDELT 2.1 DOC API: https://blog.gdeltproject.org/gdelt-2-1-api-debuts/
    Returns a list of articles (title, url, domain, seendate, snippet if available).
    """
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=lookback_days)
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max_records,
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
        "formatdatetime": "1"
    }
    r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    return js.get("articles", [])

def google_news_rss(query: str) -> List[Dict[str, Any]]:
    """
    Google News RSS (unofficial, but simple). Consider throttling.
    """
    url = "https://news.google.com/rss/search"
    params = {"q": query, "hl": "en-GB", "gl": "GB", "ceid": "GB:en"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    feed = feedparser.parse(r.text)
    out = []
    for e in feed.entries[:50]:
        out.append({
            "title": e.get("title"),
            "url": e.get("link"),
            "publisher": e.get("source", {}).get("title") if isinstance(e.get("source"), dict) else None,
            "published": e.get("published")
        })
    return out

def build_queries(commodities: Dict[str, Any]) -> Dict[str, str]:
    """
    Map commodity name to a search query string.
    You can customize per commodity for better precision.
    """
    names = []
    for group, lst in commodities.get("yahoo", {}).items():
        for it in lst:
            names.append(it["name"])
    for it in commodities.get("tushare", []):
        names.append(it["name"])
    for it in commodities.get("eia", []):
        names.append(it["name"])
    # de-dup while preserving order
    seen = set()
    out = {}
    for n in names:
        if n not in seen:
            seen.add(n)
            # simple query heuristic
            out[n] = f'"{n}" (price OR futures OR spot OR supply OR demand OR inventory OR OPEC OR harvest)'
    return out

def fetch_all_news(config: Dict[str, Any], commodities: Dict[str, Any]) -> pd.DataFrame:
    ts = utc_now_iso()
    rows = []
    qmap = build_queries(commodities)
    gdelt_cfg = config.get("news", {}).get("gdelt", {})
    max_records = int(gdelt_cfg.get("max_records", 50))
    lookback_days = int(gdelt_cfg.get("lookback_days", 2))

    google_enabled = bool(config.get("news", {}).get("google_rss", {}).get("enabled", True))

    for commodity, q in qmap.items():
        # GDELT
        try:
            articles = gdelt_search(q, max_records=max_records, lookback_days=lookback_days)
            for a in articles:
                rows.append({
                    "ts_utc": ts,
                    "source": "gdelt",
                    "commodity": commodity,
                    "query": q,
                    "title": a.get("title"),
                    "url": a.get("url"),
                    "publisher": a.get("domain"),
                    "published_utc": a.get("seendate"),
                    "snippet": a.get("excerpt") or a.get("description")
                })
        except Exception as e:
            rows.append({
                "ts_utc": ts, "source": "gdelt", "commodity": commodity, "query": q,
                "title": None, "url": f"ERROR:{e}", "publisher": None, "published_utc": None, "snippet": None
            })

        # Google RSS (optional)
        if google_enabled:
            try:
                entries = google_news_rss(q)
                for e in entries:
                    rows.append({
                        "ts_utc": ts,
                        "source": "google_rss",
                        "commodity": commodity,
                        "query": q,
                        "title": e.get("title"),
                        "url": e.get("url"),
                        "publisher": e.get("publisher"),
                        "published_utc": e.get("published"),
                        "snippet": None
                    })
            except Exception as e:
                rows.append({
                    "ts_utc": ts, "source": "google_rss", "commodity": commodity, "query": q,
                    "title": None, "url": f"ERROR:{e}", "publisher": None, "published_utc": None, "snippet": None
                })

    return pd.DataFrame(rows)
