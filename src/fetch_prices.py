\
from __future__ import annotations
import json
import datetime as dt
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import requests
import yfinance as yf

def utc_now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def fetch_yahoo_quotes(items: List[Dict[str,str]]) -> pd.DataFrame:
    """
    Fetch last price for many Yahoo tickers. Uses yfinance download for efficiency.
    Returns rows with: ts_utc, source, market, commodity, symbol, price, currency, unit, extra_json
    """
    ts = utc_now_iso()
    symbols = [i["symbol"] for i in items]
    # download last 2 days so we can take the latest close/last
    data = yf.download(symbols, period="2d", interval="1d", group_by="ticker", threads=True, progress=False)
    rows = []
    for i in items:
        sym = i["symbol"]
        name = i["name"]
        try:
            if isinstance(data.columns, pd.MultiIndex):
                close = data[(sym, "Close")].dropna()
            else:
                close = data["Close"].dropna()
            price = float(close.iloc[-1]) if len(close) else None
        except Exception:
            price = None
        # yfinance currency is per-ticker metadata (extra call). Keep lightweight: omit unless needed.
        rows.append({
            "ts_utc": ts,
            "source": "yahoo_yfinance",
            "market": None,
            "commodity": name,
            "symbol": sym,
            "price": price,
            "currency": None,
            "unit": None,
            "extra_json": json.dumps({"kind": "futures_or_index_proxy"})
        })
    return pd.DataFrame(rows)

def fetch_eia_series(api_key: str, series_items: List[Dict[str,str]]) -> pd.DataFrame:
    """
    Fetch latest value from EIA v2 series endpoint.
    """
    if not api_key:
        return pd.DataFrame()
    ts = utc_now_iso()
    rows = []
    for item in series_items:
        name = item["name"]
        sid = item["series_id"]
        url = f"https://api.eia.gov/v2/seriesid/{sid}.json"
        try:
            r = requests.get(url, params={"api_key": api_key}, timeout=30)
            r.raise_for_status()
            js = r.json()
            # Typical structure: response -> data list with period/value
            data = js.get("response", {}).get("data", [])
            latest = data[0] if data else None
            val = latest.get("value") if latest else None
            unit = latest.get("units") or js.get("response", {}).get("units")
            rows.append({
                "ts_utc": ts,
                "source": "eia",
                "market": "US",
                "commodity": name,
                "symbol": sid,
                "price": float(val) if val is not None else None,
                "currency": "USD",
                "unit": unit,
                "extra_json": json.dumps({"period": latest.get("period") if latest else None})
            })
        except Exception as e:
            rows.append({
                "ts_utc": ts, "source": "eia", "market": "US",
                "commodity": name, "symbol": sid, "price": None,
                "currency": "USD", "unit": None,
                "extra_json": json.dumps({"error": str(e)})
            })
    return pd.DataFrame(rows)

def fetch_tushare_futures(token: str, items: List[Dict[str,str]]) -> pd.DataFrame:
    """
    Minimal Tushare example: if you have a token, you can call Tushare Pro endpoints.
    Tushare endpoints vary; this template uses the generic 'api.tushare.pro' method pattern.
    You should adjust 'api_name' + fields according to your subscription/free-tier access.
    """
    if not token:
        return pd.DataFrame()

    ts = utc_now_iso()
    rows = []
    url = "http://api.tushare.pro"
    headers = {"Content-Type": "application/json"}
    for item in items:
        name = item["name"]
        code = item["code"]
        payload = {
            "api_name": "fut_daily",     # common endpoint; may require adjustments
            "token": token,
            "params": {"ts_code": code, "limit": 1},
            "fields": "ts_code,trade_date,close,settle,vol,amount"
        }
        try:
            r = requests.post(url, data=json.dumps(payload), headers=headers, timeout=30)
            r.raise_for_status()
            js = r.json()
            data = js.get("data", {})
            fields = data.get("fields", [])
            items_ = data.get("items", [])
            latest = dict(zip(fields, items_[0])) if items_ else {}
            price = latest.get("close") or latest.get("settle")
            rows.append({
                "ts_utc": ts,
                "source": "tushare",
                "market": "CN",
                "commodity": name,
                "symbol": code,
                "price": float(price) if price is not None else None,
                "currency": "CNY",
                "unit": None,
                "extra_json": json.dumps(latest)
            })
        except Exception as e:
            rows.append({
                "ts_utc": ts, "source": "tushare", "market": "CN",
                "commodity": name, "symbol": code, "price": None,
                "currency": "CNY", "unit": None,
                "extra_json": json.dumps({"error": str(e)})
            })
    return pd.DataFrame(rows)

def fetch_all_prices(config: Dict[str, Any], commodities: Dict[str, Any]) -> pd.DataFrame:
    dfs = []
    yahoo_items = []
    for group, lst in commodities.get("yahoo", {}).items():
        for it in lst:
            yahoo_items.append({"name": it["name"], "symbol": it["symbol"]})
    if yahoo_items:
        dfs.append(fetch_yahoo_quotes(yahoo_items))

    dfs.append(fetch_eia_series(config.get("eia_api_key",""), commodities.get("eia", [])))
    dfs.append(fetch_tushare_futures(config.get("tushare_token",""), commodities.get("tushare", [])))

    out = pd.concat([d for d in dfs if d is not None and not d.empty], ignore_index=True) if any((d is not None and not d.empty) for d in dfs) else pd.DataFrame()
    return out
