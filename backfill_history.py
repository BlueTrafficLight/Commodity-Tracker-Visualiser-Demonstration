from src.utils import load_yaml
from src.storage import connect, init_db, upsert_df
import yfinance as yf
import pandas as pd
import json
import time


# -------------------------------------------------
# Fetch historical daily prices from Yahoo Finance
# -------------------------------------------------

def backfill_yahoo_history(
    items,
    period="2y",      # change to "5y" or "max" if you want
    interval="1d",
    sleep_sec=0.5     # small delay to avoid Yahoo throttling
):
    rows = []

    for it in items:
        name = it["name"]
        symbol = it["symbol"]

        print(f"Fetching history for {name} ({symbol})")

        try:
            df = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
                group_by="column"
            )

            if df is None or df.empty:
                print(f"  ⚠ No data for {symbol}")
                continue

            # -----------------------------------------
            # Normalize Close column (robust handling)
            # -----------------------------------------

            if isinstance(df.columns, pd.MultiIndex):
                # Handle Yahoo MultiIndex outputs
                if ("Close", "") in df.columns:
                    close = df[("Close", "")]
                elif "Close" in df.columns.get_level_values(0):
                    close = df["Close"].iloc[:, 0]
                else:
                    print(f"  ⚠ No Close column for {symbol}")
                    continue
            else:
                if "Close" not in df.columns:
                    print(f"  ⚠ No Close column for {symbol}")
                    continue
                close = df["Close"]

            hist = pd.DataFrame({
                "ts_utc": pd.to_datetime(close.index, errors="coerce"),
                "price": pd.to_numeric(close.values, errors="coerce"),
            }).dropna()

            if hist.empty:
                print(f"  ⚠ Empty history after cleaning for {symbol}")
                continue

            for ts, price in zip(hist["ts_utc"], hist["price"]):
                rows.append({
                    "ts_utc": ts.strftime("%Y-%m-%dT00:00:00Z"),
                    "source": "yahoo_yfinance",
                    "market": None,
                    "commodity": name,
                    "symbol": symbol,
                    "price": float(price),
                    "currency": None,
                    "unit": None,
                    "extra_json": json.dumps({
                        "kind": "historical_backfill"
                    })
                })

            time.sleep(sleep_sec)

        except Exception as e:
            print(f"  ❌ Failed for {symbol}: {e}")

    return pd.DataFrame(rows)


# -------------------------------------------------
# Main entry point
# -------------------------------------------------

def main():
    commodities = load_yaml("commodities.yaml")

    # Collect Yahoo commodities only
    yahoo_items = []
    for group, lst in commodities.get("yahoo", {}).items():
        for it in lst:
            yahoo_items.append(it)

    if not yahoo_items:
        print("No Yahoo commodities found.")
        return

    hist_df = backfill_yahoo_history(
        yahoo_items,
        period="2y"   # change to "5y" or "max" if you want more history
    )

    if hist_df.empty:
        print("No historical data fetched.")
        return

    # -----------------------------------------
    # De-duplicate (CRITICAL for reruns)
    # -----------------------------------------
    hist_df = hist_df.drop_duplicates(
        subset=["ts_utc", "source", "commodity", "symbol"]
    )

    con = connect()
    init_db(con)

    upsert_df(con, hist_df, "prices")
    con.close()

    print(f"✅ Historical backfill complete: {len(hist_df)} rows inserted.")


if __name__ == "__main__":
    main()
