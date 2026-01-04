\
from __future__ import annotations
from src.utils import load_yaml
from src.fetch_prices import fetch_all_prices
from src.news import fetch_all_news
from src.storage import connect, init_db, upsert_df, export_latest

def main():
    config = load_yaml("config.yaml")
    commodities = load_yaml("commodities.yaml")

    con = connect()
    init_db(con)

    prices = fetch_all_prices(config, commodities)
    if not prices.empty:
        upsert_df(con, prices, "prices")

    news = fetch_all_news(config, commodities)
    if not news.empty:
        upsert_df(con, news, "news")

    export_latest(con)
    con.close()
    print("Update complete. See data/exports/ for CSVs and data/commodities.db for SQLite.")

if __name__ == "__main__":
    main()
