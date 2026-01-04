\
from __future__ import annotations
import os
import sqlite3
from typing import Iterable, Dict, Any
import pandas as pd

def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/exports", exist_ok=True)

def connect(db_path: str = "data/commodities.db") -> sqlite3.Connection:
    ensure_dirs()
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL;")
    return con

def init_db(con: sqlite3.Connection) -> None:
    con.execute("""
    CREATE TABLE IF NOT EXISTS prices (
        ts_utc TEXT NOT NULL,
        source TEXT NOT NULL,
        market TEXT,
        commodity TEXT NOT NULL,
        symbol TEXT,
        price REAL,
        currency TEXT,
        unit TEXT,
        extra_json TEXT,
        PRIMARY KEY (ts_utc, source, commodity, symbol)
    );
    """)
    con.execute("""
    CREATE TABLE IF NOT EXISTS news (
        ts_utc TEXT NOT NULL,
        source TEXT NOT NULL,
        commodity TEXT NOT NULL,
        query TEXT,
        title TEXT,
        url TEXT,
        publisher TEXT,
        published_utc TEXT,
        snippet TEXT,
        PRIMARY KEY (ts_utc, source, commodity, url)
    );
    """)
    con.commit()

def upsert_df(con: sqlite3.Connection, df: pd.DataFrame, table: str) -> None:
    if df.empty:
        return

    cols = df.columns.tolist()
    placeholders = ",".join(["?"] * len(cols))
    colnames = ",".join(cols)

    sql = f"""
        INSERT OR IGNORE INTO {table} ({colnames})
        VALUES ({placeholders})
    """

    con.executemany(sql, df.itertuples(index=False, name=None))
    con.commit()


def export_latest(con: sqlite3.Connection) -> None:
    ensure_dirs()
    latest_prices = pd.read_sql_query("""
        SELECT p.*
        FROM prices p
        JOIN (
            SELECT commodity, symbol, MAX(ts_utc) AS max_ts
            FROM prices
            GROUP BY commodity, symbol
        ) m
        ON p.commodity = m.commodity AND IFNULL(p.symbol,'') = IFNULL(m.symbol,'') AND p.ts_utc = m.max_ts
        ORDER BY commodity;
    """, con)
    latest_news = pd.read_sql_query("""
        SELECT n.*
        FROM news n
        JOIN (
            SELECT commodity, url, MAX(ts_utc) AS max_ts
            FROM news
            GROUP BY commodity, url
        ) m
        ON n.commodity = m.commodity AND n.url = m.url AND n.ts_utc = m.max_ts
        ORDER BY commodity, published_utc DESC;
    """, con)

    latest_prices.to_csv("data/exports/latest_prices.csv", index=False)
    latest_news.to_csv("data/exports/latest_news.csv", index=False)
