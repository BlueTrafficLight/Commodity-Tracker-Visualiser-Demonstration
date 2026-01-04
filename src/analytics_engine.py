import pandas as pd
import numpy as np

from src.utils import load_yaml
from src.analytics_core import (
    compute_returns,
    momentum_12_1,
    trend_regime,
    realized_vol,
    volatility_regime,
    max_drawdown,
    backtest_trend,
    compute_rsi,
    rsi_series,
    compute_macd,
    macd_series
)

CONFIG_PATH = "config/indicators.yaml"


def load_config():
    cfg = load_yaml(CONFIG_PATH)
    return cfg.get("indicators", {}), cfg.get("categories", {})


def compute_analytics(prices_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    indicators_cfg, category_cfg = load_config()
    rows = []

    for sym, g in prices_df.groupby("symbol", sort=False):
        pr = g.sort_values("ts")["price"].dropna()
        if len(pr) < 30:
            continue

        meta_match = meta_df[meta_df["symbol"] == sym]

        if meta_match.empty:
            category = "uncategorized"
        else:
            category = meta_match.iloc[0].get("category", "uncategorized")

        row = {
            "symbol": sym,
            "last_price": pr.iloc[-1],
        }

        def enabled(ind):
            # global + category gate
            if not indicators_cfg.get(ind, {}).get("enabled", False):
                return False
            return category_cfg.get(category, {}).get(ind, True)

        # ---- RETURNS ----
        if enabled("returns"):
            ret = compute_returns(pr, windows=(1, 7, 30))
            row.update({
                "ret_1d_pct": ret["ret_1d"] * 100,
                "ret_7d_pct": ret["ret_7d"] * 100,
                "ret_30d_pct": ret["ret_30d"] * 100,
            })

        # ---- MOMENTUM ----
        if enabled("momentum"):
            m = momentum_12_1(pr)
            row["momentum_12_1"] = m * 100 if pd.notna(m) else np.nan

        # ---- TREND ----
        if enabled("trend"):
            row["trend_regime"] = trend_regime(pr)

        # ---- VOLATILITY ----
        if enabled("volatility"):
            short = indicators_cfg["volatility"]["params"]["vol_short"]
            long = indicators_cfg["volatility"]["params"]["vol_long"]

            row.update({
                "vol_30d_ann": realized_vol(pr, short),
                "vol_90d_ann": realized_vol(pr, long),
                "vol_regime": volatility_regime(pr),
                "max_drawdown": max_drawdown(pr) * 100,
            })

        # ---- BACKTEST (dependency enforced) ----
        if enabled("backtest") and enabled("trend"):
            row["trend_pnl"] = backtest_trend(pr)

                # ---- RSI ----
        if enabled("rsi"):
            rsi_window = indicators_cfg["rsi"]["params"]["window"]
            row["rsi"] = compute_rsi(pr, rsi_window)

        # ---- MACD ----
        if enabled("macd"):
            p = indicators_cfg["macd"]["params"]
            macd, macd_signal = compute_macd(
                pr,
                fast=p["fast"],
                slow=p["slow"],
                signal=p["signal"],
            )
            row["macd"] = macd
            row["macd_signal"] = macd_signal

        rows.append(row)

    return pd.DataFrame(rows)
