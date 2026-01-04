import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from src.utils import load_yaml
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.analytics_engine import compute_analytics
from src.analytics_core import rsi_series, macd_series
st.sidebar.header("Indicator Controls")

if st.sidebar.button("Reload indicator config"):
    st.cache_data.clear()
    st.rerun()


DB = "data/commodities.db"

st.set_page_config(page_title="Commodity Tracker", layout="wide")
st.title("Commodity Tracker (Free APIs)")

indicator_cfg = load_yaml("config/indicators.yaml").get("indicators", {})
# -----------------------------
# Helpers
# -----------------------------
def indicator_enabled(name):
    return indicator_cfg.get(name, {}).get("enabled", False)


def arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert common problematic columns to Arrow-safe display types."""
    out = df.copy()
    for c in out.columns:
        if np.issubdtype(out[c].dtype, np.datetime64):
            out[c] = out[c].astype(str)
    return out


def build_meta_map(commodities_cfg: dict) -> pd.DataFrame:
    """Create a lookup table: symbol -> (base_name, category, exchange, proxy_quality)."""
    rows = []
    for group, lst in commodities_cfg.get("yahoo", {}).items():
        for it in lst:
            rows.append({
                "symbol": it.get("symbol"),
                "commodity_name": it.get("name"),
                "category": it.get("category", group),
                "exchange": it.get("exchange", ""),
                "proxy_quality": it.get("proxy_quality", "")
            })
    return pd.DataFrame(rows).dropna(subset=["symbol"]).drop_duplicates(subset=["symbol"])


def compute_returns(pr: pd.Series, windows=(1, 7, 30)) -> dict:
    """Compute simple % returns over N trading days (row-based, assuming daily history)."""
    pr = pr.dropna()
    out = {}
    for w in windows:
        if len(pr) > w:
            out[f"ret_{w}d_pct"] = (pr.iloc[-1] / pr.iloc[-(w+1)] - 1.0) * 100.0
        else:
            out[f"ret_{w}d_pct"] = np.nan
    return out


def compute_vol_metrics(pr: pd.Series) -> dict:
    """Compute realized vol & drawdown using daily close series."""
    pr = pr.dropna()
    if len(pr) < 10:
        return {"vol_30d_ann_pct": np.nan, "vol_90d_ann_pct": np.nan, "max_drawdown_pct": np.nan}

    rets = pr.pct_change(fill_method=None).dropna()
    # annualize with 252 trading days
    def ann_vol(x):
        return x.std(ddof=0) * np.sqrt(252) * 100.0 if len(x) > 2 else np.nan

    vol_30 = ann_vol(rets.tail(30))
    vol_90 = ann_vol(rets.tail(90))

    # max drawdown on price series
    cummax = pr.cummax()
    dd = (pr / cummax - 1.0) * 100.0
    mdd = dd.min() if len(dd) else np.nan

    return {"vol_30d_ann_pct": vol_30, "vol_90d_ann_pct": vol_90, "max_drawdown_pct": mdd}


def compute_ma_signals(pr: pd.Series, windows=(20, 60, 200)) -> dict:
    pr = pr.dropna()
    out = {}
    if pr.empty:
        for w in windows:
            out[f"sma_{w}"] = np.nan
            out[f"trend_vs_sma_{w}"] = ""
        return out

    last = pr.iloc[-1]
    for w in windows:
        sma = pr.tail(w).mean() if len(pr) >= w else np.nan
        out[f"sma_{w}"] = sma
        if pd.isna(sma):
            out[f"trend_vs_sma_{w}"] = ""
        else:
            out[f"trend_vs_sma_{w}"] = "Bullish" if last > sma else "Bearish"
    return out


def latest_by_symbol(prices_df: pd.DataFrame) -> pd.DataFrame:
    tmp = prices_df.copy()
    tmp["ts"] = pd.to_datetime(tmp["ts_utc"], errors="coerce", utc=True)
    tmp = tmp.dropna(subset=["ts"])
    tmp = tmp.sort_values("ts")
    return tmp.groupby(["symbol"], as_index=False).tail(1).drop(columns=["ts"])


# -----------------------------
# Data loaders (Arrow-safe at boundary)
# -----------------------------

@st.cache_data(ttl=60)
def load_prices():
    con = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT * FROM prices ORDER BY ts_utc ASC", con)
    con.close()

    # Keep as strings for Streamlit safety; convert to datetime only locally when needed
    df["ts_utc"] = df["ts_utc"].astype(str)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df

@st.cache_data(ttl=60)
def load_news():
    con = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT * FROM news ORDER BY published_utc DESC, ts_utc DESC", con)
    con.close()

    for col in ["ts_utc", "published_utc"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


commodities_cfg = load_yaml("commodities.yaml")
meta_df = build_meta_map(commodities_cfg)

prices = load_prices()
news = load_news()

# -----------------------------
# Overview: Latest + analytics table
# -----------------------------
st.subheader("Latest prices + analytics")

if prices.empty:
    st.warning(
        "No price data yet. Run `python run_update.py` and (optionally) `python backfill_history.py`."
    )
else:
    # Prepare data for analytics engine
    p = prices.copy()
    p["ts"] = pd.to_datetime(p["ts_utc"], errors="coerce", utc=True)
    p = p.dropna(subset=["ts", "price"])
    p = p.sort_values(["symbol", "ts"])

    analytics = compute_analytics(p, meta_df)
    analytics = analytics.merge(meta_df, on="symbol", how="left")

    preferred_cols = [
        "commodity_name", "category",
        "last_price",
        "ret_1d_pct", "ret_7d_pct", "ret_30d_pct",
        "trend_regime",
        "vol_regime",
        "vol_30d_ann", "vol_90d_ann",
        "max_drawdown",
        "momentum_12_1",
        "trend_pnl",
    ]

    show_cols = [c for c in preferred_cols if c in analytics.columns]

    st.dataframe(
        arrow_safe(
            analytics[show_cols].sort_values(
                ["category", "commodity_name"], na_position="last"
            )
        ),
        width="stretch"
    )


# -----------------------------
# Price history chart (with MAs)
# -----------------------------

st.subheader("Price history & technical indicators")

if not prices.empty:

    # ---- symbol selector (unchanged logic) ----
    sym_name_map = (
        meta_df[["symbol", "commodity_name"]]
        .dropna()
        .drop_duplicates()
        .set_index("symbol")["commodity_name"]
        .to_dict()
    )

    available_symbols = sorted(prices["symbol"].dropna().unique())

    display_labels = {
        sym: f"{sym_name_map.get(sym, sym)} ({sym})"
        for sym in available_symbols
    }
    label_to_symbol = {v: k for k, v in display_labels.items()}

    selected_label = st.selectbox("Commodity", list(display_labels.values()))
    sel_symbol = label_to_symbol[selected_label]

    # ---- prepare data ----
    df = prices[prices["symbol"] == sel_symbol].copy()
    df["ts"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["ts", "price"]).sort_values("ts")

    if len(df) < 30:
        st.info("Not enough data points yet.")
    else:
        # ---- moving averages ----
        df["SMA20"] = df["price"].rolling(20).mean()
        df["SMA60"] = df["price"].rolling(60).mean()
        df["SMA200"] = df["price"].rolling(200).mean()

        # ---- RSI ----
        if indicator_enabled("rsi"):
            rsi_window = indicator_cfg["rsi"]["params"]["window"]
            df["RSI"] = rsi_series(df["price"], rsi_window)

        # ---- MACD ----
        if indicator_enabled("macd"):
            macd_params = indicator_cfg["macd"]["params"]
            macd, macd_sig, macd_hist = macd_series(
                df["price"],
                fast=macd_params["fast"],
                slow=macd_params["slow"],
                signal=macd_params["signal"],
            )

            df["MACD"] = macd
            df["MACD_signal"] = macd_sig
            df["MACD_hist"] = macd_hist

        # ---- build figure with subplots ----
        from plotly.subplots import make_subplots
        rows = 1 + int(indicator_enabled("rsi")) + int(indicator_enabled("macd"))

        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5] + [0.25] * (rows - 1),
        )

        # ---- PRICE PANEL ----
        fig.add_scatter(
            x=df["ts"], y=df["price"],
            name="Price", row=1, col=1
        )
        fig.add_scatter(x=df["ts"], y=df["SMA20"], name="SMA20", row=1, col=1)
        fig.add_scatter(x=df["ts"], y=df["SMA60"], name="SMA60", row=1, col=1)
        fig.add_scatter(x=df["ts"], y=df["SMA200"], name="SMA200", row=1, col=1)

        current_row = 2

        # ---- RSI PANEL ----
        if indicator_enabled("rsi"):
            fig.add_scatter(
                x=df["ts"], y=df["RSI"],
                name="RSI", row=current_row, col=1
            )
            fig.add_hline(y=70, line_dash="dash", row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", row=current_row, col=1)
            current_row += 1

        # ---- MACD PANEL ----
        if indicator_enabled("macd"):
            fig.add_scatter(
                x=df["ts"], y=df["MACD"],
                name="MACD", row=current_row, col=1
            )
            fig.add_scatter(
                x=df["ts"], y=df["MACD_signal"],
                name="Signal", row=current_row, col=1
            )

        fig.update_layout(
            height=300 + rows * 200,
            showlegend=True,
            title=f"{display_labels[sel_symbol]} – Price & Indicators",
        )

        st.plotly_chart(fig, width="stretch")


# -----------------------------
# ---- Correlation heatmap ----
# -----------------------------
st.subheader("Correlation heatmap (90-day returns)")

# Wide returns by symbol
wide = (
    p
    .groupby(["ts", "symbol"], as_index=False)
    .agg({"price": "last"})   # <- key fix
    .pivot(index="ts", columns="symbol", values="price")
    .pct_change(fill_method=None)
    .tail(90)
)


corr = wide.corr()

# Rename symbols -> commodity names for display
sym_to_name = dict(
    zip(meta_df["symbol"], meta_df["commodity_name"])
)

corr_display = corr.rename(
    index=sym_to_name,
    columns=sym_to_name
)

fig_corr = px.imshow(
    corr_display,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu",
    zmin=-1,
    zmax=1,
    title="Commodity return correlations (90D)"
)

st.plotly_chart(fig_corr, width="stretch")

# -----------------------------------------
# PCA
# -----------------------------------------
st.subheader("PCA – Market Factors")

# Build clean returns matrix
# Aggregate duplicates safely (important!)
wide_prices = (
    p
    .groupby(["ts", "symbol"], as_index=False)
    .agg({"price": "last"})
    .pivot(index="ts", columns="symbol", values="price")
)

# Compute returns without filling
returns = wide_prices.pct_change(fill_method=None)

# Use last N days only (key fix)
WINDOW = 60   # try 30 if still sparse
returns = returns.tail(WINDOW)

# Drop symbols with too many missing values
min_non_na = int(0.7 * WINDOW)  # require 70% presence
returns = returns.loc[:, returns.count() >= min_non_na]

# Now drop remaining rows with any NaN
returns = returns.dropna(how="any")

if returns.shape[1] < 3 or returns.shape[0] < 10:
    st.info("Not enough overlapping data for PCA yet.")
else:
    # -----------------------------------------
    # Standardize returns
    # -----------------------------------------
    scaler = StandardScaler()
    X = scaler.fit_transform(returns.values)

    # Number of components
    n_components = min(5, X.shape[1])
    pca = PCA(n_components=n_components)
    factors = pca.fit_transform(X)

    # -----------------------------------------
    # Explained variance
    # -----------------------------------------
    ev = pd.DataFrame({
        "Factor": [f"PC{i+1}" for i in range(n_components)],
        "Explained Variance (%)": pca.explained_variance_ratio_ * 100
    })

    fig_ev = px.bar(
        ev,
        x="Factor",
        y="Explained Variance (%)",
        title="PCA – Explained Variance"
    )
    st.plotly_chart(fig_ev, width="stretch")

    # -----------------------------------------
    # Factor loadings
    # -----------------------------------------
    loadings = pd.DataFrame(
        pca.components_.T,
        index=returns.columns,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    # Rename symbols → commodity names
    sym_to_name = dict(zip(meta_df["symbol"], meta_df["commodity_name"]))
    loadings.index = loadings.index.map(lambda x: sym_to_name.get(x, x))

    fig_load = px.imshow(
        loadings,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="PCA – Factor Loadings"
    )
    st.plotly_chart(fig_load, width="stretch")

    # -----------------------------------------
    # Factor time series
    # -----------------------------------------
    factor_ts = pd.DataFrame(
        factors,
        index=returns.index,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    fig_fac = px.line(
        factor_ts,
        title="PCA – Factor Returns Over Time"
    )
    st.plotly_chart(fig_fac, width="stretch")

# -----------------------------
# News
# -----------------------------

st.subheader("News")
if news.empty:
    st.info("No news yet. Run `python run_update.py` first.")
else:
    commodity_n = st.selectbox(
        "News commodity",
        sorted(news["commodity"].dropna().unique()),
        key="news_commodity"
    )
    ndf = news[news["commodity"] == commodity_n].copy().head(50)

    for _, r in ndf.iterrows():
        st.markdown(
            f"- **{r.get('title', '(no title)')}**  \n"
            f"  {r.get('publisher', '')} | {r.get('published_utc', '')}  \n"
            f"  {r.get('url', '')}"
        )
