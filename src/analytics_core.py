import pandas as pd
import numpy as np


# -------------------------
# Returns & momentum
# -------------------------

def compute_returns(pr: pd.Series, windows=(1, 7, 30, 63, 126)):
    pr = pr.dropna()
    out = {}
    for w in windows:
        out[f"ret_{w}d"] = (
            pr.iloc[-1] / pr.iloc[-(w + 1)] - 1.0
        ) if len(pr) > w else np.nan
    return out


def momentum_12_1(pr: pd.Series):
    pr = pr.dropna()
    if len(pr) < 252:
        return np.nan
    return pr.iloc[-21] / pr.iloc[-252] - 1.0


# -------------------------
# Trend regime
# -------------------------

def trend_regime(pr: pd.Series):
    pr = pr.dropna()
    if len(pr) < 200:
        return "Unknown"

    sma50 = pr.rolling(50).mean().iloc[-1]
    sma200 = pr.rolling(200).mean().iloc[-1]
    last = pr.iloc[-1]

    if last > sma200 and sma50 > sma200:
        return "Bull"
    if last < sma200 and sma50 < sma200:
        return "Bear"
    return "Sideways"


# -------------------------
# Volatility & risk
# -------------------------

def realized_vol(pr: pd.Series, window):
    r = pr.pct_change(fill_method=None).dropna()
    if len(r) < window:
        return np.nan
    return r.tail(window).std() * np.sqrt(252)


def volatility_regime(pr: pd.Series):
    v30 = realized_vol(pr, 30)
    v90 = realized_vol(pr, 90)
    if pd.isna(v30) or pd.isna(v90):
        return "Unknown"
    if v30 > 1.5 * v90:
        return "High"
    if v30 < 0.7 * v90:
        return "Low"
    return "Normal"


def max_drawdown(pr: pd.Series):
    cum_max = pr.cummax()
    dd = pr / cum_max - 1.0
    return dd.min()


# -------------------------
# Simple backtest (trend)
# -------------------------

def backtest_trend(pr: pd.Series):
    pr = pr.dropna()
    if len(pr) < 200:
        return np.nan

    sma200 = pr.rolling(200).mean()
    signal = (pr > sma200).shift(1)
    returns = pr.pct_change(fill_method=None)

    strat = returns * signal
    return strat.cumsum().iloc[-1]


# -------------------------
# RSI
# -------------------------

def compute_rsi(pr: pd.Series, window: int = 14):
    pr = pr.dropna()
    if len(pr) < window + 1:
        return np.nan

    delta = pr.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.iloc[-1]


def rsi_series(pr: pd.Series, window: int = 14):
    pr = pr.dropna()
    delta = pr.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


# -------------------------
# MACD
# -------------------------
def compute_macd(pr: pd.Series, fast=12, slow=26, signal=9):
    pr = pr.dropna()
    if len(pr) < slow + signal:
        return np.nan, np.nan

    ema_fast = pr.ewm(span=fast, adjust=False).mean()
    ema_slow = pr.ewm(span=slow, adjust=False).mean()

    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()

    return macd.iloc[-1], signal_line.iloc[-1]


def macd_series(pr: pd.Series, fast=12, slow=26, signal=9):
    pr = pr.dropna()

    ema_fast = pr.ewm(span=fast, adjust=False).mean()
    ema_slow = pr.ewm(span=slow, adjust=False).mean()

    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal

    return macd, macd_signal, macd_hist
