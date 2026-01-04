I’ve written this as if the repo were public or assessed — clean, readable, and honest about limitations.

### Commodity Tracker

A modular, configuration-driven commodity analytics & research dashboard built with free data sources.

The project collects daily commodity prices, stores them locally, and provides a Streamlit-based analytics interface with technical indicators, risk metrics, correlations, and factor analysis (PCA).
All analytics are pluggable, configurable, and easily extensible.

## Features
# Data

Daily prices from free APIs (Yahoo Finance proxies)

Local SQLite storage

One-off historical backfill + daily incremental updates

# Analytics (Modular)

Returns (1D / 7D / 30D)

Momentum (12–1)

Trend regime (SMA-based)

Volatility & drawdowns

RSI & MACD

Correlation heatmap

PCA factor analysis (rolling window)

# Architecture

Config-driven indicators (YAML)

Indicators can be enabled/disabled without code changes

Per-category indicator controls

Clear separation of:

data ingestion

analytics logic

visualization

# UI

Streamlit dashboard

Interactive symbol selection

Multi-panel technical charts

Hot-reload indicator configuration

## Getting Started
# 1️ Install dependencies
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2️ Fetch data
python run_update.py
python backfill_history.py   # optional (run once)

# 3️ Launch dashboard
streamlit run dashboard_streamlit.py

# Configuration

Indicators are controlled via:

config/indicators.yaml


You can:

enable/disable indicators

adjust parameters (RSI window, MACD periods, etc.)

apply per-category overrides

Changes can be reloaded from the Streamlit sidebar.