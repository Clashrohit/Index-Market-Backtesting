# 🇮🇳 Indian Options Backtesting System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![NSE](https://img.shields.io/badge/Data-NSE%20%26%20BSE-orange)

A **comprehensive Python-based backtesting framework** for Indian stock index options such as **NIFTY**, **BANKNIFTY**, **FINNIFTY**, **SENSEX**, and others.  
It allows traders and analysts to simulate **Call/Put option strategies** with custom parameters like **Stop Loss**, **Target Profit**, and **Lot Size**, and evaluate the strategy’s performance over any historical period.

---

## 🧩 Features

- 📈 Fetches **historical index data** directly from **NSE** and **BSE**  
- ⚙️ Supports **Intraday**, **Positional**, and **Swing** trading strategies  
- 💰 Calculates key performance metrics:
  - Final Cumulative PnL  
  - Maximum Drawdown  
  - Sharpe Ratio  
  - Sortino Ratio  
  - Win Rate  
  - Monthly PnL Report  
- 📊 Generates PnL performance plots  
- 🧾 Exports results as CSV files  
- 🕒 Validates trading days (skips weekends/holidays)

---

## 🧠 Functions Overview

| Function | Description |
|-----------|--------------|
| `get_index_data()` | Fetches and cleans index historical data (NSE/BSE). |
| `option_backtest()` | Runs option strategy backtests and computes risk metrics. |
| `is_market_day()` | Checks whether a specific date is a valid market trading day. |
| `main()` | Provides an interactive CLI for user inputs and backtest execution. |

---

## 🚀 Usage

### 🔧 Prerequisites
Make sure you have the required Python packages installed:
```bash
pip install nsepython bse pandas numpy matplotlib
