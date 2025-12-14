# Indian Options Backtesting System

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive Python-based backtesting framework for Indian options trading strategies on major indices like NIFTY, BANKNIFTY, FINNIFTY, and more. This tool allows traders and analysts to evaluate the performance of various options strategies with customizable parameters, risk management features, and machine learning evaluation.

## Description

The Indian Options Backtesting System provides a robust platform for backtesting options strategies on Indian stock market indices. It supports both predefined strategies (e.g., Long Straddle, Bull Call Spread) and custom user-defined strategies. The system fetches historical data from NSE and BSE, computes key performance metrics, applies stop-loss and target-profit mechanisms, and includes machine learning models to predict daily PnL directions.

Key capabilities include:

- Backtesting on multiple Indian indices
- Support for intraday, positional, and swing trading
- Performance metrics calculation (Sharpe Ratio, Sortino Ratio, Win Rate, etc.)
- ML evaluation using various classifiers
- Visualization of PnL charts
- CSV export of results

## Features

- **Multi-Index Support**: Backtest on NIFTY 50, NIFTY BANK, FINNIFTY, NIFTY NEXT 50, NIFTY MIDCAP SELECT, SENSEX, and BANKEX
- **Strategy Variety**: Predefined strategies like Long Straddle, Long Strangle, Bull Call Spread, Bear Put Spread, and custom strategies
- **Risk Management**: Configurable stop-loss and target-profit percentages
- **Trading Types**: Intraday, Positional, and Swing trading simulations
- **Performance Analysis**: Comprehensive metrics including Sharpe Ratio, Sortino Ratio, Max Drawdown, Win Rate
- **Machine Learning Integration**: Evaluate strategy performance using Random Forest, SVM, MLP, and other ML models
- **Data Visualization**: Generate and save PnL charts with customizable styling
- **Export Capabilities**: Save backtest results, ML evaluations, and charts to CSV and PNG files
- **User-Friendly Interface**: Interactive command-line interface for easy parameter input

## Installation

### Prerequisites

- Python 3.7 or higher
- Internet connection for data fetching

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/indian-options-backtesting.git
   cd indian-options-backtesting
   ```

2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Create a virtual environment:

   ```bash
   python -m venv backtest_env
   source backtest_env/bin/activate  # On Windows: backtest_env\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

Run the script from the command line:

```bash
python sample_backtest.py
```

### Input Parameters

- **Index Symbol**: Choose from supported indices (NIFTY, BANKNIFTY, FINNIFTY, etc.)
- **Date Range**: Specify start and end dates for backtesting
- **Time Range**: Set market hours (default: 09:15 to 15:30)
- **Strategy**: Select predefined or create custom strategy
- **Lot Size**: Number of contracts (default values provided for each index)
- **Stop Loss %**: Percentage for stop-loss (optional)
- **Target Profit %**: Percentage for target profit (optional)
- **Trading Type**: Intraday, Positional, or Swing

### Example Usage

```
=========== INDIAN OPTIONS BACKTESTING SYSTEM  ===========
Enter Index Symbol (NIFTY / BANKNIFTY / FINNIFTY / NIFTY NEXT 50 / NIFTY MIDCAP SELECT / SENSEX / BANKEX): NIFTY
Enter Start Date (YYYY-MM-DD): 2023-01-01
Enter End Date (YYYY-MM-DD): 2023-12-31
Enter Start Time (HH:MM, e.g., 09:15): 09:15
Enter End Time (HH:MM, e.g., 15:30): 15:30
Enter Strategy Type (default / custom ): default
Enter Default Strategy (long_straddle / long_strangle / bull_call_spread / bear_put_spread): long_straddle
Enter Call Premium Price (₹): 150
Enter Put Premium Price (₹): 140
Default Lot Size for ^NSEI: 75
Enter Lot Size or press Enter to use default (75):
Enter Stop Loss Percentage (e.g., 0.2 for 20%) or press Enter to skip: 0.1
Enter Trading Type (intraday / positional / swing): intraday
Enter Target Profit Percentage (e.g., 0.2 for 20%) or press Enter to skip: 0.15
```

## Supported Strategies

### Predefined Strategies

1. **Long Straddle**: Buy both call and put options at the same strike
2. **Long Strangle**: Buy OTM call and OTM put at different strikes
3. **Bull Call Spread**: Buy lower strike call, sell higher strike call
4. **Bear Put Spread**: Buy higher strike put, sell lower strike put

### Custom Strategies

Create your own strategies by specifying:

- Strategy name
- Option type (call/put)
- Action (buy/sell)
- Premium price

## Output

The system generates several outputs:

### Performance Report

- Period
- Final Cumulative PnL
- Max Drawdown
- Sharpe Ratio
- Sortino Ratio
- Win Rate

### Monthly PnL Report

Breakdown of profit/loss by month

### ML Evaluation Report

Accuracy, Precision, and Recall for various ML models predicting daily PnL direction

### Charts

- Cumulative PnL chart with green/red fill for profits/losses
- Annotations for key metrics

### Files Generated

- `{symbol}_{strategy}_{trading_type}_SL_TP_backtest.csv`: Backtest results
- `{symbol}_{strategy}_{trading_type}_ml_evaluation.csv`: ML evaluation results
- `{symbol}_{strategy}_{trading_type}_pnl_chart.png`: PnL chart

Example output files:

- `^NSEI_Long Straddle_intraday_SL_TP_backtest.csv`
- `^NSEI_Long Straddle_intraday_pnl_chart.png`

## Dependencies

- `nsepython`: For fetching NSE index data
- `bse`: For fetching BSE index data
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `matplotlib`: Data visualization
- `scikit-learn`: Machine learning models
- `tabulate`: Table formatting for console output

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This tool is for educational and research purposes only. It should not be used as the sole basis for real trading decisions. Past performance does not guarantee future results. Always consult with a financial advisor before making investment decisions.
