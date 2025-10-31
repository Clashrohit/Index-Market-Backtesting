# 🦅 Options Trading Backtesting Framework

A **modular and extensible backtesting framework** for **Options Trading in Python**.  
Supports advanced strategies like **Long Straddle, Long Strangle, Bull Call Spread, Bear Put Spread**, and **custom user-defined strategies** — complete with **PnL analytics**, **ML-based performance prediction**, and **risk metrics**.

---

## 🚀 Features

- ✅ Fetches NSE/BSE data automatically  
- ✅ Implements 5 popular option strategies  
- ✅ Simulates positional, intraday, and swing modes  
- ✅ Calculates key performance metrics: Cumulative PnL, Max Drawdown, Sharpe & Sortino Ratios, Win Rate  
- ✅ Machine Learning–based direction prediction  
- ✅ Beautiful tabular summaries and graphs  

---

## 🧩 Example Usage

```python
from strategies.bull_call_spread import BullCallSpread
from utils.data_fetcher import get_index_data
from utils.metrics import backtest_strategy

data = get_index_data("^NSEI", "2024-01-01", "2024-12-31")
strategy = BullCallSpread(buy_call_premium=120, sell_call_premium=80)

results, metrics = backtest_strategy(
    strategy, data, lot_size=50, premium=100,
    stop_loss_pct=0.1, target_profit_pct=0.2,
    trading_type="positional"
)
```

---

## 📈 Output Example

```
📊 Performance Metrics
╒════════════════════╤════════════╕
│ Metric             │ Value      │
╞════════════════════╪════════════╡
│ Final PnL (₹)      │ 12,500.00  │
│ Max Drawdown (₹)   │ 3,200.00   │
│ Sharpe Ratio       │ 1.42       │
│ Sortino Ratio      │ 1.85       │
│ Win Rate (%)       │ 61.23      │
╘════════════════════╧════════════╛
```

---

## 🤖 Machine Learning Integration

```python
from utils.ml_evaluator import evaluate_with_ml
from tabulate import tabulate

ml_results = evaluate_with_ml(results)
print(tabulate(ml_results, headers="keys", tablefmt="fancy_grid"))
```

Predicts the **next-day direction of profit/loss** using models like:
- Random Forest
- SVM
- Logistic Regression
- Neural Network (MLP)
- KNN
- Decision Tree
- Naive Bayes
- Gradient Boosting

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/options-backtesting-framework.git
cd options-backtesting-framework
pip install -r requirements.txt
```

---

## 🧠 Dependencies

```
pandas
numpy
matplotlib
scikit-learn
tabulate
nsepython
bse
```

---

## 📜 License

MIT License © [Rohit Baskaran](https://github.com/rohit0369)

---

## 💡 Author

👤 **Rohit Baskaran**  
💼 CSE Student | Interested in Trading, Stock Market & AI  
📧 rohitbaskaran369@gmail.com  
🌙 GitHub: [rohit0369](https://github.com/rohit0369)

---

## 🏁 Next Steps (Optional Add-Ons)

- [ ] Add visualization dashboards using Plotly  
- [ ] Integrate live NSE option chain API  
- [ ] Add Paper-Trading mode  
- [ ] Connect with Zerodha Kite or Angel One SmartAPI  
- [ ] Deploy via Streamlit or Flask for web interface  
