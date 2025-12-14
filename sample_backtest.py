from nsepython import index_history
from bse import BSE
import pandas as pd
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import datetime
import abc
from tabulate import tabulate
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

def format_currency(value):
    """
    Format currency value to indicate lakhs (L), crores (Cr), or rupees (₹).
    """
    if abs(value) >= 10000000:  # 1 crore
        return f"₹{value / 10000000:.2f} Cr"
    elif abs(value) >= 100000:  # 1 lakh
        return f"₹{value / 100000:.2f} L"
    else:
        return f"₹{value:.2f}"

bse_index_history = lambda index_name, start_date, end_date: BSE(download_folder=os.getcwd()).fetchHistoricalIndexData(index_name, start_date, end_date)

class Strategy(abc.ABC):
    """
    Abstract base class for trading strategies.
    """
    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def get_signals(self, data):
        """
        Generate trading signals based on data.
        Returns a pandas Series with signals: 1 (buy/long), -1 (sell/short), 0 (hold/no action).
        """
        pass

    @abc.abstractmethod
    def calculate_pnl(self, data, signals, lot_size, premium, stop_loss_pct, target_profit_pct, trading_type):
        """
        Calculate PnL for the strategy.
        Premium can be a dict for multi-leg strategies.
        Returns updated data with 'Cumulative_PnL' column, and other metrics.
        """
        pass


class LongStraddle(Strategy):
    """
    Long Straddle strategy: Buy call and put at the same strike.
    """
    def __init__(self, call_premium, put_premium, strike=None, expiry_days=30):
        super().__init__("Long Straddle")
        self.call_premium = call_premium
        self.put_premium = put_premium
        self.net_premium = call_premium + put_premium  # Cost to enter
        self.strike = strike  # If None, use ATM (close price)
        self.expiry_days = expiry_days  # For time decay considerations

    def get_signals(self, data):
        # Always long straddle: signal 1
        return pd.Series(1, index=data.index)

    def calculate_pnl(self, data, signals, lot_size, premium, stop_loss_pct, target_profit_pct, trading_type):
        print(f"\n⚙️ Running {self.name} Strategy ({trading_type.upper()}) with Stop Loss ({stop_loss_pct*100:.1f}%) and Target Profit ({target_profit_pct*100:.1f}%)...")

        strike = self.strike if self.strike is not None else data['Close'].iloc[0]
        data['Intrinsic_Value'] = data['Close'].apply(lambda x: max(x - strike, 0) + max(strike - x, 0))
        data['PnL'] = data['Intrinsic_Value'] - self.net_premium
        data['Cumulative_PnL'] = data['PnL'].cumsum()

        return data


class BearPutSpread(Strategy):
    """
    Bear Put Spread: Buy higher strike put, sell lower strike put.
    """
    def __init__(self, buy_put_premium, sell_put_premium):
        super().__init__("Bear Put Spread")
        self.buy_put_premium = buy_put_premium
        self.sell_put_premium = sell_put_premium
        self.net_premium = buy_put_premium - sell_put_premium  # Debit spread

    def get_signals(self, data):
        # Short spread: signal -1
        return pd.Series(-1, index=data.index)

    def calculate_pnl(self, data, signals, lot_size, premium, stop_loss_pct, target_profit_pct, trading_type):
        print(f"\n⚙️ Running {self.name} Strategy ({trading_type.upper()}) with Stop Loss ({stop_loss_pct*100:.1f}%) and Target Profit ({target_profit_pct*100:.1f}%)...")

        # Assume higher_strike = strike + spread_width, lower_strike = strike - spread_width
        # But for simplicity, assume strikes based on initial close.
        # This is rough.

        # PnL = min(max(higher - index, 0), spread_width) - net_prem
        # But to simplify, use similar to bull call spread but for puts.

        strike = data['Close'].iloc[0]
        spread_width = 100  # assume
        higher_strike = strike + spread_width
        lower_strike = strike - spread_width

        # Vectorized calculation for bear put spread intrinsic value
        data['Intrinsic_Value'] = np.minimum(np.maximum(higher_strike - data['Close'], 0), spread_width)
        data['PnL'] = data['Intrinsic_Value'] - self.net_premium
        data['Cumulative_PnL'] = data['PnL'].cumsum()

        return data


class LongStrangle(Strategy):
    """
    Long Strangle: Buy OTM call and OTM put at different strikes.
    """
    def __init__(self, call_premium, put_premium):
        super().__init__("Long Strangle")
        self.call_premium = call_premium
        self.put_premium = put_premium
        self.net_premium = call_premium + put_premium  # Cost to enter

    def get_signals(self, data):
        # Always long strangle: signal 1
        return pd.Series(1, index=data.index)

    def calculate_pnl(self, data, signals, lot_size, premium, stop_loss_pct, target_profit_pct, trading_type):
        print(f"\n⚙️ Running {self.name} Strategy ({trading_type.upper()}) with Stop Loss ({stop_loss_pct*100:.1f}%) and Target Profit ({target_profit_pct*100:.1f}%)...")

        strike = data['Close'].iloc[0]
        spread = 100
        call_strike = strike + spread
        put_strike = strike - spread

        data['Intrinsic_Value'] = data['Close'].apply(lambda x: max(x - call_strike, 0) + max(put_strike - x, 0))
        data['PnL'] = data['Intrinsic_Value'] - self.net_premium
        data['Cumulative_PnL'] = data['PnL'].cumsum()

        return data


class BullCallSpread(Strategy):
    """
    Bull Call Spread: Buy lower strike call, sell higher strike call.
    """
    def __init__(self, buy_call_premium, sell_call_premium):
        super().__init__("Bull Call Spread")
        self.buy_call_premium = buy_call_premium
        self.sell_call_premium = sell_call_premium
        self.net_premium = buy_call_premium - sell_call_premium  # Debit spread

    def get_signals(self, data):
        # Long spread: signal 1
        return pd.Series(1, index=data.index)

    def calculate_pnl(self, data, signals, lot_size, premium, stop_loss_pct, target_profit_pct, trading_type):
        print(f"\n⚙️ Running {self.name} Strategy ({trading_type.upper()}) with Stop Loss ({stop_loss_pct*100:.1f}%) and Target Profit ({target_profit_pct*100:.1f}%)...")

        # Assume lower_strike = strike - spread_width, higher_strike = strike + spread_width
        # But for simplicity, assume strikes based on initial close.
        # This is rough.

        # PnL = min(max(index - lower, 0), spread_width) - net_prem
        # But to simplify, use similar to straddle.

        strike = data['Close'].iloc[0]
        spread_width = 100  # assume
        lower_strike = strike - spread_width
        higher_strike = strike + spread_width

        data['Intrinsic_Value'] = data['Close'].apply(lambda x: min(max(x - lower_strike, 0), spread_width))
        data['PnL'] = data['Intrinsic_Value'] - self.net_premium
        data['Cumulative_PnL'] = data['PnL'].cumsum()

        return data

class CustomStrategy(Strategy):
    """
    Custom strategy: User-defined simple signal.
    """
    def __init__(self, name, signal_type):
        super().__init__(name)
        self.signal_type = signal_type

    def get_signals(self, data):
        if self.signal_type == "call_buy":
            return pd.Series(1, index=data.index)
        elif self.signal_type == "call_sell":
            return pd.Series(-1, index=data.index)
        elif self.signal_type == "put_buy":
            return pd.Series(-1, index=data.index)
        elif self.signal_type == "put_sell":
            return pd.Series(1, index=data.index)
        else:
            return pd.Series(0, index=data.index)

    def calculate_pnl(self, data, signals, lot_size, premium, stop_loss_pct, target_profit_pct, trading_type):
        print(f"\n⚙️ Running {self.name} Strategy ({trading_type.upper()}) with Stop Loss ({stop_loss_pct*100:.1f}%) and Target Profit ({target_profit_pct*100:.1f}%)...")

        if 'call' in self.signal_type:
            option_type = 'call'
        else:
            option_type = 'put'

        if trading_type == 'positional':
            in_trade = True
            entry_price = data['Close'].iloc[1]
            total_pnl = 0.0
            exit_index = len(data) - 1
            for i in range(2, len(data)):
                price_change = (data['Close'].iloc[i] - entry_price) / entry_price
                if option_type == 'put':
                    price_change = -price_change
                pnl = price_change * premium * lot_size
                total_pnl = pnl
                if price_change >= target_profit_pct:
                    in_trade = False
                    exit_index = i
                    break
                elif price_change <= -stop_loss_pct:
                    total_pnl -= stop_loss_pct * premium * lot_size
                    in_trade = False
                    exit_index = i
                    break
            cumulative_pnl = [0.0] * len(data)
            cumulative_pnl[exit_index:] = [total_pnl] * len(cumulative_pnl[exit_index:])
            data['Cumulative_PnL'] = cumulative_pnl
        elif trading_type == 'intraday':
            data['Return'] = data['Close'].pct_change()
            if option_type == "call":
                data['Signal'] = 1
            else:
                data['Signal'] = -1
            data['Option_PnL'] = 0.0
            cumulative_pnl = []
            in_trade = False
            entry_price = 0.0
            total_pnl = 0.0
            for i in range(1, len(data)):
                if not in_trade:
                    in_trade = True
                    entry_price = data['Close'].iloc[i]
                elif in_trade:
                    price_change = (data['Close'].iloc[i] - entry_price) / entry_price
                    if option_type == "put":
                        price_change = -price_change
                    pnl = price_change * premium * lot_size
                    total_pnl += pnl
                    if price_change >= target_profit_pct:
                        in_trade = False
                    elif price_change <= -stop_loss_pct:
                        total_pnl -= stop_loss_pct * premium * lot_size
                        in_trade = False
                cumulative_pnl.append(total_pnl)
            cumulative_pnl = [0] + cumulative_pnl
            data['Cumulative_PnL'] = cumulative_pnl
        elif trading_type == 'swing':
            data['Return'] = data['Close'].pct_change()
            if option_type == "call":
                data['Signal'] = 1
            else:
                data['Signal'] = -1
            data['Option_PnL'] = 0.0
            cumulative_pnl = []
            in_trade = False
            entry_price = 0.0
            total_pnl = 0.0
            hold_days = 5
            trade_start = 0
            for i in range(1, len(data)):
                if not in_trade or (i - trade_start) >= hold_days:
                    in_trade = True
                    entry_price = data['Close'].iloc[i]
                    trade_start = i
                    total_pnl = 0.0
                else:
                    price_change = (data['Close'].iloc[i] - entry_price) / entry_price
                    if option_type == "put":
                        price_change = -price_change
                    pnl = price_change * premium * lot_size
                    total_pnl += pnl
                    if price_change >= target_profit_pct:
                        in_trade = False
                    elif price_change <= -stop_loss_pct:
                        total_pnl -= stop_loss_pct * premium * lot_size
                        in_trade = False
                cumulative_pnl.append(total_pnl)
            cumulative_pnl = [0] + cumulative_pnl
            data['Cumulative_PnL'] = cumulative_pnl
        else:
            # default to simple
            data['Signal'] = signals
            data['Return'] = data['Close'].pct_change()
            data['PnL'] = data['Signal'].shift(1) * data['Return'] * premium * lot_size
            data['Cumulative_PnL'] = data['PnL'].cumsum()

        return data





def compute_metrics(data):
    final_pnl = data['Cumulative_PnL'].iloc[-1]
    max_drawdown = (data['Cumulative_PnL'].cummax() - data['Cumulative_PnL']).max()
    daily_ret = pd.Series(data['Cumulative_PnL']).diff().dropna()
    sharpe_ratio = np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252) if np.std(daily_ret) != 0 else 0

    monthly_pnl = data['Cumulative_PnL'].groupby(pd.Grouper(freq='M')).agg(lambda x: x.iloc[-1] - x.iloc[0])
    weekly_pnl = data['Cumulative_PnL'].groupby(pd.Grouper(freq='W')).agg(lambda x: x.iloc[-1] - x.iloc[0])

    downside_returns = daily_ret[daily_ret < 0]
    sortino_ratio = np.mean(daily_ret) / np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 and np.std(downside_returns) != 0 else 0
    win_rate = (daily_ret > 0).sum() / len(daily_ret) * 100 if len(daily_ret) > 0 else 0

    # Create Performance Metrics DataFrame
    metrics_data = {
        'Metric': ['Period', 'Final Cumulative PnL', 'Max Drawdown', 'Sharpe Ratio', 'Sortino Ratio', 'Win Rate'],
        'Value': [f"{data.index[0].date()} → {data.index[-1].date()}", f"₹{final_pnl:.2f}", f"₹{max_drawdown:.2f}", f"{sharpe_ratio:.2f}", f"{sortino_ratio:.2f}", f"{win_rate:.2f}%"]
    }
    performance_df = pd.DataFrame(metrics_data)

    # Create Monthly PnL DataFrame
    monthly_pnl_df = pd.DataFrame({
        'Month': [date.strftime('%B %Y') for date in monthly_pnl.index],
        'PnL': [f"₹{pnl:.2f}" for pnl in monthly_pnl.values]
    })

    return data, monthly_pnl, weekly_pnl, final_pnl, max_drawdown, sharpe_ratio, sortino_ratio, win_rate, performance_df, monthly_pnl_df


# ------------------------------
# FUNCTION: EVALUATE WITH ML
# ------------------------------
def evaluate_with_ml(results):
    """
    Evaluate ML models to predict the direction of daily PnL.
    Returns a DataFrame with metrics: Model, Accuracy, Precision, Recall.
    """
    # Create daily PnL
    results['Daily_PnL'] = results['Cumulative_PnL'].diff().fillna(0)

    # Target: 1 if Daily_PnL > 0, 0 otherwise
    results['Target'] = (results['Daily_PnL'] > 0).astype(int)

    # Features: lagged close prices (1-5 days), volatility (rolling std of returns)
    for lag in range(1, 6):
        results[f'Close_lag_{lag}'] = results['Close'].shift(lag)
    results['Volatility'] = results['Close'].pct_change().rolling(5).std()

    # Drop NaN rows
    features = [f'Close_lag_{i}' for i in range(1, 6)] + ['Volatility']
    data = results[features + ['Target']].dropna()

    if data.empty:
        print("Not enough data for ML evaluation.")
        return pd.DataFrame()

    X = data[features]
    y = data['Target']

    # Check if there are at least two classes
    if y.nunique() <= 1:
        print("Only one class in target, skipping ML evaluation.")
        return pd.DataFrame()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42),
        'MLP': MLPClassifier(random_state=42, max_iter=500),
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB()
    }

    metrics_list = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        metrics_list.append({'Model': name, 'Accuracy': f"{acc:.2f}", 'Precision': f"{prec:.2f}", 'Recall': f"{rec:.2f}"})

    return pd.DataFrame(metrics_list)


def backtest_strategy(strategy, data, lot_size, premium, stop_loss_pct, target_profit_pct, trading_type):
    signals = strategy.get_signals(data)
    data = strategy.calculate_pnl(data, signals, lot_size, premium, stop_loss_pct, target_profit_pct, trading_type)
    return compute_metrics(data)

# ------------------------------
# FUNCTION: DOWNLOAD INDEX DATA
# ------------------------------
def get_index_data(symbol, start, end, interval="1d"):
    print(f"\n📥 Fetching data for {symbol} ({interval} interval) from {start} to {end}...")
    # Convert dates to DD-MMM-YYYY format
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    # Cap end date at today's date to avoid future date errors
    end_dt = min(end_dt, datetime.datetime.now())
    start_str = start_dt.strftime('%d %b %Y')
    end_str = end_dt.strftime('%d %b %Y')
    
    if symbol.startswith("^"):
        if symbol == "^NSEI":
            index_name = "NIFTY 50"
        elif symbol == "^NSEBANK":
            index_name = "NIFTY BANK"
        elif symbol == "^FINNIFTY":
            index_name = "NIFTY FINANCIAL SERVICES"
        elif symbol == "^NIFTYNEXT50":
            index_name = "NIFTY NEXT 50"
        elif symbol == "^MIDCAPSELECT":
            index_name = "NIFTY MIDCAP SELECT"
        elif symbol == "^SENSEX":
            index_name = "SENSEX"
        elif symbol == "^BANKEX":
            index_name = "BANKEX"
        else:
            index_name = "NIFTY 50"  # default
        if symbol in ["^SENSEX", "^BANKEX"]:
            try:
                csv_path = bse_index_history(index_name, start_dt, end_dt)
                data = pd.read_csv(csv_path)
            except Exception as e:
                print(f"Error fetching BSE data: {e}")
                raise ValueError(f"Failed to fetch BSE data for {symbol}: {e}")
        else:
            try:
                data = index_history(index_name, start_str, end_str)
            except Exception as e:
                print(f"Error fetching NSE data: {e}")
                raise ValueError(f"Failed to fetch NSE data for {symbol}: {e}")
        if not data.empty:
            data = data.rename(columns={'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low', 'CLOSE': 'Close', 'HistoricalDate': 'Date'})
            data['Date'] = pd.to_datetime(data['Date'], format='%d %b %Y')
            data.set_index('Date', inplace=True)
            data = data.sort_index()
            data[['Open', 'High', 'Low', 'Close']] = data[['Open', 'High', 'Low', 'Close']].astype(float)
    else:
        print("Error: Only index symbols are supported. Please enter a valid index symbol.")
        exit()
    if data.empty:
        print("Error: No data fetched. Check symbol/date range.")
        exit()
    if 'Close' not in data.columns:
        print("Error: 'Close' column not found in data.")
        print("Available columns:", list(data.columns))
        exit()
    data.dropna(inplace=True)
    if data.empty:
        print("Error: No data fetched after cleaning. Check symbol/date range.")
        exit()
    return data

# ------------------------------
# FUNCTION: OPTION BACKTEST WITH STOP LOSS
# ------------------------------
def option_backtest(data, option_type, lot_size, premium, stop_loss_pct, target_profit_pct, trading_type):
    print(f"\n⚙️ Running {option_type.upper()} Option Strategy ({trading_type.upper()}) with Stop Loss ({stop_loss_pct*100:.1f}%) and Target Profit ({target_profit_pct*100:.1f}%)...")

    if trading_type == 'positional':
        # Positional: enter once at start, hold until stop loss
        in_trade = True
        entry_price = data['Close'].iloc[1]
        total_pnl = 0.0
        exit_index = len(data) - 1
        for i in range(2, len(data)):
            price_change = (data['Close'].iloc[i] - entry_price) / entry_price
            if option_type == 'put':
                price_change = -price_change
            pnl = price_change * premium * lot_size
            total_pnl = pnl
            if price_change >= target_profit_pct:
                # Target profit hit
                in_trade = False
                exit_index = i
                break
            elif price_change <= -stop_loss_pct:
                total_pnl -= stop_loss_pct * premium * lot_size
                in_trade = False
                exit_index = i
                break
        # Cumulative PnL: 0 until exit, then final pnl
        cumulative_pnl = [0.0] * len(data)
        cumulative_pnl[exit_index:] = [total_pnl] * len(cumulative_pnl[exit_index:])
        data['Cumulative_PnL'] = cumulative_pnl
    elif trading_type == 'intraday':
        data['Return'] = data['Close'].pct_change()
        if option_type == "call":
            data['Signal'] = 1
        else:
            data['Signal'] = -1
        data['Option_PnL'] = 0.0
        cumulative_pnl = []
        in_trade = False
        entry_price = 0.0
        total_pnl = 0.0
        for i in range(1, len(data)):
            if not in_trade:
                in_trade = True
                entry_price = data['Close'].iloc[i]
                entry_signal = data['Signal'].iloc[i]
            elif in_trade:
                price_change = (data['Close'].iloc[i] - entry_price) / entry_price
                if option_type == "put":
                    price_change = -price_change
                pnl = price_change * premium * lot_size
                total_pnl += pnl
                if price_change >= target_profit_pct:
                    in_trade = False
                elif price_change <= -stop_loss_pct:
                    total_pnl -= stop_loss_pct * premium * lot_size
                    in_trade = False
            cumulative_pnl.append(total_pnl)
        cumulative_pnl = [0] + cumulative_pnl
        data['Cumulative_PnL'] = cumulative_pnl
    elif trading_type == 'swing':
        data['Return'] = data['Close'].pct_change()
        if option_type == "call":
            data['Signal'] = 1
        else:
            data['Signal'] = -1
        data['Option_PnL'] = 0.0
        cumulative_pnl = []
        in_trade = False
        entry_price = 0.0
        total_pnl = 0.0
        hold_days = 5
        trade_start = 0
        for i in range(1, len(data)):
            if not in_trade or (i - trade_start) >= hold_days:
                in_trade = True
                entry_price = data['Close'].iloc[i]
                entry_signal = data['Signal'].iloc[i]
                trade_start = i
                total_pnl = 0.0
            else:
                price_change = (data['Close'].iloc[i] - entry_price) / entry_price
                if option_type == "put":
                    price_change = -price_change
                pnl = price_change * premium * lot_size
                total_pnl += pnl
                if price_change >= target_profit_pct:
                    in_trade = False
                elif price_change <= -stop_loss_pct:
                    total_pnl -= stop_loss_pct * premium * lot_size
                    in_trade = False
            cumulative_pnl.append(total_pnl)
        cumulative_pnl = [0] + cumulative_pnl
        data['Cumulative_PnL'] = cumulative_pnl

    # Performance Metrics
    final_pnl = data['Cumulative_PnL'].iloc[-1]
    max_drawdown = (data['Cumulative_PnL'].cummax() - data['Cumulative_PnL']).max()
    daily_ret = pd.Series(data['Cumulative_PnL']).diff().dropna()
    sharpe_ratio = np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252) if np.std(daily_ret) != 0 else 0

    print("\n========== PERFORMANCE REPORT ==========")
    print(f"Period: {data.index[0].date()} → {data.index[-1].date()}")
    print(f"Final Cumulative PnL: ₹{final_pnl:.2f}")
    print(f"Max Drawdown: ₹{max_drawdown:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print("===========================================")

    # Monthly PnL
    monthly_pnl = data['Cumulative_PnL'].groupby(pd.Grouper(freq='M')).agg(lambda x: x.iloc[-1] - x.iloc[0])
    print("\n========== MONTHLY PnL REPORT ==========")
    for date, pnl in monthly_pnl.items():
        print(f"{date.strftime('%B %Y')}: ₹{pnl:.2f}")
    print("===========================================")

    downside_returns = daily_ret[daily_ret < 0]
    sortino_ratio = np.mean(daily_ret) / np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 and np.std(downside_returns) != 0 else 0
    win_rate = (daily_ret > 0).sum() / len(daily_ret) * 100 if len(daily_ret) > 0 else 0

    print(f"Sortino Ratio: {sortino_ratio:.2f}")
    print(f"Win Rate: {win_rate:.2f}%")

    return data, monthly_pnl, final_pnl, max_drawdown, sharpe_ratio, sortino_ratio, win_rate

# ------------------------------
# FUNCTION: CHECK IF DATE IS MARKET DAY
# ------------------------------
def is_market_day(date, data_index):
    """
    Check if the given date is a market day.
    Returns (is_market, reason)
    """
    if date.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return False, "weekend"
    elif date in data_index:
        return True, "market day"
    else:
        return False, "holiday"


# ------------------------------
# FUNCTION: GET MANUAL INPUTS
# ------------------------------
def get_manual_inputs(symbol):
    while True:
        strategy_type = input("Enter Strategy Type (default / custom ): ").lower()
        if strategy_type == "default":
            default_strategy = input("Enter Default Strategy (long_straddle / long_strangle / bull_call_spread / bear_put_spread): ").lower()
            if default_strategy == "long_straddle":
                call_premium = float(input("Enter Call Premium Price (₹): "))
                put_premium = float(input("Enter Put Premium Price (₹): "))
                strategy = LongStraddle(call_premium, put_premium)
                premium_value = call_premium + put_premium
            elif default_strategy == "long_strangle":
                call_premium = float(input("Enter Call Premium Price (₹): "))
                put_premium = float(input("Enter Put Premium Price (₹): "))
                strategy = LongStrangle(call_premium, put_premium)
                premium_value = call_premium + put_premium
            elif default_strategy == "bull_call_spread":
                buy_call_premium = float(input("Enter Buy Call Premium Price (₹): "))
                sell_call_premium = float(input("Enter Sell Call Premium Price (₹): "))
                strategy = BullCallSpread(buy_call_premium, sell_call_premium)
                premium_value = buy_call_premium - sell_call_premium
            elif default_strategy == "bear_put_spread":
                buy_put_premium = float(input("Enter Buy Put Premium Price (₹): "))
                sell_put_premium = float(input("Enter Sell Put Premium Price (₹): "))
                strategy = BearPutSpread(buy_put_premium, sell_put_premium)
                premium_value = buy_put_premium - sell_put_premium
            else:
                print("Error: Invalid default strategy.")
                continue
        elif strategy_type == "custom":
            name = input("Enter Custom Strategy Name: ")
            option_type = input("Enter Option Type (call / put): ").lower()
            action = input("Enter Action (buy / sell): ").lower()
            signal_type = f"{option_type}_{action}"
            premium = float(input("Enter Option Premium Price (₹): "))
            strategy = CustomStrategy(name, signal_type)
            premium_value = premium
        else:
            print("Error: Invalid strategy type.")
            continue

        strategy_name = strategy.name

        # Auto lot size
        if symbol == "^NSEI":
            default_lot = 75
        elif symbol == "^NSEBANK":
            default_lot = 30
        elif symbol == "^FINNIFTY":
            default_lot = 65
        elif symbol == "^NIFTYNEXT50":
            default_lot = 25
        elif symbol == "^MIDCAPSELECT":
            default_lot = 120
        elif symbol == "^SENSEX":
            default_lot = 20
        elif symbol == "^BANKEX":
            default_lot = 30
        print(f"Default Lot Size for {symbol}: {default_lot}")
        lot_size_input = input(f"Enter Lot Size or press Enter to use default ({default_lot}): ")
        lot_size = int(lot_size_input) if lot_size_input.strip() != "" else default_lot

        while True:
            stop_loss_pct_input = input("Enter Stop Loss Percentage (e.g., 0.2 for 20%) or press Enter to skip: ").strip()
            if not stop_loss_pct_input:
                stop_loss_pct = 0.0
                break
            try:
                stop_loss_pct = float(stop_loss_pct_input)
                break
            except ValueError:
                print("Error: Invalid input. Please enter a number or press Enter to skip.")
        trading_type = input("Enter Trading Type (intraday / positional / swing): ").lower()

        while True:
            target_profit_pct_input = input("Enter Target Profit Percentage (e.g., 0.2 for 20%) or press Enter to skip: ").strip()
            if not target_profit_pct_input:
                target_profit_pct = 0.0
                break
            try:
                target_profit_pct = float(target_profit_pct_input)
                break
            except ValueError:
                print("Error: Invalid input. Please enter a number or press Enter to skip.")
        is_selling = strategy_type == "custom" and action == "sell"
        option_type = option_type if strategy_type == "custom" else None
        return strategy, strategy_name, lot_size, stop_loss_pct, trading_type, target_profit_pct, premium_value, is_selling, option_type, strategy_type



# ------------------------------
# MAIN PROGRAM
# ------------------------------
def main():
    print("=========== INDIAN OPTIONS BACKTESTING SYSTEM  ===========")

    while True:
        # User Inputs
        symbol = input("Enter Index Symbol (NIFTY / BANKNIFTY / FINNIFTY / NIFTY NEXT 50 / NIFTY MIDCAP SELECT / SENSEX / BANKEX): ").upper()
        if symbol == "NIFTY":
            symbol = "^NSEI"
        elif symbol == "BANKNIFTY":
            symbol = "^NSEBANK"
        elif symbol == "FINNIFTY":
            symbol = "^FINNIFTY"
        elif symbol == "NIFTY NEXT 50":
            symbol = "^NIFTYNEXT50"
        elif symbol == "NIFTY MIDCAP SELECT":
            symbol = "^MIDCAPSELECT"
        elif symbol == "SENSEX":
            symbol = "^SENSEX"
        elif symbol == "BANKEX":
            symbol = "^BANKEX"
        else:
            print("Error: Invalid index symbol. Please enter one of the supported indices.")
            continue
        start = input("Enter Start Date (YYYY-MM-DD): ")
        end = input("Enter End Date (YYYY-MM-DD): ")

        # Input and validate start and end times
        market_open = datetime.time(9, 15)
        market_close = datetime.time(15, 30)
        while True:
            start_time_input = input("Enter Start Time (HH:MM, e.g., 09:15): ")
            end_time_input = input("Enter End Time (HH:MM, e.g., 15:30): ")
            try:
                start_time = datetime.datetime.strptime(start_time_input, '%H:%M').time()
                end_time = datetime.datetime.strptime(end_time_input, '%H:%M').time()
                if start_time >= market_open and end_time <= market_close and start_time < end_time:
                    break
                else:
                    print("Invalid times. Start time must be >= 09:15, end time <= 15:30, and start time < end time.")
            except ValueError:
                print("Invalid format. Please use HH:MM format.")


        # Manual inputs
        strategy, strategy_name, lot_size, stop_loss_pct, trading_type, target_profit_pct, premium_value, is_selling, option_type, strategy_type = get_manual_inputs(symbol)

        interval = "1d"
        data = get_index_data(symbol, start, end, interval)
        results, monthly_pnl, weekly_pnl, final_pnl, max_drawdown, sharpe_ratio, sortino_ratio, win_rate, performance_df, monthly_pnl_df = backtest_strategy(strategy, data, lot_size, premium_value, stop_loss_pct, target_profit_pct, trading_type)

        # Display Performance Metrics Table
        print("\n========== PERFORMANCE REPORT ==========")
        print(tabulate(performance_df, headers='keys', tablefmt='grid', showindex=False))
        print("===========================================")

        # Display Monthly PnL Table
        print("\n========== MONTHLY PnL REPORT ==========")
        print(tabulate(monthly_pnl_df, headers='keys', tablefmt='grid', showindex=False))
        print("===========================================")

        # ML Evaluation
        ml_df = evaluate_with_ml(results)
        if not ml_df.empty:
            print("\n========== ML EVALUATION REPORT ==========")
            print(tabulate(ml_df, headers='keys', tablefmt='grid', showindex=False))
            print("===========================================")

            # Save ML results
            save_ml = input("\nDo you want to save ML evaluation results to CSV? (yes/no): ").lower()
            if save_ml == "yes":
                ml_filename = f"{symbol}_{strategy_name}_{trading_type}_ml_evaluation.csv"
                ml_df.to_csv(ml_filename, index=False)
                print(f"ML evaluation results saved as '{ml_filename}'")

        # Specific Date PnL
        date_input = input("\nEnter a specific date to view PnL (YYYY-MM-DD) or press Enter to skip: ").strip()
        if date_input:
            try:
                specific_date = pd.to_datetime(date_input)
                is_market, reason = is_market_day(specific_date, results.index)
                if is_market:
                    pnl_value = results.loc[specific_date, 'Cumulative_PnL']
                    print(f"\nPnL as of {specific_date.date()} (Market Day): ₹{pnl_value:.2f}")
                else:
                    print(f"\n{specific_date.date()} is a {reason}. No trading data available.")
            except ValueError:
                print("\nInvalid date format. Please use YYYY-MM-DD.")

        # Weekly PnL for a specific month
        weekly_pnl = None
        month_input = input("\nEnter a specific month to view weekly PnL (YYYY-MM) or press Enter to skip: ").strip()
        if month_input:
            try:
                month_start = pd.to_datetime(month_input + "-01")
                month_end = month_start + pd.offsets.MonthEnd(1)
                monthly_data = results.loc[month_start:month_end]
                if not monthly_data.empty:
                    weekly_groups = monthly_data.groupby(pd.Grouper(freq='W'))
                    daily_data = []
                    for name, group in weekly_groups:
                        if not group.empty:
                            week_start = name - pd.Timedelta(days=6)
                            week_end = name
                            week_label = f"{week_start.strftime('%d %b %Y')} - {week_end.strftime('%d %b %Y')}"
                            daily_pnl = group['Cumulative_PnL'].diff().dropna()
                            for date, pnl in daily_pnl.items():
                                daily_data.append({
                                    'Week': week_label,
                                    'Date': date.strftime('%d %b %Y'),
                                    'Day': date.strftime('%A'),
                                    'PnL': f"₹{pnl:.2f}"
                                })
                    weekly_pnl_df = pd.DataFrame(daily_data)
                    print(f"\n========== WEEKLY PnL REPORT FOR {month_start.strftime('%B %Y')} ==========")
                    print(tabulate(weekly_pnl_df, headers='keys', tablefmt='grid', showindex=False))
                    print("===========================================")
                else:
                    print(f"\nNo data available for {month_start.strftime('%B %Y')}.")
            except ValueError:
                print("\nInvalid month format. Please use YYYY-MM.")

        # Save results
        save = input("\nDo you want to save results to CSV? (yes/no): ").lower()
        if save == "yes":
            filename = f"{symbol}_{strategy_name}_{trading_type}_SL_TP_backtest.csv"
            results.to_csv(filename)
            print(f"Backtest results saved as '{filename}'")

        # Plot PnL
        plot = input("\nDo you want to generate and save performance chart? (yes/no): ").lower()
        if plot == "yes":
            plt.style.use('ggplot')
            plt.figure(figsize=(10, 6))
            plt.plot(results['Cumulative_PnL'], label='Cumulative PnL (₹)', linewidth=2, color='darkblue', marker='o', markersize=3, markerfacecolor='white', markeredgewidth=1)
            plt.fill_between(results.index, results['Cumulative_PnL'], where=(results['Cumulative_PnL'] >= 0), color='green', alpha=0.4, interpolate=True)
            plt.fill_between(results.index, results['Cumulative_PnL'], where=(results['Cumulative_PnL'] < 0), color='red', alpha=0.4, interpolate=True)
            if weekly_pnl is not None:
                plt.bar(weekly_pnl.index, weekly_pnl.values, width=5, alpha=0.7, color='orange', label='Weekly PnL (₹)', edgecolor='black', linewidth=0.5)
            plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
            plt.title(f"{symbol} {strategy_name} Strategy ({trading_type.upper()}) (with Stop Loss and Target Profit)", fontsize=16, fontweight='bold', color='navy')
            plt.xlabel("Date", fontsize=14, fontweight='bold')
            plt.ylabel("Cumulative PnL (₹)", fontsize=14, fontweight='bold')
            plt.legend(fontsize=14, loc='upper left')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            plt.gca().set_facecolor('#f5f5f5')
            # Annotations for key metrics
            final_pnl = results['Cumulative_PnL'].iloc[-1]
            max_dd = (results['Cumulative_PnL'].cummax() - results['Cumulative_PnL']).max()
            plt.text(results.index[int(len(results)*0.75)], results['Cumulative_PnL'].max() * 0.9, f'Final PnL: {format_currency(final_pnl)}\nMax Drawdown: {format_currency(max_dd)}', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8), ha='center')
            plt.tight_layout()
            chart_filename = f"{symbol}_{strategy_name}_{trading_type}_pnl_chart.png"
            plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
            print(f"PnL chart saved as '{chart_filename}'")


        # Continue or exit
        continue_bt = input("\nDo you want to continue backtesting? (yes/no): ").lower()
        if continue_bt != "yes":
            break

# ------------------------------
# RUN
# ------------------------------
if __name__ == "__main__":
        main()
