from nsepython import index_history
from bse import BSE
import pandas as pd
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse

# Define bse_index_history using BSE class
bse_index_history = lambda index_name, start_date, end_date: BSE(download_folder=os.getcwd()).fetchHistoricalIndexData(index_name, start_date, end_date)

# ------------------------------
# FUNCTION: DOWNLOAD INDEX DATA
# ------------------------------
def get_index_data(symbol, start, end, interval="1d"):
    print(f"\n📥 Fetching data for {symbol} ({interval} interval) from {start} to {end}...")
    # Convert dates to DD-MMM-YYYY format
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    # Cap end date at today's date to avoid future date errors
    end_dt = min(end_dt, datetime.now())
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
                print(f"❌ Error fetching BSE data: {e}")
                data = pd.DataFrame()
        else:
            try:
                data = index_history(index_name, start_str, end_str)
            except Exception as e:
                print(f"❌ Error fetching NSE data: {e}")
                data = pd.DataFrame()
        if not data.empty:
            data = data.rename(columns={'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low', 'CLOSE': 'Close', 'HistoricalDate': 'Date'})
            data['Date'] = pd.to_datetime(data['Date'], format='%d %b %Y')
            data.set_index('Date', inplace=True)
            data = data.sort_index()
            data[['Open', 'High', 'Low', 'Close']] = data[['Open', 'High', 'Low', 'Close']].astype(float)
    else:
        print("❌ Error: Only index symbols are supported. Please enter a valid index symbol.")
        exit()
    if data.empty:
        print("❌ Error: No data fetched. Check symbol/date range.")
        exit()
    if 'Close' not in data.columns:
        print("❌ Error: 'Close' column not found in data.")
        print("Available columns:", list(data.columns))
        exit()
    data.dropna(inplace=True)
    if data.empty:
        print("❌ Error: No data fetched after cleaning. Check symbol/date range.")
        exit()
    return data

# ------------------------------
# FUNCTION: OPTION BACKTEST WITH STOP LOSS (NO THRESHOLD)
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
        # Intraday: entry every day
        data['Return'] = data['Close'].pct_change()
        if option_type == "call":
            data['Signal'] = 1
        else:
            data['Signal'] = -1
        # Initialize
        data['Option_PnL'] = 0.0
        cumulative_pnl = []
        in_trade = False
        entry_price = 0.0
        total_pnl = 0.0
        for i in range(1, len(data)):
            if not in_trade:
                # Enter trade daily
                in_trade = True
                entry_price = data['Close'].iloc[i]
                entry_signal = data['Signal'].iloc[i]
            elif in_trade:
                # Calculate change since entry
                price_change = (data['Close'].iloc[i] - entry_price) / entry_price
                if option_type == "put":
                    price_change = -price_change
                pnl = price_change * premium * lot_size
                total_pnl += pnl
                # Target Profit
                if price_change >= target_profit_pct:
                    in_trade = False  # exit trade
                # Stop Loss
                elif price_change <= -stop_loss_pct:
                    total_pnl -= stop_loss_pct * premium * lot_size
                    in_trade = False  # exit trade
            cumulative_pnl.append(total_pnl)
        # Fill first row
        cumulative_pnl = [0] + cumulative_pnl
        data['Cumulative_PnL'] = cumulative_pnl
    elif trading_type == 'swing':
        # Swing: enter every 5 days, hold until stop loss or 5 days
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
                # Enter trade every 5 days or after exit
                in_trade = True
                entry_price = data['Close'].iloc[i]
                entry_signal = data['Signal'].iloc[i]
                trade_start = i
                total_pnl = 0.0
            else:
                # Calculate change since entry
                price_change = (data['Close'].iloc[i] - entry_price) / entry_price
                if option_type == "put":
                    price_change = -price_change
                pnl = price_change * premium * lot_size
                total_pnl += pnl
                # Target Profit
                if price_change >= target_profit_pct:
                    in_trade = False  # exit trade
                # Stop Loss
                elif price_change <= -stop_loss_pct:
                    total_pnl -= stop_loss_pct * premium * lot_size
                    in_trade = False  # exit trade
            cumulative_pnl.append(total_pnl)
        # Fill first row
        cumulative_pnl = [0] + cumulative_pnl
        data['Cumulative_PnL'] = cumulative_pnl

    # Performance Metrics
    final_pnl = data['Cumulative_PnL'].iloc[-1]
    max_drawdown = (data['Cumulative_PnL'].cummax() - data['Cumulative_PnL']).max()
    daily_ret = pd.Series(data['Cumulative_PnL']).diff().dropna()
    sharpe_ratio = np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252) if np.std(daily_ret) != 0 else 0

    print("\n========== 📊 PERFORMANCE REPORT ==========")
    print(f"📅 Period: {data.index[0].date()} → {data.index[-1].date()}")
    print(f"💰 Final Cumulative PnL: ₹{final_pnl:.2f}")
    print(f"🔻 Max Drawdown: ₹{max_drawdown:.2f}")
    print(f"⚙️ Sharpe Ratio: {sharpe_ratio:.2f}")
    print("===========================================")

    # Monthly PnL
    monthly_pnl = data['Cumulative_PnL'].groupby(pd.Grouper(freq='M')).agg(lambda x: x.iloc[-1] - x.iloc[0])
    print("\n========== 📅 MONTHLY PnL REPORT ==========")
    for date, pnl in monthly_pnl.items():
        print(f"{date.strftime('%B %Y')}: ₹{pnl:.2f}")
    print("===========================================")

    downside_returns = daily_ret[daily_ret < 0]
    sortino_ratio = np.mean(daily_ret) / np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 and np.std(downside_returns) != 0 else 0
    win_rate = (daily_ret > 0).sum() / len(daily_ret) * 100 if len(daily_ret) > 0 else 0

    print(f"⚡ Sortino Ratio: {sortino_ratio:.2f}")
    print(f"🏆 Win Rate: {win_rate:.2f}%")

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
# MAIN PROGRAM
# ------------------------------
def main():
    print("=========== 🇮🇳 INDIAN OPTIONS BACKTESTING SYSTEM (STOP LOSS, TARGET PROFIT, AUTO LOT) ===========")

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
            print("❌ Error: Invalid index symbol. Please enter one of the supported indices.")
            continue
        start = input("Enter Start Date (YYYY-MM-DD): ")
        end = input("Enter End Date (YYYY-MM-DD): ")
        interval = "1d"
        option_type = input("Enter Option Type (call / put): ").lower()

        # Auto lot size
        if symbol == "^NSEI":
            default_lot = 75  # NIFTY 50
        elif symbol == "^NSEBANK":
            default_lot = 30  # BANKNIFTY
        elif symbol == "^FINNIFTY":
            default_lot = 65  # FINNIFTY
        elif symbol == "^NIFTYNEXT50":
            default_lot = 25  # NIFTY NEXT 50
        elif symbol == "^MIDCAPSELECT":
            default_lot = 120  # NIFTY MIDCAP SELECT
        elif symbol == "^SENSEX":
            default_lot = 20  # SENSEX
        elif symbol == "^BANKEX":
            default_lot = 30  # BANKEX
        print(f"Default Lot Size for {symbol}: {default_lot}")

        lot_size_input = input(f"Enter Lot Size or press Enter to use default ({default_lot}): ")
        lot_size = int(lot_size_input) if lot_size_input.strip() != "" else default_lot

        premium = float(input("Enter Option Premium Price (₹): "))
        stop_loss_pct = float(input("Enter Stop Loss Percentage (e.g., 0.2 for 20%): "))
        trading_type = input("Enter Trading Type (intraday / positional / swing): ").lower()

        # Run backtest
        target_profit_pct = float(input("Enter Target Profit Percentage (e.g., 0.2 for 20%): "))
        data = get_index_data(symbol, start, end, interval)
        results, monthly_pnl, final_pnl, max_drawdown, sharpe_ratio, sortino_ratio, win_rate = option_backtest(data, option_type, lot_size, premium, stop_loss_pct, target_profit_pct, trading_type)

        # Specific Date PnL
        date_input = input("\nEnter a specific date to view PnL (YYYY-MM-DD) or press Enter to skip: ").strip()
        if date_input:
            try:
                specific_date = pd.to_datetime(date_input)
                is_market, reason = is_market_day(specific_date, results.index)
                if is_market:
                    pnl_value = results.loc[specific_date, 'Cumulative_PnL']
                    print(f"\n📅 PnL as of {specific_date.date()} (Market Day): ₹{pnl_value:.2f}")
                else:
                    print(f"\n❌ {specific_date.date()} is a {reason}. No trading data available.")
            except ValueError:
                print("\n❌ Invalid date format. Please use YYYY-MM-DD.")

        # Save results
        save = input("\nDo you want to save results to CSV? (yes/no): ").lower()
        if save == "yes":
            filename = f"{symbol}_{option_type}_{trading_type}_SL_TP_backtest.csv"
            results.to_csv(filename)
            print(f"✅ Backtest results saved as '{filename}'")

        # Plot PnL
        plot = input("\nDo you want to view performance chart? (yes/no): ").lower()
        if plot == "yes":
            plt.figure(figsize=(10, 5))
            plt.plot(results['Cumulative_PnL'], label='Cumulative PnL (₹)', linewidth=2)
            plt.title(f"{symbol} {option_type.upper()} Option Strategy ({trading_type.upper()}) (with Stop Loss and Target Profit)")
            plt.xlabel("Date")
            plt.ylabel("Cumulative PnL (₹)")
            plt.legend()
            plt.grid(True)
            plt.show()

        # Continue or exit
        continue_bt = input("\nDo you want to continue backtesting? (yes/no): ").lower()
        if continue_bt != "yes":
            break

# ------------------------------
# RUN
# ------------------------------
if __name__ == "__main__":
        main()
