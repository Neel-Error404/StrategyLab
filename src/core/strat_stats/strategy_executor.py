import os
import logging
from typing import Dict, Any
import pandas as pd

from config import BACKTESTER_CONFIG  # Updated import
from src.core.etl.loader import load_base_data
from src.core.strat_stats.statistics import (
    save_base_file,
    save_trade_file,
    calculate_metrics,
    save_summary_file
)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategies.strategy_mse import MSEStrategy  # or dynamic import

def execute_strategy(ticker: str, pull_date: str, strategy_name: str) -> Dict[str, Any]:
    """
    1) Load base data
    2) Execute the strategy
    3) Extract trades
    4) Save _Base_ and _Trades_ files
    5) Append to _Summary_ if needed
    """
    logging.info(f"Executing strategy '{strategy_name}' for {ticker} on {pull_date}")

    # 1) Load data
    base_df = load_base_data(date_range, ticker)
    if base_df is None or base_df.empty:
        logging.warning(f"No data loaded for {ticker} on {pull_date}. Skipping.")
        return {}

    # 2) Run the strategy (returns a DataFrame with signals)
    df_with_flags = prepare_strategy_data(base_df.copy(), pull_date, ticker)
    if df_with_flags is None or df_with_flags.empty:
        logging.warning(f"Strategy returned no data for {ticker} on {pull_date}.")
        return {}

    # 3) Extract trades from signals
    trades = extract_trades(df_with_flags)

    # 4) Save base & trades
    strategy_output_folder = os.path.join(
        BACKTESTER_CONFIG['OUTPUT_FOLDER'], 
        strategy_name, 
        date_range
    )
    os.makedirs(strategy_output_folder, exist_ok=True)

    base_file = os.path.join(strategy_output_folder, f"{ticker}_Base_{pull_date}.csv")
    print(f"{ticker}_Base_{pull_date}.csv",", Saved")
    save_base_file(df_with_flags, base_file)

    trade_file = os.path.join(strategy_output_folder, f"{ticker}_Trades_{pull_date}.csv")
    print(f"{ticker}_Trades_{pull_date}.csv",", Saved")
    save_trade_file(trades, trade_file)

    # 5) Compute metrics & append to summary
    metrics = calculate_metrics(trades)
    metrics["Ticker"] = ticker
    metrics["Strategy"] = strategy_name
    metrics["Pull Date"] = pull_date

    summary_file = os.path.join(strategy_output_folder, f"{pull_date}_Summary.csv")
    
    # or if you prefer a single summary per strategy, do:
    # summary_file = os.path.join(BACKTESTER_CONFIG['OUTPUT_FOLDER'], strategy_name, "All_Summary.csv")
    save_summary_file(metrics, summary_file)

    return metrics

def extract_trades(df: pd.DataFrame) -> list:
    """
    Iterates through the strategy DataFrame row‐by‐row to detect trade entry and exit signals.
    This updated version also records:
      - "High Time": the timestamp when the highest price (for buy trades) was reached.
      - "Low Time": the timestamp when the lowest price (for buy trades) was reached.
      - "Trade Duration (min)": the time in minutes from entry to exit.
    For sell trades, the logic is similar but the extreme is reversed.
    """
    trades = []
    in_buy_trade = False
    in_sell_trade = False
    trade = {}

    for idx, row in df.iterrows():
        # Check for BUY entry if no trade is active.
        if not in_buy_trade and not in_sell_trade and row.get("entry_signal_buy"):
            in_buy_trade = True
            trade = {
                "Trade Type": "Buy",
                "Entry Time": row["timestamp"],
                "Entry Price": row["close"],
                "High During Trade": row["close"],
                "Low During Trade": row["close"],
                "High Time": row["timestamp"],
                "Low Time": row["timestamp"],
            }
        # Check for SELL entry if no trade is active.
        elif not in_buy_trade and not in_sell_trade and row.get("entry_signal_sell"):
            in_sell_trade = True
            trade = {
                "Trade Type": "Sell",
                "Entry Time": row["timestamp"],
                "Entry Price": row["close"],
                "High During Trade": row["close"],
                "Low During Trade": row["close"],
                "High Time": row["timestamp"],
                "Low Time": row["timestamp"],
            }
        
        # If in a BUY trade, update high/low values and their timestamps.
        if in_buy_trade:
            if row["close"] > trade["High During Trade"]:
                trade["High During Trade"] = row["close"]
                trade["High Time"] = row["timestamp"]
            if row["close"] < trade["Low During Trade"]:
                trade["Low During Trade"] = row["close"]
                trade["Low Time"] = row["timestamp"]
            if row.get("exit_signal_buy"):
                trade["Exit Time"] = row["timestamp"]
                trade["Exit Price"] = row["close"]
                trade["Profit (Currency)"] = trade["Exit Price"] - trade["Entry Price"]
                trade["Profit (%)"] = (100 * trade["Profit (Currency)"] / trade["Entry Price"]
                                       if trade["Entry Price"] != 0 else 0)
                # Calculate trade duration in minutes.
                entry_time = pd.to_datetime(trade["Entry Time"])
                exit_time = pd.to_datetime(trade["Exit Time"])
                trade["Trade Duration (min)"] = round((exit_time - entry_time).total_seconds() / 60, 2)
                trades.append(trade)
                in_buy_trade = False
                trade = {}
        
        # If in a SELL trade, update high/low values and their timestamps.
        if in_sell_trade:
            if row["close"] < trade["Low During Trade"]:
                trade["Low During Trade"] = row["close"]
                trade["Low Time"] = row["timestamp"]
            if row["close"] > trade["High During Trade"]:
                trade["High During Trade"] = row["close"]
                trade["High Time"] = row["timestamp"]
            if row.get("exit_signal_sell"):
                trade["Exit Time"] = row["timestamp"]
                trade["Exit Price"] = row["close"]
                trade["Profit (Currency)"] = trade["Entry Price"] - trade["Exit Price"]
                trade["Profit (%)"] = (100 * trade["Profit (Currency)"] / trade["Entry Price"]
                                       if trade["Entry Price"] != 0 else 0)
                entry_time = pd.to_datetime(trade["Entry Time"])
                exit_time = pd.to_datetime(trade["Exit Time"])
                trade["Trade Duration (min)"] = round((exit_time - entry_time).total_seconds() / 60, 2)
                trades.append(trade)
                in_sell_trade = False
                trade = {}
                
    return trades
