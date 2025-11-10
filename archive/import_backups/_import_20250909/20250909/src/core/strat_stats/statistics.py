# statistics.py

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any
import os
from datetime import datetime

def save_base_file(df_with_flags: pd.DataFrame, file_name: str):
    try:
        df_with_flags.to_csv(file_name, index=False)
        logging.info(f"Saved base data to {file_name}")
    except Exception as e:
        logging.error(f"Error saving base data to {file_name}: {e}")

def save_trade_file(trades: List[Dict], file_name: str):
    # Define the complete set of expected trade columns
    trade_columns = [
        "Trade Type",
        "Entry Time", 
        "Exit Time",
        "Entry Price",
        "Exit Price", 
        "Profit (Currency)",
        "Profit (%)",
        "High During Trade",
        "Low During Trade", 
        "High Time",
        "Low Time",
        "Trade Duration (min)",
        "Target (%)",
        "Drawdown (%)",
        "RRR",
        "Recovery Time (min)"
    ]
    
    try:
        if not trades:
            # Create empty DataFrame with proper headers
            empty_df = pd.DataFrame(columns=trade_columns)
            empty_df.to_csv(file_name, index=False)
            logging.info(f"Created empty trade file with headers: {file_name}")
        else:
            # Save trades as before
            pd.DataFrame(trades).to_csv(file_name, index=False)
            logging.info(f"Saved {len(trades)} trades to {file_name}")
    except Exception as e:
        logging.error(f"Error saving trades to {file_name}: {e}")

def calculate_metrics(trades: List[Dict], base_data: pd.DataFrame = None) -> Dict[str, Any]:
    if not trades:
        return {
            "Total Trades": 0,
            "Wins": 0,
            "Losses": 0,
            "Win/Loss Ratio": 0,
            "Accuracy (%)": 0,
            "Average Profit (Currency)": 0,
            "Average Profit (%)": 0,
            "Average Target (%)": 0,
            "Max Drawdown (%)": 0,
            "Average Drawdown (%)": 0,
            "Average Trade Duration (min)": 0,
            "Average RRR": 0,
            "Average Recovery Time (min)": 0,
            "Return StdDev (%)": 0,
            "Sharpe Ratio": 0,
            "Buy Trade Metrics": {},
            "Sell Trade Metrics": {}
        }

    rrr_list = []
    duration_list = []
    recovery_list = []
    drawdown_list = []
    
    # First, ensure that for each trade, the "Trade Duration (min)" and "Recovery Time (min)" are computed.
    for trade in trades:
        trade_type = trade.get("Trade Type", "").lower()
        entry = trade.get("Entry Price")
        exit_price = trade.get("Exit Price")
        high = trade.get("High During Trade", entry)
        low = trade.get("Low During Trade", entry)
        # Target and drawdown are computed as before.
        if entry and entry != 0:
            if trade_type == "buy":
                trade["Target (%)"] = round(((high - entry) / entry) * 100, 2)
                trade["Drawdown (%)"] = round(((entry - low) / entry) * 100, 2)
                risk = entry - low
                reward = exit_price - entry if exit_price is not None else 0
                # Recovery: time from the low time to exit time.
                try:
                    low_time = pd.to_datetime(trade.get("Low Time"))
                    exit_time = pd.to_datetime(trade.get("Exit Time"))
                    recovery = (exit_time - low_time).total_seconds() / 60.0
                except Exception:
                    recovery = None
            elif trade_type == "sell":
                trade["Target (%)"] = round(((entry - low) / entry) * 100, 2)
                trade["Drawdown (%)"] = round(((high - entry) / entry) * 100, 2)
                risk = high - entry
                reward = entry - exit_price if exit_price is not None else 0
                # Recovery: time from the high time to exit time.
                try:
                    high_time = pd.to_datetime(trade.get("High Time"))
                    exit_time = pd.to_datetime(trade.get("Exit Time"))
                    recovery = (exit_time - high_time).total_seconds() / 60.0
                except Exception:
                    recovery = None
            else:
                trade["Target (%)"] = 0
                trade["Drawdown (%)"] = 0
                risk = 0
                reward = 0
                recovery = None
        else:
            trade["Target (%)"] = 0
            trade["Drawdown (%)"] = 0
            risk = 0
            reward = 0
            recovery = None

        drawdown_list.append(trade["Drawdown (%)"])
        
        # Calculate RRR
        if risk and risk != 0:
            trade_rrr = reward / risk
            trade["RRR"] = round(trade_rrr, 2)
            rrr_list.append(trade["RRR"])
        else:
            trade["RRR"] = None

        # Trade Duration is expected to be computed in the extraction function.
        try:
            duration = float(trade.get("Trade Duration (min)", 0))
            duration_list.append(duration)
        except Exception:
            pass

        # Recovery Time
        if recovery is not None and recovery < 0:
            recovery = 0
        if recovery is not None:
            trade["Recovery Time (min)"] = round(recovery, 2)
            recovery_list.append(recovery)
        else:
            trade["Recovery Time (min)"] = None

    df = pd.DataFrame(trades)
    total_trades = len(df)
    wins = sum(1 for t in trades if t.get("Profit (Currency)", 0) > 0)
    losses = total_trades - wins

    avg_profit_cur = df["Profit (Currency)"].mean()
    avg_profit_pct = df["Profit (%)"].mean()
    avg_target_pct = df["Target (%)"].mean() if "Target (%)" in df.columns else 0
    max_drawdown_pct = df["Drawdown (%)"].max() if "Drawdown (%)" in df.columns else 0
    avg_drawdown_pct = df["Drawdown (%)"].mean() if "Drawdown (%)" in df.columns else 0

    if losses == 0 and wins > 0:
        win_loss_ratio = float("inf")
    elif losses == 0 and wins == 0:
        win_loss_ratio = 0
    else:
        win_loss_ratio = wins / losses

    accuracy = (wins / total_trades) * 100 if total_trades > 0 else 0
    avg_duration = sum(duration_list) / len(duration_list) if duration_list else 0
    avg_rrr = sum(rrr for rrr in rrr_list if rrr is not None) / len(rrr_list) if rrr_list else 0
    avg_recovery = sum(recovery_list) / len(recovery_list) if recovery_list else 0

    # Return StdDev (%) of Profit (%)
    profit_std = df["Profit (%)"].std() if "Profit (%)" in df.columns else 0

    # Sharpe Ratio (using Profit (%) returns)
    returns_series = df["Profit (%)"] if "Profit (%)" in df.columns else pd.Series([0])
    std_returns = returns_series.std()
    mean_returns = returns_series.mean()
    sharpe_ratio = (mean_returns) / std_returns if std_returns != 0 else 0

    # Optionally, you can also group by trade type.
    buy_df = df[df["Trade Type"].str.lower() == "buy"] if "Trade Type" in df.columns else pd.DataFrame()
    sell_df = df[df["Trade Type"].str.lower() == "sell"] if "Trade Type" in df.columns else pd.DataFrame()

    def group_metrics(group_df):
        if group_df.empty:
            return {}
        return {
            "Total Trades": len(group_df),
            "Wins": sum(1 for t in group_df["Profit (Currency)"] if t > 0),
            "Average Profit (%)": round(group_df["Profit (%)"].mean(), 2),
            "Average RRR": round(group_df["RRR"].mean(), 2) if "RRR" in group_df.columns else None,
            "Average Duration (min)": round(group_df["Trade Duration (min)"].mean(), 2) if "Trade Duration (min)" in group_df.columns else None,
            "Average Recovery Time (min)": round(group_df["Recovery Time (min)"].mean(), 2) if "Recovery Time (min)" in group_df.columns else None,
        }

    buy_metrics = group_metrics(buy_df)
    sell_metrics = group_metrics(sell_df)

    summary = {
        "Total Trades": total_trades,
        "Wins": wins,
        "Losses": losses,
        "Win/Loss Ratio": round(win_loss_ratio, 2),
        "Accuracy (%)": round(accuracy, 2),
        "Average Profit (Currency)": round(avg_profit_cur, 4),
        "Average Profit (%)": round(avg_profit_pct, 2),
        "Average Target (%)": round(avg_target_pct, 2),
        "Max Drawdown (%)": round(max_drawdown_pct, 2),
        "Average Drawdown (%)": round(avg_drawdown_pct, 2),
        "Average Trade Duration (min)": round(avg_duration, 2),
        "Average RRR": round(avg_rrr, 2),
        "Average Recovery Time (min)": round(avg_recovery, 2),
        "Return StdDev (%)": round(profit_std, 2),
        "Sharpe Ratio": round(sharpe_ratio, 2),
        "Buy Trade Metrics": buy_metrics,
        "Sell Trade Metrics": sell_metrics,
    }
    
    return summary

def save_summary_file(metrics: Dict[str, Any], file_name: str):
    try:
        file_exists = os.path.isfile(file_name)
        df_metrics = pd.DataFrame([metrics])
        df_metrics.to_csv(file_name, mode='a', header=not file_exists, index=False)
        logging.info(f"Appended metrics to summary file '{file_name}'")
    except Exception as e:
        logging.error(f"Error saving summary to {file_name}: {e}")

# Additional functions to add to existing statistics.py

def calculate_advanced_metrics(trades: List[Dict], initial_capital: float = 100000) -> Dict[str, Any]:
    """
    Calculate production-grade statistics for strategy evaluation.
    """
    if not trades:
        return _empty_advanced_metrics()
    
    # Convert to DataFrame for easier analysis
    trades_df = pd.DataFrame(trades)
    
    # Calculate returns series
    returns = calculate_returns_series(trades_df, initial_capital)
    
    # Basic metrics (existing)
    basic_metrics = calculate_metrics(trades)
    
    # Advanced metrics - only include implemented functions for now
    advanced_metrics = {
        **basic_metrics,
        
        # Risk-adjusted returns
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'sortino_ratio': calculate_sortino_ratio(returns),
        'calmar_ratio': calculate_calmar_ratio(returns),
        
        # Drawdown analysis
        'max_drawdown_duration': calculate_max_drawdown_duration(returns),
        
        # Trade analysis
        'profit_factor': calculate_profit_factor(trades_df),
        
        # Time-based analysis
        'monthly_returns': calculate_monthly_returns(returns),
        'rolling_sharpe': calculate_rolling_sharpe(returns),
        
        # Statistical metrics
        'skewness': returns.skew() if len(returns) > 0 else 0,
        'kurtosis': returns.kurtosis() if len(returns) > 0 else 0,
        
        # Stability metrics
        'stability_of_returns': calculate_stability(returns),
        'tail_ratio': calculate_tail_ratio(returns)
    }
    
    return advanced_metrics

def _empty_advanced_metrics() -> Dict[str, Any]:
    """
    Return empty metrics dictionary for cases with no trades.
    """
    return {
        # Basic metrics
        "Total Trades": 0,
        "Wins": 0,
        "Losses": 0,
        "Win/Loss Ratio": 0,
        "Accuracy (%)": 0,
        "Average Profit (Currency)": 0,
        "Average Profit (%)": 0,
        "Average Target (%)": 0,
        "Max Drawdown (%)": 0,
        "Average Drawdown (%)": 0,
        "Average Trade Duration (min)": 0,
        "Average RRR": 0,
        "Average Recovery Time (min)": 0,
        "Return StdDev (%)": 0,
        "Sharpe Ratio": 0,
        "Buy Trade Metrics": {},
        "Sell Trade Metrics": {},
        
        # Advanced metrics - simplified for now
        "sharpe_ratio": 0,
        "sortino_ratio": 0,
        "calmar_ratio": 0,
        "max_drawdown_duration": 0,
        "profit_factor": 0,
        "monthly_returns": {},
        "rolling_sharpe": [],
        "skewness": 0,
        "kurtosis": 0,
        "stability_of_returns": 0,
        "tail_ratio": 0
    }

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252
    
    if returns.std() == 0:
        return 0.0
    
    return np.sqrt(252) * excess_returns.mean() / returns.std()

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
   """Calculate Sortino ratio (uses downside deviation)."""
   if len(returns) < 2:
       return 0.0
   
   excess_returns = returns - risk_free_rate / 252
   downside_returns = returns[returns < 0]
   
   if len(downside_returns) == 0:
       return float('inf')
   
   downside_std = downside_returns.std()
   
   if downside_std == 0:
       return 0.0
   
   return np.sqrt(252) * excess_returns.mean() / downside_std

def calculate_calmar_ratio(returns: pd.Series) -> float:
   """Calculate Calmar ratio (annual return / max drawdown)."""
   if len(returns) < 2:
       return 0.0
   
   cum_returns = (1 + returns).cumprod()
   rolling_max = cum_returns.expanding().max()
   drawdown = (cum_returns - rolling_max) / rolling_max
   max_drawdown = abs(drawdown.min())
   
   if max_drawdown == 0:
       return float('inf')
   
   annual_return = returns.mean() * 252
   return annual_return / max_drawdown

def calculate_returns_series(trades_df: pd.DataFrame, initial_capital: float) -> pd.Series:
   """Convert trades to daily returns series."""
   if trades_df.empty:
       return pd.Series()
   
   # Create equity curve from trades
   equity = initial_capital
   equity_curve = []
   dates = []
   
   for _, trade in trades_df.iterrows():
       equity += trade['Profit (Currency)']
       equity_curve.append(equity)
       dates.append(trade['Exit Time'])
   
   # Convert to daily returns
   equity_series = pd.Series(equity_curve, index=pd.to_datetime(dates))
   equity_series = equity_series.resample('D').last().fillna(method='ffill')
   returns = equity_series.pct_change().dropna()
   
   return returns

def calculate_max_drawdown_duration(returns: pd.Series) -> int:
   """Calculate maximum drawdown duration in days."""
   cum_returns = (1 + returns).cumprod()
   rolling_max = cum_returns.expanding().max()
   underwater = cum_returns < rolling_max
   
   # Find consecutive underwater periods
   drawdown_periods = []
   current_dd_start = None
   
   for date, is_underwater in underwater.items():
       if is_underwater and current_dd_start is None:
           current_dd_start = date
       elif not is_underwater and current_dd_start is not None:
           drawdown_periods.append((current_dd_start, date))
           current_dd_start = None
   
   if current_dd_start is not None:
       drawdown_periods.append((current_dd_start, underwater.index[-1]))
   
   if not drawdown_periods:
       return 0
   
   max_duration = max((end - start).days for start, end in drawdown_periods)
   return max_duration

def calculate_profit_factor(trades_df: pd.DataFrame) -> float:
   """Calculate profit factor (gross profits / gross losses)."""
   if trades_df.empty:
       return 0.0
   
   gross_profits = trades_df[trades_df['Profit (Currency)'] > 0]['Profit (Currency)'].sum()
   gross_losses = abs(trades_df[trades_df['Profit (Currency)'] < 0]['Profit (Currency)'].sum())
   
   if gross_losses == 0:
       return float('inf') if gross_profits > 0 else 0.0
   
   return gross_profits / gross_losses

def calculate_rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
   """Calculate rolling Sharpe ratio."""
   if len(returns) < window:
       return pd.Series()
   
   rolling_mean = returns.rolling(window).mean() * 252
   rolling_std = returns.rolling(window).std() * np.sqrt(252)
   
   rolling_sharpe = rolling_mean / rolling_std
   rolling_sharpe = rolling_sharpe.fillna(0)
   
   return rolling_sharpe

def calculate_monthly_returns(returns: pd.Series) -> pd.DataFrame:
   """Calculate monthly returns statistics."""
   if returns.empty:
       return pd.DataFrame()
   
   try:
       # Resample to monthly frequency and calculate metrics
       monthly_grouped = returns.resample('M')
       
       monthly_return = monthly_grouped.apply(lambda x: (1 + x).prod() - 1)
       volatility = monthly_grouped.apply(lambda x: x.std() * np.sqrt(21))
       sharpe = monthly_grouped.apply(lambda x: calculate_sharpe_ratio(x) if len(x) > 1 else 0)
       
       # Create DataFrame with results
       monthly = pd.DataFrame({
           'monthly_return': monthly_return,
           'volatility': volatility, 
           'sharpe': sharpe
       })
       
       return monthly
   except Exception:
       # Return empty DataFrame on error
       return pd.DataFrame()

def calculate_stability(returns: pd.Series) -> float:
   """Calculate stability of returns using R-squared of equity curve."""
   if len(returns) < 2:
       return 0.0
   
   cum_returns = (1 + returns).cumprod()
   
   # Fit linear regression to log cumulative returns
   x = np.arange(len(cum_returns))
   y = np.log(cum_returns)
   
   # Remove any infinite values
   mask = np.isfinite(y)
   x = x[mask]
   y = y[mask]
   
   if len(x) < 2:
       return 0.0
   
   # Calculate R-squared
   slope, intercept = np.polyfit(x, y, 1)
   y_pred = slope * x + intercept
   ss_res = np.sum((y - y_pred) ** 2)
   ss_tot = np.sum((y - np.mean(y)) ** 2)
   
   r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
   
   return r_squared

def calculate_tail_ratio(returns: pd.Series) -> float:
   """Calculate tail ratio (right tail / left tail)."""
   if len(returns) < 10:
       return 0.0
   
   # Use 95th and 5th percentiles
   right_tail = returns.quantile(0.95)
   left_tail = abs(returns.quantile(0.05))
   
   if left_tail == 0:
       return float('inf') if right_tail > 0 else 0.0
   
   return right_tail / left_tail
