#!/usr/bin/env python3
"""
Analyze market conditions on days with multiple consecutive same-direction trades
to understand if they occur during volatile, choppy, or directional markets
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_market_conditions():
    """
    Analyze market conditions on days with consecutive same-direction trades
    """
    
    print("🔍 MARKET CONDITIONS ANALYSIS FOR CONSECUTIVE TRADES")
    print("=" * 65)
    
    # Load the original consolidated trades data
    file_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/basic-config/code/Users/back_tester/backtester/consolidated_trades.csv"
    df = pd.read_csv(file_path)
    
    # Convert datetime columns
    df['Entry Time'] = pd.to_datetime(df['Entry Time'])
    df['Exit Time'] = pd.to_datetime(df['Exit Time'])
    df['Entry Date'] = df['Entry Time'].dt.date
    
    print(f"📊 Loaded {len(df):,} total trades")
    
    # Focus on the 30 tickers we analyzed for consistency
    target_tickers = ['ASTRAL', 'ADANIGREEN', 'PVRINOX', 'BIRLACORPN', 'ACCELYA', 
                     'BATAINDIA', 'DLF', 'BIOCON', 'ADANIPORTS', 'ADANIENSOL', 
                     'OLECTRA', 'CANFINHOME', 'LODHA', 'NELCO', 'MANYAVAR', 
                     'CLEAN', 'CREDITACC', 'IRCTC', 'ASAHIINDIA', 'INTELLECT', 
                     'TATACHEM', 'FINCABLES', 'INDUSINDBK', 'IPCALAB', 'INOXINDIA', 
                     'CONCOR', 'JINDALSTEL', 'MEDANTA', 'VOLTAS', 'JSWENERGY']
    
    df_filtered = df[df['ticker'].isin(target_tickers)].copy()
    print(f"🎯 Analyzing {len(df_filtered):,} trades from {len(target_tickers)} tickers")
    
    # Identify days with consecutive same-direction trades
    consecutive_days = []
    normal_days = []
    
    for ticker in target_tickers:
        ticker_data = df_filtered[df_filtered['ticker'] == ticker].sort_values('Entry Time')
        
        # Group by date
        daily_groups = ticker_data.groupby('Entry Date')
        
        for date, day_trades in daily_groups:
            if len(day_trades) < 2:
                continue
                
            # Check for consecutive same-direction trades
            has_consecutive = False
            trade_types = day_trades['Trade Type'].tolist()
            
            for i in range(len(trade_types) - 1):
                if trade_types[i] == trade_types[i + 1]:
                    has_consecutive = True
                    break
            
            # Calculate market condition metrics for this day
            day_metrics = calculate_daily_market_metrics(day_trades, ticker, date)
            
            if has_consecutive:
                consecutive_days.append(day_metrics)
            else:
                normal_days.append(day_metrics)
    
    print(f"\n📈 Found {len(consecutive_days)} days with consecutive same-direction trades")
    print(f"📉 Found {len(normal_days)} normal trading days for comparison")
    
    # Analyze market conditions
    analyze_conditions(consecutive_days, normal_days)
    
    return consecutive_days, normal_days

def calculate_daily_market_metrics(day_trades, ticker, date):
    """
    Calculate market condition metrics for a single day
    """
    
    # Basic trade information
    num_trades = len(day_trades)
    trade_types = day_trades['Trade Type'].tolist()
    entry_times = pd.to_datetime(day_trades['Entry Time'])
    
    # Price action metrics using High/Low during trades
    highs = day_trades['High During Trade'].values
    lows = day_trades['Low During Trade'].values
    entry_prices = day_trades['Entry Price'].values
    exit_prices = day_trades['Exit Price'].values
    
    # Calculate daily range and volatility
    daily_high = max(highs) if len(highs) > 0 else 0
    daily_low = min(lows) if len(lows) > 0 else 0
    daily_range = daily_high - daily_low if daily_high > 0 and daily_low > 0 else 0
    daily_range_pct = (daily_range / daily_low * 100) if daily_low > 0 else 0
    
    # Calculate intraday volatility (average of individual trade ranges)
    trade_ranges = []
    for i in range(len(day_trades)):
        trade_range = (highs[i] - lows[i]) / entry_prices[i] * 100 if entry_prices[i] > 0 else 0
        trade_ranges.append(trade_range)
    
    avg_trade_volatility = np.mean(trade_ranges) if trade_ranges else 0
    
    # Direction analysis
    first_price = entry_prices[0] if len(entry_prices) > 0 else 0
    last_price = exit_prices[-1] if len(exit_prices) > 0 else 0
    daily_direction_pct = ((last_price - first_price) / first_price * 100) if first_price > 0 else 0
    
    # Profit analysis
    profits = day_trades['Profit (%)'].values
    total_daily_pnl = sum(profits)
    avg_trade_pnl = np.mean(profits) if len(profits) > 0 else 0
    profitable_trades = sum(1 for p in profits if p > 0)
    win_rate = profitable_trades / len(profits) * 100 if len(profits) > 0 else 0
    
    # Trading pattern analysis
    buy_count = sum(1 for t in trade_types if t == 'Buy')
    sell_count = sum(1 for t in trade_types if t == 'Sell')
    directional_bias = 'Buy' if buy_count > sell_count else 'Sell' if sell_count > buy_count else 'Neutral'
    
    # Time span analysis
    first_trade_time = entry_times.min()
    last_trade_time = pd.to_datetime(day_trades['Exit Time']).max()
    trading_span_hours = (last_trade_time - first_trade_time).total_seconds() / 3600
    
    # Market condition classification
    market_condition = classify_market_condition(daily_range_pct, avg_trade_volatility, 
                                               daily_direction_pct, num_trades)
    
    return {
        'ticker': ticker,
        'date': date,
        'num_trades': num_trades,
        'daily_range_pct': daily_range_pct,
        'avg_trade_volatility': avg_trade_volatility,
        'daily_direction_pct': daily_direction_pct,
        'total_daily_pnl': total_daily_pnl,
        'avg_trade_pnl': avg_trade_pnl,
        'win_rate': win_rate,
        'buy_count': buy_count,
        'sell_count': sell_count,
        'directional_bias': directional_bias,
        'trading_span_hours': trading_span_hours,
        'market_condition': market_condition,
        'trade_sequence': ''.join(trade_types)
    }

def classify_market_condition(daily_range_pct, avg_volatility, direction_pct, num_trades):
    """
    Classify market condition based on metrics
    """
    
    # Thresholds (these can be adjusted based on data analysis)
    HIGH_VOLATILITY = 3.0  # 3%+ daily range
    HIGH_DIRECTION = 2.0   # 2%+ directional move
    HIGH_TRADE_COUNT = 4   # 4+ trades in a day
    
    # Classification logic
    is_volatile = daily_range_pct > HIGH_VOLATILITY
    is_directional = abs(direction_pct) > HIGH_DIRECTION
    is_active = num_trades > HIGH_TRADE_COUNT
    
    if is_volatile and is_directional:
        return "TRENDING_VOLATILE"
    elif is_volatile and not is_directional:
        return "CHOPPY_VOLATILE" 
    elif is_directional and not is_volatile:
        return "TRENDING_CALM"
    elif is_active and is_volatile:
        return "ACTIVE_VOLATILE"
    elif is_active:
        return "ACTIVE_NORMAL"
    else:
        return "NORMAL"

def analyze_conditions(consecutive_days, normal_days):
    """
    Compare market conditions between consecutive trade days and normal days
    """
    
    print(f"\n🎯 MARKET CONDITION ANALYSIS")
    print("-" * 35)
    
    if not consecutive_days or not normal_days:
        print("❌ Insufficient data for comparison")
        return
    
    # Convert to DataFrames for easier analysis
    cons_df = pd.DataFrame(consecutive_days)
    norm_df = pd.DataFrame(normal_days)
    
    print(f"\n📊 VOLATILITY COMPARISON:")
    print(f"{'Metric':<25} {'Consecutive':<15} {'Normal':<15} {'Difference':<15}")
    print("-" * 70)
    
    # Daily range comparison
    cons_range = cons_df['daily_range_pct'].mean()
    norm_range = norm_df['daily_range_pct'].mean()
    print(f"{'Daily Range %':<25} {cons_range:<15.2f} {norm_range:<15.2f} {cons_range-norm_range:<15.2f}")
    
    # Trade volatility comparison
    cons_vol = cons_df['avg_trade_volatility'].mean()
    norm_vol = norm_df['avg_trade_volatility'].mean()
    print(f"{'Avg Trade Volatility %':<25} {cons_vol:<15.2f} {norm_vol:<15.2f} {cons_vol-norm_vol:<15.2f}")
    
    # Direction comparison
    cons_dir = cons_df['daily_direction_pct'].mean()
    norm_dir = norm_df['daily_direction_pct'].mean()
    print(f"{'Daily Direction %':<25} {cons_dir:<15.2f} {norm_dir:<15.2f} {cons_dir-norm_dir:<15.2f}")
    
    # Trade count comparison
    cons_trades = cons_df['num_trades'].mean()
    norm_trades = norm_df['num_trades'].mean()
    print(f"{'Trades per Day':<25} {cons_trades:<15.1f} {norm_trades:<15.1f} {cons_trades-norm_trades:<15.1f}")
    
    print(f"\n📈 PERFORMANCE COMPARISON:")
    print("-" * 30)
    
    # PnL comparison
    cons_pnl = cons_df['total_daily_pnl'].mean()
    norm_pnl = norm_df['total_daily_pnl'].mean()
    print(f"{'Avg Daily PnL %':<25} {cons_pnl:<15.2f} {norm_pnl:<15.2f} {cons_pnl-norm_pnl:<15.2f}")
    
    # Per-trade PnL comparison
    cons_trade_pnl = cons_df['avg_trade_pnl'].mean()
    norm_trade_pnl = norm_df['avg_trade_pnl'].mean()
    print(f"{'Avg Trade PnL %':<25} {cons_trade_pnl:<15.3f} {norm_trade_pnl:<15.3f} {cons_trade_pnl-norm_trade_pnl:<15.3f}")
    
    # Win rate comparison
    cons_wr = cons_df['win_rate'].mean()
    norm_wr = norm_df['win_rate'].mean()
    print(f"{'Win Rate %':<25} {cons_wr:<15.1f} {norm_wr:<15.1f} {cons_wr-norm_wr:<15.1f}")
    
    print(f"\n🏷️ MARKET CONDITION BREAKDOWN:")
    print("-" * 35)
    
    # Condition distribution for consecutive days
    cons_conditions = cons_df['market_condition'].value_counts()
    print(f"\nConsecutive Trade Days:")
    for condition, count in cons_conditions.items():
        pct = count / len(cons_df) * 100
        print(f"  {condition:<20} {count:>4} ({pct:>5.1f}%)")
    
    # Condition distribution for normal days
    norm_conditions = norm_df['market_condition'].value_counts()
    print(f"\nNormal Trade Days:")
    for condition, count in norm_conditions.items():
        pct = count / len(norm_df) * 100
        print(f"  {condition:<20} {count:>4} ({pct:>5.1f}%)")
    
    print(f"\n🎯 KEY INSIGHTS:")
    print("-" * 15)
    
    # Volatility insight
    if cons_range > norm_range * 1.2:
        print(f"✅ Consecutive trade days are {cons_range/norm_range:.1f}x more volatile")
    elif cons_range < norm_range * 0.8:
        print(f"✅ Consecutive trade days are {norm_range/cons_range:.1f}x less volatile")
    else:
        print(f"➡️ Similar volatility between consecutive and normal days")
    
    # Performance insight
    if cons_pnl > norm_pnl * 1.1:
        print(f"✅ Consecutive trade days are MORE profitable (+{cons_pnl-norm_pnl:.2f}% daily)")
    elif cons_pnl < norm_pnl * 0.9:
        print(f"❌ Consecutive trade days are LESS profitable ({cons_pnl-norm_pnl:.2f}% daily)")
    else:
        print(f"➡️ Similar profitability between consecutive and normal days")
    
    # Direction insight
    if abs(cons_dir) > abs(norm_dir) * 1.5:
        print(f"✅ Consecutive trade days show stronger directional moves")
    else:
        print(f"➡️ Similar directional characteristics")
    
    # Most common patterns
    print(f"\n📋 MOST COMMON CONSECUTIVE PATTERNS:")
    pattern_counts = cons_df['trade_sequence'].value_counts().head(10)
    for pattern, count in pattern_counts.items():
        pct = count / len(cons_df) * 100
        print(f"  {pattern:<15} {count:>4} ({pct:>5.1f}%)")

if __name__ == "__main__":
    consecutive_days, normal_days = analyze_market_conditions()