#!/usr/bin/env python3
"""
Sample Market Data Generator
Creates realistic market data for backtesting system testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import List, Dict, Any

def generate_realistic_price_series(
    start_price: float = 100.0,
    num_days: int = 252,
    annual_volatility: float = 0.2,
    annual_drift: float = 0.08,
    start_date: str = "2023-01-01"
) -> pd.DataFrame:
    """
    Generate realistic stock price data using geometric Brownian motion.
    
    Args:
        start_price: Initial stock price
        num_days: Number of trading days
        annual_volatility: Annual volatility (e.g., 0.2 = 20%)
        annual_drift: Annual expected return (e.g., 0.08 = 8%)
        start_date: Starting date for the series
    
    Returns:
        DataFrame with Date, Open, High, Low, Close, Volume columns
    """
    np.random.seed(42)  # For reproducible data
    
    # Daily parameters
    dt = 1/252  # Daily time step
    daily_drift = annual_drift * dt
    daily_vol = annual_volatility * np.sqrt(dt)
    
    # Generate price path using geometric Brownian motion
    returns = np.random.normal(daily_drift, daily_vol, num_days)
    prices = np.zeros(num_days + 1)
    prices[0] = start_price
    
    for i in range(1, num_days + 1):
        prices[i] = prices[i-1] * np.exp(returns[i-1])
    
    # Create dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    dates = [start_dt + timedelta(days=i) for i in range(num_days)]
    
    # Remove weekends (simple approximation)
    weekday_dates = [d for d in dates if d.weekday() < 5][:num_days]
    
    # Generate OHLC data
    closes = prices[1:num_days+1]
    
    # Opens are previous close + small gap
    opens = np.roll(closes, 1)
    opens[0] = start_price
    gap_factor = np.random.normal(1.0, 0.002, num_days)  # Small gaps
    opens = opens * gap_factor
    
    # Highs and lows based on intraday volatility
    intraday_vol = daily_vol * 0.5  # Intraday volatility
    high_factor = np.random.lognormal(0, intraday_vol, num_days)
    low_factor = np.random.lognormal(0, intraday_vol, num_days)
    
    highs = np.maximum(opens, closes) * (1 + high_factor)
    lows = np.minimum(opens, closes) * (1 - low_factor)
    
    # Ensure OHLC consistency
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))
    
    # Generate realistic volume (log-normal distribution)
    avg_volume = 1000000
    volumes = np.random.lognormal(np.log(avg_volume), 0.5, num_days).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': weekday_dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    })
    
    # Round prices to 2 decimal places
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = df[col].round(2)
    
    return df

def create_sample_datasets() -> None:
    """Create sample datasets for different market scenarios."""
    
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data_pools")
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Bull Market Stock (AAPL-like)
    bull_data = generate_realistic_price_series(
        start_price=150.0,
        num_days=252,
        annual_volatility=0.25,
        annual_drift=0.15,
        start_date="2023-01-01"
    )
    bull_data.to_csv(f"{data_dir}/AAPL_2023.csv", index=False)
    print("✓ Created bull market data: AAPL_2023.csv")
    
    # 2. Volatile Tech Stock (TSLA-like)
    volatile_data = generate_realistic_price_series(
        start_price=200.0,
        num_days=252,
        annual_volatility=0.45,
        annual_drift=0.10,
        start_date="2023-01-01"
    )
    volatile_data.to_csv(f"{data_dir}/TSLA_2023.csv", index=False)
    print("✓ Created volatile stock data: TSLA_2023.csv")
    
    # 3. Stable Dividend Stock (KO-like)
    stable_data = generate_realistic_price_series(
        start_price=60.0,
        num_days=252,
        annual_volatility=0.15,
        annual_drift=0.06,
        start_date="2023-01-01"
    )
    stable_data.to_csv(f"{data_dir}/KO_2023.csv", index=False)
    print("✓ Created stable stock data: KO_2023.csv")
    
    # 4. Market Index (SPY-like)
    index_data = generate_realistic_price_series(
        start_price=400.0,
        num_days=252,
        annual_volatility=0.18,
        annual_drift=0.08,
        start_date="2023-01-01"
    )
    index_data.to_csv(f"{data_dir}/SPY_2023.csv", index=False)
    print("✓ Created market index data: SPY_2023.csv")
    
    # 5. Bear Market Scenario
    bear_data = generate_realistic_price_series(
        start_price=100.0,
        num_days=252,
        annual_volatility=0.30,
        annual_drift=-0.20,
        start_date="2023-01-01"
    )
    bear_data.to_csv(f"{data_dir}/BEAR_2023.csv", index=False)
    print("✓ Created bear market data: BEAR_2023.csv")
    
    # 6. Create multi-year data for longer backtests
    long_data = generate_realistic_price_series(
        start_price=100.0,
        num_days=252*3,  # 3 years
        annual_volatility=0.22,
        annual_drift=0.10,
        start_date="2021-01-01"
    )
    long_data.to_csv(f"{data_dir}/MSFT_3Y.csv", index=False)
    print("✓ Created 3-year data: MSFT_3Y.csv")
    
    # Create a summary file
    summary = {
        "datasets": [
            {"file": "AAPL_2023.csv", "description": "Bull market tech stock", "volatility": "25%", "drift": "15%"},
            {"file": "TSLA_2023.csv", "description": "High volatility growth stock", "volatility": "45%", "drift": "10%"},
            {"file": "KO_2023.csv", "description": "Stable dividend stock", "volatility": "15%", "drift": "6%"},
            {"file": "SPY_2023.csv", "description": "Market index ETF", "volatility": "18%", "drift": "8%"},
            {"file": "BEAR_2023.csv", "description": "Bear market scenario", "volatility": "30%", "drift": "-20%"},
            {"file": "MSFT_3Y.csv", "description": "3-year technology stock", "volatility": "22%", "drift": "10%"}
        ],
        "format": "CSV with columns: Date, Open, High, Low, Close, Volume",
        "date_range": "Trading days only (weekdays)",
        "note": "Generated using geometric Brownian motion with realistic parameters"
    }
    
    import json
    with open(f"{data_dir}/README.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("✓ Created dataset summary: README.json")
    
    print(f"\n📊 Sample data created in: {data_dir}")
    print("🎯 Ready for backtesting!")

if __name__ == "__main__":
    create_sample_datasets()
