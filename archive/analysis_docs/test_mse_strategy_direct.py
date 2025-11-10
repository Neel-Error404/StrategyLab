#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct test of MSE strategy with our data
"""

import sys
import os
from pathlib import Path

# Set UTF-8 encoding for Windows PowerShell
if os.name == 'nt':  # Windows
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.append(str(Path(__file__).resolve().parent))

from src.strategies.register_strategies import register_all_strategies
from src.strategies.strategy_factory import StrategyFactory
from src.core.etl.loader import load_multi_timeframe_data

def test_mse_strategy_direct():
    """Test MSE strategy directly with our data"""
    
    # Register strategies
    print("🔧 Registering strategies...")
    register_all_strategies()
    
    # Create strategy instance
    print("🎯 Creating MSE backtesting strategy...")
    strategy = StrategyFactory.create_strategy('mse_backtesting')
    if not strategy:
        print("❌ Failed to create strategy")
        return
    
    print(f"✅ Strategy created: {strategy.name}")
    print(f"📊 Required timeframes: {strategy.required_timeframes}")
    print(f"⏱️ Warmup periods: {strategy.warmup_periods}")
    
    # Load data
    print("\n📁 Loading multi-timeframe data for RELIANCE...")
    data = load_multi_timeframe_data(
        '2022-01-01_to_2025-08-31', 
        'RELIANCE', 
        strategy.required_timeframes
    )
    
    if not data:
        print("❌ No data loaded")
        return
        
    print(f"✅ Data loaded successfully:")
    for tf, df in data.items():
        print(f"   {tf}: {len(df)} records ({df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']})")
    
    # Prepare data using strategy method
    print("\n🔄 Preparing data with strategy...")
    try:
        prepared_data = strategy.prepare_data(data, 'RELIANCE', '2022-01-01_to_2025-08-31')
        print(f"✅ Data prepared successfully:")
        for tf, df in prepared_data.items():
            print(f"   {tf}: {len(df)} records with {len(df.columns)} columns")
            # Check for indicators
            indicator_cols = [col for col in df.columns if any(x in col for x in ['macd', 'ema', 'signal'])]
            print(f"      Indicators: {indicator_cols}")
    except Exception as e:
        print(f"❌ Error preparing data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate signals
    print("\n🎲 Generating signals...")
    try:
        signals_df = strategy.generate_signals(prepared_data)
        print(f"✅ Signals generated: {len(signals_df)} records")
        
        # Check for signals
        buy_signals = signals_df['entry_signal_buy'].sum() if 'entry_signal_buy' in signals_df.columns else 0
        sell_signals = signals_df['entry_signal_sell'].sum() if 'entry_signal_sell' in signals_df.columns else 0
        print(f"   Buy signals: {buy_signals}")
        print(f"   Sell signals: {sell_signals}")
        
        if buy_signals + sell_signals == 0:
            print("⚠️ No trading signals generated")
        
    except Exception as e:
        print(f"❌ Error generating signals: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Execute strategy
    print("\n⚡ Executing strategy...")
    try:
        trades_df, summary_stats = strategy.execute_strategy(signals_df)
        print(f"✅ Strategy executed successfully:")
        print(f"   Total trades: {summary_stats.get('total_trades', 0)}")
        print(f"   Winning trades: {summary_stats.get('winning_trades', 0)}")
        print(f"   Win rate: {summary_stats.get('win_rate', 0):.2f}%")
        print(f"   Total return: {summary_stats.get('total_return', 0):.2f}")
        print(f"   Total return %: {summary_stats.get('total_return_percent', 0):.2f}%")
        
        if len(trades_df) > 0:
            print(f"\n📈 Sample trades:")
            print(trades_df.head())
        
    except Exception as e:
        print(f"❌ Error executing strategy: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n🎉 MSE strategy test completed successfully!")

if __name__ == "__main__":
    test_mse_strategy_direct()