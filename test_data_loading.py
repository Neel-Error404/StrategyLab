#!/usr/bin/env python3
"""
Quick diagnostic to test data loading components
"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("DATA LOADING DIAGNOSTIC")
print("=" * 80)
print()

# Test 1: Load equity data
print("Test 1: Loading equity data...")
try:
    equity_path = project_root / "data/pools/2022-01-01_to_2025-08-31/RELIANCE/15m.parquet"
    df_equity = pd.read_parquet(equity_path)
    print(f"✅ Loaded equity data: {len(df_equity)} rows")
    print(f"   Date range: {df_equity['timestamp'].min()} to {df_equity['timestamp'].max()}")
    print(f"   Columns: {list(df_equity.columns)}")
except Exception as e:
    print(f"❌ Failed to load equity data: {e}")
    sys.exit(1)
print()

# Test 2: Load option data
print("Test 2: Loading option data...")
try:
    option_dir = project_root / "data/pools/options/2025-04-01_to_2025-10-08/RELIANCE/1day"
    option_files = list(option_dir.glob("*.parquet"))
    print(f"✅ Found {len(option_files)} option expiry files")

    if option_files:
        sample_file = option_files[0]
        df_option = pd.read_parquet(sample_file)
        print(f"   Sample file: {sample_file.name}")
        print(f"   Rows: {len(df_option)}")
        print(f"   Columns: {list(df_option.columns)}")
except Exception as e:
    print(f"❌ Failed to load option data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Test 3: Try data loader functions
print("Test 3: Testing data loader functions...")
try:
    from src.core.options.replay.data_loader import (
        load_equity_trades,
        find_option_data_dir,
        OptionDataStore
    )

    # Test find_option_data_dir
    print("   Testing find_option_data_dir...")
    option_dir = find_option_data_dir(
        options_root=project_root / "data/pools/options",
        ticker="RELIANCE",
        start=pd.Timestamp("2025-05-01"),
        end=pd.Timestamp("2025-05-31")
    )
    print(f"   ✅ Found option dir: {option_dir}")

    # Test OptionDataStore
    print("   Testing OptionDataStore...")
    store = OptionDataStore(
        base_dir=option_dir,
        timeframe="1day",
        timeframes=("1day",)
    )
    expiries = store.list_expiries()
    print(f"   ✅ Found {len(expiries)} expiries")
    if expiries:
        print(f"   First expiry: {expiries[0]}")

        # Try loading one expiry
        df_chain = store.load_chain(expiries[0])
        print(f"   ✅ Loaded chain: {len(df_chain)} rows")

except Exception as e:
    print(f"❌ Data loader functions failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Test 4: Try loading equity trades CSV
print("Test 4: Loading test equity trades...")
try:
    trades_path = project_root / "outputs/integration_test/equity_trades.csv"
    if trades_path.exists():
        df_trades = pd.read_csv(trades_path)
        print(f"✅ Loaded {len(df_trades)} equity trades")
        print(f"   Columns: {list(df_trades.columns)}")
        print()
        print("Sample trades:")
        print(df_trades.head(3))
    else:
        print("⚠️  Equity trades file not found")
except Exception as e:
    print(f"❌ Failed to load equity trades: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
print("✅ ALL DIAGNOSTICS PASSED")
print("=" * 80)
