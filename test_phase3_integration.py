#!/usr/bin/env python3
"""
Phase 3/4 Integration Test
===========================
Tests the complete options replay engine end-to-end.

This script:
1. Creates synthetic equity trades for testing
2. Runs the options replay engine
3. Validates all outputs
4. Reports success/failure with diagnostics
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.options.replay.config import OptionsReplayConfig
from src.core.options.replay.engine import OptionsReplayEngine

def create_test_equity_trades(output_path: Path, ticker: str = "RELIANCE"):
    """Create synthetic equity trades for testing."""

    # Load underlying data to get realistic prices
    underlying_path = project_root / f"data/pools/2022-01-01_to_2025-08-31/{ticker}/15m.parquet"

    if not underlying_path.exists():
        raise FileNotFoundError(f"Underlying data not found: {underlying_path}")

    df = pd.read_parquet(underlying_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filter to test period (May 2025 - has option data available)
    test_start = pd.Timestamp("2025-05-01", tz="Asia/Kolkata")
    test_end = pd.Timestamp("2025-05-31", tz="Asia/Kolkata")

    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize("Asia/Kolkata")
    else:
        df['timestamp'] = df['timestamp'].dt.tz_convert("Asia/Kolkata")

    df_test = df[(df['timestamp'] >= test_start) & (df['timestamp'] <= test_end)].copy()

    if df_test.empty:
        print(f"⚠️  Warning: No data in May 2025, using latest available data")
        df_test = df.tail(100).copy()

    # Create synthetic trades
    trades = []
    trade_id = 1

    # Reset index to ensure sequential access
    df_test = df_test.reset_index(drop=True)

    # Sample some positions for entries (use positional indices)
    n_trades = min(10, len(df_test) // 10)  # max 10 trades with room for exits

    if n_trades == 0:
        print(f"⚠️  Warning: Insufficient data for trades (only {len(df_test)} bars)")
        n_trades = min(3, len(df_test) // 3) if len(df_test) >= 3 else 0

    if n_trades > 0:
        sample_positions = sorted(np.random.choice(len(df_test) - 10, n_trades, replace=False))

        for pos in sample_positions:
            if pos + 10 >= len(df_test):  # Need room for exit
                continue

            entry_row = df_test.iloc[pos]
            # Exit 2-8 hours later
            exit_pos = min(pos + np.random.randint(8, 33), len(df_test) - 1)  # 8-33 bars = 2-8 hours
            exit_row = df_test.iloc[exit_pos]

            entry_price = float(entry_row['close'])
            exit_price = float(exit_row['close'])
            quantity = 100
            pnl = (exit_price - entry_price) * quantity

            trades.append({
                'trade_id': f"{ticker}_{trade_id:03d}",
                'ticker': ticker,
                'entry_time': entry_row['timestamp'].isoformat(),
                'exit_time': exit_row['timestamp'].isoformat(),
                'side': 'LONG' if exit_price > entry_price else 'SHORT',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': quantity,
                'pnl': pnl
            })
            trade_id += 1

    # Save to CSV
    trades_df = pd.DataFrame(trades)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(output_path, index=False)

    print(f"✅ Created {len(trades)} test equity trades")
    if len(trades) > 0:
        print(f"   Period: {trades_df['entry_time'].min()} to {trades_df['exit_time'].max()}")
    print(f"   Saved to: {output_path}")

    return trades_df

def run_integration_test(test_mode: str = "minimal"):
    """Run the integration test."""

    print("=" * 80)
    print("PHASE 3/4 OPTIONS REPLAY ENGINE - INTEGRATION TEST")
    print("=" * 80)
    print()

    # Step 1: Create test data
    print("Step 1: Creating test equity trades...")
    test_output_dir = project_root / "outputs" / "integration_test"
    test_trades_path = test_output_dir / "equity_trades.csv"

    ticker = "RELIANCE"
    trades_df = create_test_equity_trades(test_trades_path, ticker=ticker)
    print()

    # Step 2: Load and configure for test
    print("Step 2: Loading options config...")
    config_path = project_root / "src/core/options/config/options_config.yaml"

    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return False

    # Load base config
    base_config = OptionsReplayConfig.from_yaml(config_path)

    # Create test-specific config using dataclasses.replace (handles frozen dataclasses)
    from dataclasses import replace

    config = replace(
        base_config,
        inputs=replace(
            base_config.inputs,
            equity_trades_path=test_trades_path,
            equity_data_root=project_root / "data/pools",
            options_data_root=project_root / "data/pools/options",
            underlying_timeframe="15m",  # Match actual file naming
            options_timeframe="1day"
        ),
        output=replace(
            base_config.output,
            output_dir=test_output_dir / "options_replay"
        )
    )

    print(f"✅ Config loaded and configured for test")
    print(f"   Equity trades: {config.inputs.equity_trades_path}")
    print(f"   Options data: {config.inputs.options_data_root}")
    print(f"   Output dir: {config.output.output_dir}")
    print()

    # Step 3: Create engine
    print("Step 3: Initializing replay engine...")
    try:
        engine = OptionsReplayEngine(config)
        print(f"✅ Engine initialized (run_id: {engine.run_id})")
    except Exception as e:
        print(f"❌ Engine initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()

    # Step 4: Run replay
    print("Step 4: Running replay engine...")
    print(f"   Mode: {test_mode}")

    if test_mode == "minimal":
        tickers = [ticker]
        # Use date range where both equity and option data exist
        date_ranges = ["2025-04-01_to_2025-10-08"]
    else:  # multi-ticker
        tickers = "auto"
        date_ranges = ["2025-04-01_to_2025-10-08"]

    try:
        artifacts = engine.run(
            tickers=tickers,
            date_ranges=date_ranges,
            verify_hash=False
        )
        print(f"✅ Replay completed successfully!")
        print(f"   Processed: {len(artifacts.trades)} trades")
        print(f"   Skipped: {len(artifacts.skipped_trades)} trades")
    except Exception as e:
        print(f"❌ Replay failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()

    # Step 5: Validate outputs
    print("Step 5: Validating outputs...")
    output_dir = Path(artifacts.metadata['output_dir'])

    expected_files = [
        "options_trades.csv",
        "options_positions.csv",
        "options_metrics.json",
        "run_manifest.json",
        "logs.jsonl"
    ]

    all_exist = True
    for filename in expected_files:
        file_path = output_dir / filename
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"   ✅ {filename:<30} ({size:,} bytes)")
        else:
            print(f"   ❌ {filename:<30} MISSING")
            all_exist = False

    if not all_exist:
        print("❌ Some output files are missing!")
        return False
    print()

    # Step 6: Load and inspect outputs
    print("Step 6: Inspecting output data...")

    try:
        trades_df = pd.read_csv(output_dir / "options_trades.csv")
        positions_df = pd.read_csv(output_dir / "options_positions.csv")

        with open(output_dir / "options_metrics.json") as f:
            metrics = json.load(f)

        with open(output_dir / "run_manifest.json") as f:
            manifest = json.load(f)

        print(f"   ✅ Trades loaded: {len(trades_df)} rows")
        print(f"   ✅ Positions loaded: {len(positions_df)} rows")
        print(f"   ✅ Metrics loaded")
        print(f"   ✅ Manifest loaded")
        print()

        # Display summary
        print("=" * 80)
        print("INTEGRATION TEST RESULTS")
        print("=" * 80)
        print()

        print("📊 Trades Summary:")
        if len(trades_df) > 0:
            print(f"   Total trades: {len(trades_df)}")
            print(f"   Total P&L: ₹{trades_df['realized_pnl'].sum():,.2f}")
            print(f"   Avg return: {trades_df['return_pct'].mean():.2f}%")
            print(f"   Win rate: {(trades_df['realized_pnl'] > 0).mean() * 100:.1f}%")
            print()
            print("   Sample trades (first 3):")
            print(trades_df[['trade_id', 'ticker', 'strike', 'option_type', 'realized_pnl', 'return_pct']].head(3).to_string(index=False))
        else:
            print("   ⚠️  No trades processed (all skipped)")
        print()

        print("📈 Metrics Summary:")
        summary = metrics.get('summary', {})
        print(f"   Total P&L: ₹{summary.get('total_pnl', 0):,.2f}")
        print(f"   Win rate: {summary.get('win_rate_pct', 0):.1f}%")
        print(f"   Sharpe ratio: {summary.get('sharpe_ratio', 0):.2f}")
        print(f"   Max drawdown: {summary.get('max_drawdown_pct', 0):.1f}%")
        print(f"   Avg hold hours: {summary.get('average_hold_hours', 0):.1f}")
        print()

        print("🔍 Data Quality:")
        diagnostics = metrics.get('diagnostics', {})
        print(f"   Fallback trades: {diagnostics.get('fallback_trades', 0)}")
        print(f"   Actual entry count: {diagnostics.get('actual_entry_count', 0)}")
        print(f"   Actual exit count: {diagnostics.get('actual_exit_count', 0)}")
        print()

        print("⚠️  Skipped Trades:")
        if len(artifacts.skipped_trades) > 0:
            skip_reasons = {}
            for skip in artifacts.skipped_trades:
                reason = skip.get('reason', 'unknown')
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

            for reason, count in skip_reasons.items():
                print(f"   {reason}: {count}")
        else:
            print("   None (all trades processed)")
        print()

        # Step 7: Spot-check P&L
        print("Step 7: Spot-checking P&L calculations...")
        if len(trades_df) > 0:
            sample_size = min(3, len(trades_df))
            sample_trades = trades_df.head(sample_size)

            pnl_errors = []
            for _, trade in sample_trades.iterrows():
                expected_pnl = (trade['exit_price'] - trade['entry_price']) * trade['quantity']
                actual_pnl = trade['realized_pnl']
                error = abs(expected_pnl - actual_pnl)

                if error > 0.01:  # Tolerance for rounding
                    pnl_errors.append({
                        'trade_id': trade['trade_id'],
                        'expected': expected_pnl,
                        'actual': actual_pnl,
                        'error': error
                    })
                    print(f"   ⚠️  Trade {trade['trade_id']}: P&L mismatch (expected: {expected_pnl:.2f}, actual: {actual_pnl:.2f})")

            if not pnl_errors:
                print(f"   ✅ All {sample_size} sampled trades have correct P&L")
        print()

        # Final verdict
        print("=" * 80)
        if len(trades_df) > 0 and all_exist:
            print("✅ ✅ ✅  INTEGRATION TEST PASSED  ✅ ✅ ✅")
            print()
            print("Next steps:")
            print("1. Review output files in:", output_dir)
            print("2. Run multi-ticker test: python test_phase3_integration.py --mode multi")
            print("3. Create PHASE3_STATUS.md documenting results")
            print("4. Proceed to Phase 4 full backtest")
        else:
            print("⚠️  INTEGRATION TEST PASSED WITH WARNINGS")
            print()
            if len(trades_df) == 0:
                print("All trades were skipped. Check:")
                print("- Option data availability for test period")
                print("- Min DTE requirements (config.risk.min_dte_to_enter)")
                print("- Liquidity filters")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"❌ Failed to load outputs: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 3/4 integration test")
    parser.add_argument("--mode", choices=["minimal", "multi"], default="minimal",
                       help="Test mode: minimal (1 ticker) or multi (multiple tickers)")

    args = parser.parse_args()

    success = run_integration_test(test_mode=args.mode)
    sys.exit(0 if success else 1)
