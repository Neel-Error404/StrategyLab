"""
Stage 6 - Step 1: Run Dual Backtests

Runs MSE strategy twice on test period (2024-07-01 to 2025-08-31):
1. With 80% exit threshold (baseline)
2. With 95% exit threshold (optimal)

Each backtest outputs to separate directory for comparison.

Author: Strategy Optimization Pipeline
Date: 2025-10-05
"""

import sys
import subprocess
from pathlib import Path
import shutil
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent


def modify_exit_threshold(threshold: float) -> None:
    """
    Temporarily modify exit threshold in MSE strategy.

    Args:
        threshold: Exit threshold value (0.80 or 0.95)
    """

    strategy_file = PROJECT_ROOT / 'src' / 'strategies' / 'mse_strategy_backtesting.py'

    print(f"\n🔧 Modifying exit threshold to {threshold*100:.0f}%...")

    # Read current file
    with open(strategy_file, 'r') as f:
        content = f.read()

    # Replace threshold line
    # Line 80: self.exit_threshold = 0.80  # Exit at 80% of peak (let winners run)
    import re
    pattern = r'self\.exit_threshold = [0-9.]+.*'
    replacement = f'self.exit_threshold = {threshold}  # Modified for Stage 6 testing'

    new_content = re.sub(pattern, replacement, content)

    # Write modified file
    with open(strategy_file, 'w') as f:
        f.write(new_content)

    print(f"   ✓ Set exit_threshold = {threshold}")


def run_backtest(threshold: float, output_suffix: str, tickers: list) -> str:
    """
    Run MSE backtest with specified threshold.

    Args:
        threshold: Exit threshold (0.80 or 0.95)
        output_suffix: Suffix for output directory (e.g., '80pct', '95pct')
        tickers: List of tickers to backtest

    Returns:
        Path to output directory
    """

    print(f"\n{'='*70}")
    print(f"RUNNING BACKTEST: {threshold*100:.0f}% EXIT THRESHOLD")
    print(f"{'='*70}")

    # Modify strategy code
    modify_exit_threshold(threshold)

    # Prepare command
    ticker_str = ','.join(tickers)

    cmd = [
        'python',
        'src/runners/unified_runner.py',
        '--mode', 'backtest',
        '--template', 'conservative',
        '--dates', '2024-07-01',
        '--end-date', '2025-08-31',
        '--tickers', ticker_str
    ]

    print(f"\n🚀 Starting backtest...")
    print(f"   Command: {' '.join(cmd)}")
    print(f"   Tickers: {len(tickers)} stocks")
    print(f"   Period: 2024-07-01 to 2025-08-31")
    print(f"   Exit threshold: {threshold*100:.0f}%")

    # Run backtest
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"\n❌ Backtest failed!")
        print(f"Error: {result.stderr}")
        raise RuntimeError(f"Backtest failed for {threshold*100:.0f}%")

    print(f"\n✅ Backtest completed successfully")

    # Find output directory (most recent)
    outputs_dir = PROJECT_ROOT / 'outputs'
    output_dirs = sorted(outputs_dir.glob('*/mse_backtesting/*'), key=lambda x: x.stat().st_mtime)

    if not output_dirs:
        raise RuntimeError("No output directory found")

    latest_output = output_dirs[-1]

    # Rename to include threshold suffix
    new_name = f"{latest_output.parent.name}_test_{output_suffix}"
    new_output = latest_output.parent.parent / new_name / latest_output.name

    # Move directory
    shutil.move(str(latest_output.parent), str(new_output.parent))

    print(f"   ✓ Output saved to: {new_output}")

    return str(new_output)


def main():
    """Execute dual backtests."""

    print("\n" + "="*70)
    print("STAGE 6 - STEP 1: RUN DUAL BACKTESTS")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n📋 Test Plan:")
    print("   1. Run MSE backtest with 80% threshold (baseline)")
    print("   2. Run MSE backtest with 95% threshold (optimal)")
    print("   3. Both on test period: 2024-07-01 to 2025-08-31")
    print("   4. Same 24 tickers as optimization")

    # Tickers from optimization
    tickers = [
        "RELIANCE", "TCS", "INFY", "HINDUNILVR", "ITC",
        "SBIN", "KOTAKBANK", "LT", "ASIANPAINT", "AXISBANK",
        "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO",
        "NESTLEIND", "HCLTECH", "POWERGRID", "NTPC", "ONGC",
        "TATASTEEL", "JSWSTEEL", "ADANIPORTS", "TECHM"
    ]

    print(f"\n   Tickers: {len(tickers)} stocks")

    # Backup original strategy file
    strategy_file = PROJECT_ROOT / 'src' / 'strategies' / 'mse_strategy_backtesting.py'
    backup_file = strategy_file.with_suffix('.py.backup_stage6')

    print(f"\n💾 Backing up original strategy file...")
    shutil.copy(strategy_file, backup_file)
    print(f"   ✓ Backup: {backup_file}")

    try:
        # Run 80% backtest
        output_80 = run_backtest(
            threshold=0.80,
            output_suffix='80pct',
            tickers=tickers
        )

        # Run 95% backtest
        output_95 = run_backtest(
            threshold=0.95,
            output_suffix='95pct',
            tickers=tickers
        )

        # Save output paths
        paths_file = PROJECT_ROOT / 'analysis' / 'strategy_optimization' / 'checkpoints' / 'stage6_backtest_paths.txt'
        with open(paths_file, 'w') as f:
            f.write(f"baseline_80pct={output_80}\n")
            f.write(f"optimal_95pct={output_95}\n")

        print(f"\n{'='*70}")
        print("DUAL BACKTESTS COMPLETE")
        print(f"{'='*70}")
        print(f"\n✅ Both backtests completed successfully")
        print(f"\n📁 Output Directories:")
        print(f"   Baseline (80%): {output_80}")
        print(f"   Optimal (95%):  {output_95}")
        print(f"\n📝 Paths saved to: {paths_file}")
        print(f"\n➡️  Next: Run stage6_02_merge_trades.py to consolidate ticker files")

    finally:
        # Restore original strategy file
        print(f"\n🔄 Restoring original strategy file...")
        shutil.copy(backup_file, strategy_file)
        backup_file.unlink()
        print(f"   ✓ Original file restored")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
