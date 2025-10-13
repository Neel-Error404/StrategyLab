"""
Stage 6: Analyze Real Backtest Results

Loads actual backtest trades from both threshold runs, filters to test period,
and performs final validation comparison.

Usage:
------
python stage6_analyze_real_backtests.py \
    --baseline-dir outputs/TIMESTAMP1/mse_backtesting/2022-01-01_to_2025-08-31 \
    --optimal-dir outputs/TIMESTAMP2/mse_backtesting/2022-01-01_to_2025-08-31

Author: Strategy Optimization Pipeline
Date: 2025-10-05
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import yaml

# Add modules to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / 'modules'))

from metrics_calculator import calculate_traditional_metrics


def load_strategy_trades(backtest_dir: str, threshold_name: str) -> pd.DataFrame:
    """
    Load and merge all strategy trade files from a backtest directory.

    Args:
        backtest_dir: Path to backtest output (e.g., outputs/.../mse_backtesting/2022-01-01_to_2025-08-31)
        threshold_name: Name for logging (e.g., '80% Baseline', '95% Optimal')

    Returns:
        Combined DataFrame with all trades
    """

    print(f"\n{'='*70}")
    print(f"LOADING {threshold_name.upper()}")
    print(f"{'='*70}")

    backtest_path = Path(backtest_dir)

    # Find strategy_trades directory
    strategy_trades_dir = backtest_path / 'data' / 'strategy_trades'

    if not strategy_trades_dir.exists():
        raise FileNotFoundError(
            f"Strategy trades directory not found: {strategy_trades_dir}\n"
            f"Expected structure: {backtest_dir}/data/strategy_trades/"
        )

    # Find all strategy trade CSV files
    trade_files = list(strategy_trades_dir.glob('*_StrategyTrades_*.csv'))

    if not trade_files:
        raise FileNotFoundError(
            f"No strategy trade files found in {strategy_trades_dir}\n"
            f"Expected files like: TICKER_StrategyTrades_*.csv"
        )

    print(f"\n📂 Found {len(trade_files)} ticker trade files")

    # Load and combine all trades
    all_trades = []

    for trade_file in sorted(trade_files):
        ticker = trade_file.stem.split('_StrategyTrades_')[0]

        try:
            trades = pd.read_csv(trade_file)

            if len(trades) > 0:
                # Add ticker column if not present
                if 'ticker' not in trades.columns:
                    trades['ticker'] = ticker

                all_trades.append(trades)
                print(f"   ✓ {ticker:15s}: {len(trades):5d} trades")

        except Exception as e:
            print(f"   ⚠️  {ticker:15s}: Error - {e}")
            continue

    if not all_trades:
        raise ValueError("No valid trade files could be loaded")

    # Combine all trades
    combined_trades = pd.concat(all_trades, ignore_index=True)

    # Standardize column names (handle different naming conventions)
    column_mapping = {
        'entry_timestamp': 'Entry Time',
        'exit_timestamp': 'Exit Time',
        'entry_price': 'Entry Price',
        'exit_price': 'Exit Price',
        'trade_type': 'Trade Type',
        'pnl_pct': 'percentage_return',
        'return_pct': 'percentage_return'
    }

    for old_col, new_col in column_mapping.items():
        if old_col in combined_trades.columns and new_col not in combined_trades.columns:
            combined_trades[new_col] = combined_trades[old_col]

    # Ensure datetime columns
    for col in ['Entry Time', 'Exit Time', 'entry_timestamp', 'exit_timestamp']:
        if col in combined_trades.columns:
            combined_trades[col] = pd.to_datetime(combined_trades[col], errors='coerce')

    print(f"\n✅ Total trades merged: {len(combined_trades):,}")

    return combined_trades


def filter_to_test_period(trades_df: pd.DataFrame,
                          test_start: str = '2024-07-01',
                          test_end: str = '2025-08-31') -> pd.DataFrame:
    """
    Filter trades to test period.

    Args:
        trades_df: All trades
        test_start: Test period start date
        test_end: Test period end date

    Returns:
        Filtered DataFrame
    """

    print(f"\n🔍 Filtering to test period: {test_start} to {test_end}")

    # Ensure Entry Time is datetime
    entry_col = 'Entry Time' if 'Entry Time' in trades_df.columns else 'entry_timestamp'
    trades_df[entry_col] = pd.to_datetime(trades_df[entry_col])

    # Filter
    test_start_dt = pd.Timestamp(test_start)
    test_end_dt = pd.Timestamp(test_end)

    test_trades = trades_df[
        (trades_df[entry_col] >= test_start_dt) &
        (trades_df[entry_col] <= test_end_dt)
    ].copy()

    print(f"   Original: {len(trades_df):,} trades")
    print(f"   Test period: {len(test_trades):,} trades ({len(test_trades)/len(trades_df)*100:.1f}%)")

    # Summary by trade type
    if 'Trade Type' in test_trades.columns:
        buy_count = len(test_trades[test_trades['Trade Type'] == 'Buy'])
        sell_count = len(test_trades[test_trades['Trade Type'] == 'Sell'])
        print(f"   Buy: {buy_count:,} | Sell: {sell_count:,}")

    return test_trades


def calculate_performance(trades_df: pd.DataFrame, threshold_name: str) -> dict:
    """
    Calculate performance metrics.

    Args:
        trades_df: Trade data
        threshold_name: Name for display

    Returns:
        Dictionary with metrics
    """

    print(f"\n📊 Calculating {threshold_name} Performance...")

    # Ensure percentage_return column exists
    if 'percentage_return' not in trades_df.columns:
        if 'pnl_pct' in trades_df.columns:
            trades_df['percentage_return'] = trades_df['pnl_pct']
        elif 'return_pct' in trades_df.columns:
            trades_df['percentage_return'] = trades_df['return_pct']
        else:
            raise ValueError("No return percentage column found")

    # Calculate metrics
    metrics = calculate_traditional_metrics(trades_df)

    # Print summary
    print(f"\n   Overall Performance:")
    print(f"      Total Trades: {metrics['total_trades']:,}")
    print(f"      Win Rate: {metrics['win_rate_pct']:.2f}%")
    print(f"      Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"      Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"      Avg Win: {metrics['avg_win_pct']:.2f}%")
    print(f"      Avg Loss: {metrics['avg_loss_pct']:.2f}%")

    return metrics


def compare_and_decide(baseline_metrics: dict,
                       optimal_metrics: dict,
                       success_criteria: dict) -> dict:
    """
    Compare performance and make final decision.

    Args:
        baseline_metrics: Metrics from 80% threshold
        optimal_metrics: Metrics from 95% threshold
        success_criteria: Success criteria from config

    Returns:
        Decision dictionary
    """

    print(f"\n{'='*70}")
    print("FINAL COMPARISON & DECISION")
    print(f"{'='*70}")

    # Calculate improvements
    improvements = {
        'win_rate': optimal_metrics['win_rate_pct'] - baseline_metrics['win_rate_pct'],
        'profit_factor': optimal_metrics['profit_factor'] - baseline_metrics['profit_factor'],
        'sharpe_ratio': optimal_metrics['sharpe_ratio'] - baseline_metrics['sharpe_ratio']
    }

    print(f"\n📊 Performance Comparison (95% - 80%):")
    print(f"   Win Rate: {improvements['win_rate']:+.2f}% "
          f"({'✅' if improvements['win_rate'] > 0 else '❌'})")
    print(f"   Profit Factor: {improvements['profit_factor']:+.2f} "
          f"({'✅' if improvements['profit_factor'] > 0 else '❌'})")
    print(f"   Sharpe Ratio: {improvements['sharpe_ratio']:+.2f} "
          f"({'✅' if improvements['sharpe_ratio'] > 0 else '❌'})")

    # Check Condition 1: 95% beats 80%
    condition_1 = all(improvements[m] > 0 for m in ['win_rate', 'profit_factor', 'sharpe_ratio'])

    print(f"\n✓ Condition 1: 95% outperforms 80% on all metrics")
    print(f"  {'✅ PASS' if condition_1 else '❌ FAIL'}")

    # Check Condition 2: 95% meets success criteria
    criteria_check = {
        'win_rate': optimal_metrics['win_rate_pct'] >= success_criteria['traditional']['min_win_rate'] * 100,
        'profit_factor': optimal_metrics['profit_factor'] >= success_criteria['traditional']['min_profit_factor'],
        'sharpe_ratio': optimal_metrics['sharpe_ratio'] >= success_criteria['traditional']['min_sharpe_ratio']
    }

    print(f"\n✓ Condition 2: 95% meets success criteria")
    for criterion, passed in criteria_check.items():
        status = '✅' if passed else '❌'
        value = optimal_metrics.get(f'{criterion}_pct' if criterion == 'win_rate' else criterion, optimal_metrics[criterion])
        target = success_criteria['traditional'][f'min_{criterion}'] * (100 if criterion == 'win_rate' else 1)
        print(f"  {status} {criterion.replace('_', ' ').title()}: {value:.2f} (target: ≥{target:.2f})")

    condition_2 = all(criteria_check.values())
    print(f"  {'✅ PASS' if condition_2 else '❌ FAIL'}")

    # Final Decision
    print(f"\n{'='*70}")

    if condition_1 and condition_2:
        print("🎉 FINAL VALIDATION PASSED 🎉")
        print(f"{'='*70}")
        print("\n✅ DECISION: IMPLEMENT 95% THRESHOLD IN PRODUCTION")
        print("\n📋 Implementation Steps:")
        print("   1. Update mse_strategy_backtesting.py:")
        print("      Line 80: self.exit_threshold = 0.95")
        print("   2. Deploy to paper trading for 1 week")
        print("   3. Monitor Win Rate and Profit Factor")
        print("   4. If paper trading confirms, deploy to live")
        decision = "IMPLEMENT"

    else:
        print("⛔ FINAL VALIDATION FAILED ⛔")
        print(f"{'='*70}")
        print("\n❌ DECISION: REJECT - STAY WITH 80% BASELINE")

        if not condition_1:
            print("\n   Reason: 95% did NOT outperform 80% on test data")
            print("   → Suggests overfitting to validation period (2024 H1)")

        if not condition_2:
            print("\n   Reason: 95% did NOT meet success criteria")
            print("   → Performance targets not achieved on fresh data")

        print("\n   Options:")
        print("   1. Accept 80% baseline as optimal")
        print("   2. Re-run Stage 2 with different threshold range (e.g., 85-92%)")
        print("   3. Investigate why test performance differs from validation")
        decision = "REJECT"

    print(f"{'='*70}")

    return {
        'decision': decision,
        'condition_1_pass': condition_1,
        'condition_2_pass': condition_2,
        'improvements': improvements,
        'baseline_metrics': baseline_metrics,
        'optimal_metrics': optimal_metrics
    }


def main():
    """Execute Stage 6 analysis."""

    parser = argparse.ArgumentParser(description='Stage 6: Analyze real backtest results')
    parser.add_argument('--baseline-dir', required=True,
                        help='Path to 80%% baseline backtest output')
    parser.add_argument('--optimal-dir', required=True,
                        help='Path to 95%% optimal backtest output')
    parser.add_argument('--test-start', default='2024-07-01',
                        help='Test period start date (default: 2024-07-01)')
    parser.add_argument('--test-end', default='2025-08-31',
                        help='Test period end date (default: 2025-08-31)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("STAGE 6: FINAL OUT-OF-SAMPLE VALIDATION")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\n📂 Input Directories:")
    print(f"   Baseline (80%): {args.baseline_dir}")
    print(f"   Optimal (95%):  {args.optimal_dir}")

    # Load config for success criteria
    config_path = PROJECT_ROOT / 'config' / 'optimization_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load trades from both backtests
    baseline_trades_all = load_strategy_trades(args.baseline_dir, '80% Baseline')
    optimal_trades_all = load_strategy_trades(args.optimal_dir, '95% Optimal')

    # Filter to test period
    baseline_trades_test = filter_to_test_period(baseline_trades_all, args.test_start, args.test_end)
    optimal_trades_test = filter_to_test_period(optimal_trades_all, args.test_start, args.test_end)

    # Calculate performance
    baseline_metrics = calculate_performance(baseline_trades_test, '80% Baseline')
    optimal_metrics = calculate_performance(optimal_trades_test, '95% Optimal')

    # Compare and decide
    decision_result = compare_and_decide(
        baseline_metrics,
        optimal_metrics,
        config['success_criteria']
    )

    # Save results
    checkpoints_dir = PROJECT_ROOT / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)

    # Save decision
    decision_df = pd.DataFrame([{
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'decision': decision_result['decision'],
        'condition_1_outperforms': decision_result['condition_1_pass'],
        'condition_2_meets_criteria': decision_result['condition_2_pass'],
        'baseline_win_rate': baseline_metrics['win_rate_pct'],
        'optimal_win_rate': optimal_metrics['win_rate_pct'],
        'win_rate_improvement': decision_result['improvements']['win_rate'],
        'baseline_profit_factor': baseline_metrics['profit_factor'],
        'optimal_profit_factor': optimal_metrics['profit_factor'],
        'profit_factor_improvement': decision_result['improvements']['profit_factor'],
        'baseline_sharpe': baseline_metrics['sharpe_ratio'],
        'optimal_sharpe': optimal_metrics['sharpe_ratio'],
        'sharpe_improvement': decision_result['improvements']['sharpe_ratio'],
        'test_period_start': args.test_start,
        'test_period_end': args.test_end,
        'baseline_trades': baseline_metrics['total_trades'],
        'optimal_trades': optimal_metrics['total_trades']
    }])

    decision_path = checkpoints_dir / 'stage6_final_decision.csv'
    decision_df.to_csv(decision_path, index=False)

    print(f"\n💾 Results saved to: {decision_path}")

    print("\n" + "="*70)
    print("STAGE 6 COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
