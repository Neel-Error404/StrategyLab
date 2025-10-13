"""
Stage 6: Final Out-of-Sample Test Comparison

Compares baseline (0.80) vs optimal (0.95) on unseen test data (2024 H2 - 2025).

Usage:
    python scripts/06_final_test_comparison.py \
        --baseline "outputs/20251005_121223/mse_backtesting/2022-01-01_to_2025-08-31/all_trade_merged.csv" \
        --optimal "outputs/20251005_124708/mse_backtesting/2022-01-01_to_2025-08-31/all_trade_merged.csv"

Author: Strategy Optimization Pipeline
Date: 2025-10-05
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from datetime import datetime

# Add modules to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / 'modules'))

from metrics_calculator import calculate_traditional_metrics, print_metrics_summary


def load_and_filter_trades(file_path: str, start_date: str, end_date: str, valid_tickers: list) -> pd.DataFrame:
    """Load trades and filter to test period and valid tickers."""

    print(f"\n📂 Loading: {Path(file_path).parent.parent.name}")

    # Check file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Trade file not found: {file_path}")

    # Load trades
    df = pd.read_csv(file_path)
    print(f"   Total trades (all tickers): {len(df):,}")

    # Check required columns
    required_cols = ['Entry Time', 'Exit Time', 'ticker', 'percentage_return']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert dates
    df['Entry Time'] = pd.to_datetime(df['Entry Time'])
    df['Exit Time'] = pd.to_datetime(df['Exit Time'])

    # Filter to valid tickers FIRST
    df_filtered = df[df['ticker'].isin(valid_tickers)].copy()
    print(f"   After ticker filter (24 tickers): {len(df_filtered):,} trades")

    if len(df_filtered) == 0:
        raise ValueError(f"No trades found for valid tickers: {valid_tickers}")

    # Show ticker distribution
    ticker_counts = df_filtered['ticker'].value_counts()
    print(f"   Tickers present: {len(ticker_counts)} / {len(valid_tickers)}")

    # Filter to test period
    test_df = df_filtered[
        (df_filtered['Entry Time'] >= start_date) &
        (df_filtered['Entry Time'] <= end_date)
    ].copy()

    print(f"   Test period ({start_date} to {end_date}): {len(test_df):,} trades")

    if len(test_df) == 0:
        raise ValueError(f"No trades in test period {start_date} to {end_date}")

    return test_df


def compare_performance(baseline_df: pd.DataFrame, optimal_df: pd.DataFrame) -> dict:
    """Calculate and compare metrics."""

    print("\n" + "="*70)
    print("CALCULATING METRICS")
    print("="*70)

    # Calculate metrics
    baseline_metrics = calculate_traditional_metrics(baseline_df)
    optimal_metrics = calculate_traditional_metrics(optimal_df)

    # Print summaries
    print_metrics_summary(baseline_metrics, "BASELINE (0.80 THRESHOLD)")
    print_metrics_summary(optimal_metrics, "OPTIMAL (0.95 THRESHOLD)")

    # Calculate improvements
    improvements = {
        'win_rate': optimal_metrics['win_rate_pct'] - baseline_metrics['win_rate_pct'],
        'profit_factor': optimal_metrics['profit_factor'] - baseline_metrics['profit_factor'],
        'sharpe_ratio': optimal_metrics['sharpe_ratio'] - baseline_metrics['sharpe_ratio'],
        'total_return': optimal_metrics['total_return_pct'] - baseline_metrics['total_return_pct'],
        'max_drawdown': optimal_metrics['max_drawdown_pct'] - baseline_metrics['max_drawdown_pct'],
        'avg_win': optimal_metrics['avg_win_pct'] - baseline_metrics['avg_win_pct'],
        'avg_loss': optimal_metrics['avg_loss_pct'] - baseline_metrics['avg_loss_pct'],
    }

    return {
        'baseline': baseline_metrics,
        'optimal': optimal_metrics,
        'improvements': improvements
    }


def evaluate_criteria(results: dict) -> dict:
    """Evaluate success criteria."""

    baseline = results['baseline']
    optimal = results['optimal']
    improvements = results['improvements']

    print("\n" + "="*70)
    print("SUCCESS CRITERIA EVALUATION")
    print("="*70)

    # Criterion 1: Relative Performance
    print(f"\n📊 CRITERION 1: Relative Performance")
    print(f"   95% must outperform 80% on test data")

    better_wr = improvements['win_rate'] > 0
    better_pf = improvements['profit_factor'] > 0

    print(f"\n   Win Rate:")
    print(f"      Baseline: {baseline['win_rate_pct']:.2f}%")
    print(f"      Optimal:  {optimal['win_rate_pct']:.2f}%")
    print(f"      Improvement: {improvements['win_rate']:+.2f}% {'✅' if better_wr else '❌'}")

    print(f"\n   Profit Factor:")
    print(f"      Baseline: {baseline['profit_factor']:.2f}")
    print(f"      Optimal:  {optimal['profit_factor']:.2f}")
    print(f"      Improvement: {improvements['profit_factor']:+.2f} {'✅' if better_pf else '❌'}")

    criterion_1 = better_wr and better_pf

    if criterion_1:
        print(f"\n   ✅ CRITERION 1: PASS")
    else:
        print(f"\n   ❌ CRITERION 1: FAIL")
        if not better_wr:
            print(f"      Reason: Win Rate degraded")
        if not better_pf:
            print(f"      Reason: Profit Factor degraded")

    # Criterion 2: Absolute Performance
    print(f"\n📊 CRITERION 2: Absolute Performance")
    print(f"   95% must meet ALL targets")

    meets_wr = optimal['win_rate_pct'] >= 52.0
    meets_pf = optimal['profit_factor'] >= 1.25
    meets_sharpe = optimal['sharpe_ratio'] >= 1.5

    print(f"\n   Win Rate: {optimal['win_rate_pct']:.2f}% (target: ≥52%) {'✅' if meets_wr else '❌'}")
    print(f"   Profit Factor: {optimal['profit_factor']:.2f} (target: ≥1.25) {'✅' if meets_pf else '❌'}")
    print(f"   Sharpe Ratio: {optimal['sharpe_ratio']:.2f} (target: ≥1.5) {'✅' if meets_sharpe else '❌'}")

    criterion_2 = meets_wr and meets_pf and meets_sharpe

    if criterion_2:
        print(f"\n   ✅ CRITERION 2: PASS")
    else:
        print(f"\n   ❌ CRITERION 2: FAIL")
        if not meets_wr:
            print(f"      Reason: Win Rate below 52%")
        if not meets_pf:
            print(f"      Reason: Profit Factor below 1.25")
        if not meets_sharpe:
            print(f"      Reason: Sharpe Ratio below 1.5")

    # Final Decision
    print("\n" + "="*70)
    print("FINAL DECISION")
    print("="*70)

    final_pass = criterion_1 and criterion_2

    if final_pass:
        print("\n✅ ✅ ✅ STAGE 6: PASS ✅ ✅ ✅")
        print("\nBoth criteria met on unseen test data!")
        print("\n🎉 OPTIMIZATION VALIDATED")
        print("\n➡️ RECOMMENDATION: IMPLEMENT 95% THRESHOLD IN PRODUCTION")

        print("\n📋 Implementation Plan:")
        print("   1. Update production config: exit_threshold = 0.95")
        print("   2. Paper trading: 1 month (0% capital)")
        print("   3. Live deployment: Gradual 10% → 100%")
        print("   4. Monitoring: Daily WR/PF, weekly Sharpe, monthly review")

    else:
        print("\n❌ ❌ ❌ STAGE 6: FAIL ❌ ❌ ❌")
        print("\nOptimization did NOT validate on test data")
        print("\n⛔ RECOMMENDATION: REJECT 95% THRESHOLD, KEEP 80% BASELINE")

        print("\n🔍 Root Cause Analysis:")
        if not criterion_1:
            print("   - Validation data (2024 H1) was unrepresentative")
            print("   - Optimization overfit to validation period")
        if not criterion_2:
            print("   - Performance degraded on unseen data")
            print("   - Walk-forward validation warnings were correct")

        print("\n📋 Options:")
        print("   1. Accept 80% baseline (safest)")
        print("   2. Investigate regime differences (validation vs test)")
        print("   3. Start fresh Phase 3 with different approach")

    return {
        'criterion_1': criterion_1,
        'criterion_2': criterion_2,
        'final_pass': final_pass,
        'better_wr': better_wr,
        'better_pf': better_pf,
        'meets_wr': meets_wr,
        'meets_pf': meets_pf,
        'meets_sharpe': meets_sharpe
    }


def save_results(results: dict, evaluation: dict, output_dir: Path):
    """Save results to files."""

    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    # Create summary DataFrame
    summary = pd.DataFrame([
        {
            'Metric': 'Win Rate (%)',
            'Baseline_0.80': results['baseline']['win_rate_pct'],
            'Optimal_0.95': results['optimal']['win_rate_pct'],
            'Improvement': results['improvements']['win_rate'],
            'Target': 52.0,
            'Meets_Target': evaluation['meets_wr']
        },
        {
            'Metric': 'Profit Factor',
            'Baseline_0.80': results['baseline']['profit_factor'],
            'Optimal_0.95': results['optimal']['profit_factor'],
            'Improvement': results['improvements']['profit_factor'],
            'Target': 1.25,
            'Meets_Target': evaluation['meets_pf']
        },
        {
            'Metric': 'Sharpe Ratio',
            'Baseline_0.80': results['baseline']['sharpe_ratio'],
            'Optimal_0.95': results['optimal']['sharpe_ratio'],
            'Improvement': results['improvements']['sharpe_ratio'],
            'Target': 1.5,
            'Meets_Target': evaluation['meets_sharpe']
        },
        {
            'Metric': 'Total Return (%)',
            'Baseline_0.80': results['baseline']['total_return_pct'],
            'Optimal_0.95': results['optimal']['total_return_pct'],
            'Improvement': results['improvements']['total_return'],
            'Target': None,
            'Meets_Target': None
        },
        {
            'Metric': 'Max Drawdown (%)',
            'Baseline_0.80': results['baseline']['max_drawdown_pct'],
            'Optimal_0.95': results['optimal']['max_drawdown_pct'],
            'Improvement': results['improvements']['max_drawdown'],
            'Target': -15.0,
            'Meets_Target': results['optimal']['max_drawdown_pct'] >= -15.0
        }
    ])

    summary_path = output_dir / 'stage6_final_test_results.csv'
    summary.to_csv(summary_path, index=False)
    print(f"\n✅ Summary: {summary_path}")

    # Save decision
    decision = pd.DataFrame([{
        'Criterion_1_Pass': evaluation['criterion_1'],
        'Criterion_2_Pass': evaluation['criterion_2'],
        'Final_Decision': 'PASS' if evaluation['final_pass'] else 'FAIL',
        'Recommendation': 'IMPLEMENT 0.95' if evaluation['final_pass'] else 'KEEP 0.80',
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }])

    decision_path = output_dir / 'stage6_final_decision.csv'
    decision.to_csv(decision_path, index=False)
    print(f"✅ Decision: {decision_path}")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='Stage 6: Final Out-of-Sample Test Comparison')
    parser.add_argument('--baseline', required=True, help='Path to baseline (0.80) all_trade_merged.csv')
    parser.add_argument('--optimal', required=True, help='Path to optimal (0.95) all_trade_merged.csv')
    parser.add_argument('--test-start', default='2024-07-01', help='Test period start date')
    parser.add_argument('--test-end', default='2025-08-31', help='Test period end date')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("STAGE 6: FINAL OUT-OF-SAMPLE TEST")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test Period: {args.test_start} to {args.test_end}")

    # Load valid tickers (24 tickers used in optimization)
    valid_tickers = [
        "RELIANCE", "TCS", "INFY", "HINDUNILVR", "ITC", "SBIN", "KOTAKBANK", "LT",
        "ASIANPAINT", "AXISBANK", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO",
        "WIPRO", "NESTLEIND", "HCLTECH", "POWERGRID", "NTPC", "ONGC",
        "TATASTEEL", "JSWSTEEL", "ADANIPORTS", "TECHM"
    ]

    print(f"\n📋 Valid Tickers: {len(valid_tickers)} tickers")
    print(f"   {', '.join(valid_tickers[:8])}...")

    # Validate input files exist
    print(f"\n🔍 Validating input files...")
    baseline_path = Path(args.baseline)
    optimal_path = Path(args.optimal)

    if not baseline_path.exists():
        raise FileNotFoundError(f"❌ Baseline file not found: {baseline_path}")
    print(f"   ✅ Baseline: {baseline_path.name}")

    if not optimal_path.exists():
        raise FileNotFoundError(f"❌ Optimal file not found: {optimal_path}")
    print(f"   ✅ Optimal: {optimal_path.name}")

    # Load and filter trades
    baseline_df = load_and_filter_trades(args.baseline, args.test_start, args.test_end, valid_tickers)
    optimal_df = load_and_filter_trades(args.optimal, args.test_start, args.test_end, valid_tickers)

    # Verify same number of trades
    if len(baseline_df) != len(optimal_df):
        print(f"\n⚠️ WARNING: Different trade counts!")
        print(f"   Baseline: {len(baseline_df):,} trades")
        print(f"   Optimal: {len(optimal_df):,} trades")
        print(f"\n   This may indicate different entry logic was used.")
        print(f"   Proceeding with comparison, but results may not be valid.")

    # Compare performance
    results = compare_performance(baseline_df, optimal_df)

    # Evaluate criteria
    evaluation = evaluate_criteria(results)

    # Save results
    output_dir = PROJECT_ROOT / 'checkpoints'
    save_results(results, evaluation, output_dir)

    # Print final summary
    print("\n" + "="*70)
    print("STAGE 6 COMPLETE")
    print("="*70)

    if evaluation['final_pass']:
        print("\n✅ RESULT: VALIDATION SUCCESSFUL")
        print("   95% threshold APPROVED for production deployment")
    else:
        print("\n❌ RESULT: VALIDATION FAILED")
        print("   Keep 80% baseline threshold")

    print("\n" + "="*70)

    # Return exit code for scripting
    sys.exit(0 if evaluation['final_pass'] else 1)


if __name__ == "__main__":
    main()
