"""
Stage 6: Final Out-of-Sample Test

THE CRITICAL VALIDATION - Tests optimized threshold on completely unseen data.

Purpose:
--------
- Test 95% threshold on 2024 H2 data (NEVER touched during optimization)
- Compare against 80% baseline on same data
- Verify meets ALL success criteria
- Make final GO/NO-GO decision for production

Success Criteria:
-----------------
BOTH must be true:
1. 95% outperforms 80% on test data
2. 95% meets all targets: WR ≥52%, PF ≥1.25, Sharpe ≥1.5

If EITHER fails → REJECT optimization, stay with 80% baseline

Author: Strategy Optimization Pipeline
Date: 2025-10-05
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
from typing import Dict, Tuple

# Add modules to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / 'modules'))

from metrics_calculator import calculate_traditional_metrics
from exit_simulator import simulate_all_thresholds


def load_test_trades() -> pd.DataFrame:
    """
    Load test trades from 2024 H2 (July-August 2025).

    This data has NEVER been used for:
    - Baseline calculation
    - Threshold optimization
    - Walk-forward validation
    - Statistical testing

    It is completely unseen and represents true future performance.
    """

    print("\n📂 Loading test trades (2024 H2 - UNSEEN DATA)...")

    # Test trades should be in integration/validation output
    test_trades_path = PROJECT_ROOT.parent / 'integration' / 'validation' / 'all_trades_with_context_test.csv'

    if not test_trades_path.exists():
        # Alternative path
        test_trades_path = PROJECT_ROOT / 'data' / 'test_trades.csv'

    if not test_trades_path.exists():
        raise FileNotFoundError(
            f"Test trades not found.\n"
            f"Expected: {test_trades_path}\n"
            f"Make sure test data (2024-07-01 to 2025-08-31) exists."
        )

    test_trades = pd.read_csv(test_trades_path)

    # Ensure datetime columns
    for col in ['Entry Time', 'Exit Time']:
        if col in test_trades.columns:
            test_trades[col] = pd.to_datetime(test_trades[col]).dt.tz_localize(None)

    # Filter to test period
    test_start = pd.Timestamp('2024-07-01')
    test_end = pd.Timestamp('2025-08-31')

    test_trades = test_trades[
        (test_trades['Entry Time'] >= test_start) &
        (test_trades['Entry Time'] <= test_end)
    ].copy()

    print(f"   ✓ Loaded {len(test_trades)} test trades")
    print(f"   Period: {test_trades['Entry Time'].min()} to {test_trades['Entry Time'].max()}")
    print(f"   Buy trades: {len(test_trades[test_trades['Trade Type'] == 'Buy'])}")
    print(f"   Sell trades: {len(test_trades[test_trades['Trade Type'] == 'Sell'])}")

    return test_trades


def test_threshold_performance(
    test_trades: pd.DataFrame,
    base_data_dir: str,
    threshold: float,
    threshold_name: str
) -> Dict:
    """
    Test a threshold on out-of-sample data.

    Args:
        test_trades: Test period trades
        base_data_dir: Directory with base data
        threshold: Threshold to test (0.80 or 0.95)
        threshold_name: Display name

    Returns:
        Performance metrics dictionary
    """

    print(f"\n{'='*70}")
    print(f"TESTING {threshold_name.upper()} ({threshold*100:.0f}% THRESHOLD)")
    print(f"{'='*70}")

    # Simulate this threshold
    print(f"\n🔄 Simulating {threshold*100:.0f}% threshold on {len(test_trades)} test trades...")

    results = simulate_all_thresholds(
        test_trades.copy(),
        base_data_dir,
        thresholds=[threshold],
        trade_type_filter=None,
        progress_callback=None
    )

    # Filter out errors
    if 'error' in results.columns:
        valid_results = results[~results['error'].notna()].copy()
        if len(valid_results) == 0:
            raise ValueError(f"All simulations failed for {threshold_name}")
        results = valid_results

    print(f"   ✓ Simulated {len(results)} trades successfully")

    # Calculate metrics by trade type
    buy_results = results[results['trade_type'] == 'Buy'].copy()
    sell_results = results[results['trade_type'] == 'Sell'].copy()

    # Prepare for metrics calculator
    for df in [buy_results, sell_results, results]:
        df['percentage_return'] = df['sim_return_pct']
        df['Entry Time'] = pd.to_datetime(df['sim_exit_time']) - pd.to_timedelta(df['sim_duration_minutes'], unit='m')
        df['Exit Time'] = pd.to_datetime(df['sim_exit_time'])

    # Calculate overall metrics
    overall_metrics = calculate_traditional_metrics(results)
    buy_metrics = calculate_traditional_metrics(buy_results)
    sell_metrics = calculate_traditional_metrics(sell_results)

    print(f"\n📊 {threshold_name} Performance on Test Data:")
    print(f"\n   Overall:")
    print(f"      Win Rate: {overall_metrics['win_rate_pct']:.2f}%")
    print(f"      Profit Factor: {overall_metrics['profit_factor']:.2f}")
    print(f"      Sharpe Ratio: {overall_metrics['sharpe_ratio']:.2f}")
    print(f"      Total Trades: {overall_metrics['total_trades']}")

    print(f"\n   Buy Trades:")
    print(f"      Win Rate: {buy_metrics['win_rate_pct']:.2f}%")
    print(f"      Profit Factor: {buy_metrics['profit_factor']:.2f}")
    print(f"      Sharpe Ratio: {buy_metrics['sharpe_ratio']:.2f}")

    print(f"\n   Sell Trades:")
    print(f"      Win Rate: {sell_metrics['win_rate_pct']:.2f}%")
    print(f"      Profit Factor: {sell_metrics['profit_factor']:.2f}")
    print(f"      Sharpe Ratio: {sell_metrics['sharpe_ratio']:.2f}")

    return {
        'threshold': threshold,
        'threshold_name': threshold_name,
        'overall': overall_metrics,
        'buy': buy_metrics,
        'sell': sell_metrics,
        'results_df': results
    }


def evaluate_success_criteria(metrics: Dict, criteria: Dict) -> Dict:
    """
    Evaluate if metrics meet success criteria.

    Args:
        metrics: Performance metrics
        criteria: Success criteria from config

    Returns:
        Dictionary with pass/fail for each criterion
    """

    evaluation = {
        'win_rate': {
            'value': metrics['win_rate_pct'],
            'target': criteria['traditional']['min_win_rate'] * 100,
            'pass': metrics['win_rate_pct'] >= criteria['traditional']['min_win_rate'] * 100
        },
        'profit_factor': {
            'value': metrics['profit_factor'],
            'target': criteria['traditional']['min_profit_factor'],
            'pass': metrics['profit_factor'] >= criteria['traditional']['min_profit_factor']
        },
        'sharpe_ratio': {
            'value': metrics['sharpe_ratio'],
            'target': criteria['traditional']['min_sharpe_ratio'],
            'pass': metrics['sharpe_ratio'] >= criteria['traditional']['min_sharpe_ratio']
        }
    }

    evaluation['all_pass'] = all(e['pass'] for e in evaluation.values())

    return evaluation


def main():
    """Execute Stage 6: Final Out-of-Sample Test."""

    print("\n" + "="*70)
    print("STAGE 6: FINAL OUT-OF-SAMPLE TEST")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n" + "🎯"*35)
    print("THIS IS THE MOMENT OF TRUTH")
    print("Testing on completely unseen 2024 H2 data")
    print("If this passes, we implement in production")
    print("If this fails, we stay with 80% baseline")
    print("🎯"*35)

    # Load configuration
    config_path = PROJECT_ROOT / 'config' / 'optimization_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    base_data_dir = PROJECT_ROOT / config['data']['base_data_dir']
    success_criteria = config['success_criteria']

    # Output paths
    checkpoints_dir = PROJECT_ROOT / 'checkpoints'
    docs_dir = PROJECT_ROOT / 'docs'

    # Load test trades
    test_trades = load_test_trades()

    # Test baseline (80%)
    baseline_performance = test_threshold_performance(
        test_trades,
        str(base_data_dir),
        threshold=0.80,
        threshold_name="Baseline"
    )

    # Test optimal (95%)
    optimal_performance = test_threshold_performance(
        test_trades,
        str(base_data_dir),
        threshold=0.95,
        threshold_name="Optimal"
    )

    # Compare performance
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON (Test Data)")
    print("="*70)

    comparison = {
        'win_rate': optimal_performance['overall']['win_rate_pct'] - baseline_performance['overall']['win_rate_pct'],
        'profit_factor': optimal_performance['overall']['profit_factor'] - baseline_performance['overall']['profit_factor'],
        'sharpe_ratio': optimal_performance['overall']['sharpe_ratio'] - baseline_performance['overall']['sharpe_ratio']
    }

    print(f"\n📊 Overall Comparison (95% - 80%):")
    print(f"   Win Rate: {comparison['win_rate']:+.2f}% {'✅' if comparison['win_rate'] > 0 else '❌'}")
    print(f"   Profit Factor: {comparison['profit_factor']:+.2f} {'✅' if comparison['profit_factor'] > 0 else '❌'}")
    print(f"   Sharpe Ratio: {comparison['sharpe_ratio']:+.2f} {'✅' if comparison['sharpe_ratio'] > 0 else '❌'}")

    # Evaluate success criteria for optimal threshold
    print("\n" + "="*70)
    print("SUCCESS CRITERIA EVALUATION (95% Threshold)")
    print("="*70)

    evaluation = evaluate_success_criteria(
        optimal_performance['overall'],
        success_criteria
    )

    print(f"\n📋 Criteria Check:")
    for criterion, details in evaluation.items():
        if criterion == 'all_pass':
            continue
        status = '✅' if details['pass'] else '❌'
        print(f"   {status} {criterion.replace('_', ' ').title()}: {details['value']:.2f} (target: ≥{details['target']:.2f})")

    # Final decision
    print("\n" + "="*70)
    print("FINAL DECISION")
    print("="*70)

    condition_1 = all(comparison[m] > 0 for m in ['win_rate', 'profit_factor', 'sharpe_ratio'])
    condition_2 = evaluation['all_pass']

    print(f"\n✓ Condition 1: 95% outperforms 80% on all metrics")
    print(f"  {'✅ PASS' if condition_1 else '❌ FAIL'}")

    print(f"\n✓ Condition 2: 95% meets all success criteria")
    print(f"  {'✅ PASS' if condition_2 else '❌ FAIL'}")

    if condition_1 and condition_2:
        print("\n" + "🎉"*35)
        print("✅ FINAL VALIDATION PASSED")
        print("🎉"*35)
        print("\n🚀 RECOMMENDATION: IMPLEMENT 95% THRESHOLD IN PRODUCTION")
        print("\nNext Steps:")
        print("1. Update MSE strategy exit logic to use 95% threshold")
        print("2. Deploy to paper trading for 1 week")
        print("3. Monitor Win Rate and Profit Factor closely")
        print("4. If paper trading confirms, deploy to live")
        decision = "IMPLEMENT"
    else:
        print("\n" + "⛔"*35)
        print("❌ FINAL VALIDATION FAILED")
        print("⛔"*35)
        print("\n🛑 RECOMMENDATION: STAY WITH 80% BASELINE")

        if not condition_1:
            print("\nReason: 95% did NOT outperform 80% on test data")
            print("This suggests overfitting to validation period")

        if not condition_2:
            print("\nReason: 95% did NOT meet success criteria")
            print("Performance targets not achieved on fresh data")

        print("\nOptions:")
        print("1. Accept 80% baseline as optimal")
        print("2. Re-run Stage 2 with different threshold range")
        print("3. Investigate why test performance differs from validation")
        decision = "REJECT"

    # Save results
    print(f"\n💾 Saving results...")

    results_df = pd.DataFrame([
        {
            'threshold': 0.80,
            'threshold_name': 'Baseline',
            'win_rate': baseline_performance['overall']['win_rate_pct'],
            'profit_factor': baseline_performance['overall']['profit_factor'],
            'sharpe_ratio': baseline_performance['overall']['sharpe_ratio'],
            'total_trades': baseline_performance['overall']['total_trades']
        },
        {
            'threshold': 0.95,
            'threshold_name': 'Optimal',
            'win_rate': optimal_performance['overall']['win_rate_pct'],
            'profit_factor': optimal_performance['overall']['profit_factor'],
            'sharpe_ratio': optimal_performance['overall']['sharpe_ratio'],
            'total_trades': optimal_performance['overall']['total_trades']
        }
    ])

    results_path = checkpoints_dir / 'stage6_test_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"   ✓ Test results: {results_path}")

    # Save decision
    decision_df = pd.DataFrame([{
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'decision': decision,
        'condition_1_pass': condition_1,
        'condition_2_pass': condition_2,
        'win_rate_improvement': comparison['win_rate'],
        'profit_factor_improvement': comparison['profit_factor'],
        'sharpe_ratio_improvement': comparison['sharpe_ratio']
    }])

    decision_path = checkpoints_dir / 'stage6_final_decision.csv'
    decision_df.to_csv(decision_path, index=False)
    print(f"   ✓ Final decision: {decision_path}")

    print("\n" + "="*70)
    print("STAGE 6 COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
