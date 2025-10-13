"""
Walk-Forward Validation Module
===============================

Purpose:
--------
Test threshold stability across different time windows to prevent overfitting.

Methodology:
------------
1. Split validation period into rolling windows (e.g., monthly)
2. For each window, test all thresholds
3. Identify optimal threshold for each window
4. Check if optimal threshold is consistent across windows
5. Calculate stability metrics (coefficient of variation, consistency rate)

This ensures the optimal threshold from Stage 2 isn't just lucky on one period.

Author: Strategy Optimization Pipeline
Date: 2025-10-04
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import timedelta


def create_rolling_windows(trades_df: pd.DataFrame,
                          window_size_days: int = 30,
                          step_size_days: int = 15) -> List[Dict]:
    """
    Create rolling time windows from trade data.

    Parameters:
    -----------
    trades_df : pd.DataFrame
        Trade data with Entry Time column
    window_size_days : int
        Size of each window in days (default: 30 days = 1 month)
    step_size_days : int
        Step size between windows (default: 15 days = 50% overlap)

    Returns:
    --------
    list of dict
        Each dict contains: window_id, start_date, end_date, trades_df
    """

    # Ensure Entry Time is datetime
    trades_df = trades_df.copy()
    trades_df['Entry Time'] = pd.to_datetime(trades_df['Entry Time'])

    # Get date range
    min_date = trades_df['Entry Time'].min()
    max_date = trades_df['Entry Time'].max()

    windows = []
    window_id = 1
    current_start = min_date

    while current_start < max_date:
        current_end = current_start + timedelta(days=window_size_days)

        # Filter trades in this window
        window_trades = trades_df[
            (trades_df['Entry Time'] >= current_start) &
            (trades_df['Entry Time'] < current_end)
        ].copy()

        # Only include windows with sufficient trades
        if len(window_trades) >= 50:  # Minimum 50 trades per window
            windows.append({
                'window_id': window_id,
                'start_date': current_start,
                'end_date': current_end,
                'num_trades': len(window_trades),
                'trades_df': window_trades
            })
            window_id += 1

        # Move to next window
        current_start += timedelta(days=step_size_days)

    return windows


def validate_threshold_on_window(window_trades: pd.DataFrame,
                                 threshold: float,
                                 base_data_dir: str,
                                 trade_type: str) -> Dict:
    """
    Test a threshold on a specific time window.

    Parameters:
    -----------
    window_trades : pd.DataFrame
        Trades in this window
    threshold : float
        Threshold to test
    base_data_dir : str
        Path to base_data
    trade_type : str
        'Buy' or 'Sell'

    Returns:
    --------
    dict
        Performance metrics for this threshold on this window
    """

    from exit_simulator import simulate_all_thresholds, calculate_threshold_metrics

    # Filter to trade type
    type_trades = window_trades[window_trades['Trade Type'] == trade_type]

    if len(type_trades) == 0:
        return {'error': 'No trades for this type'}

    # Simulate this threshold
    results = simulate_all_thresholds(
        type_trades,
        base_data_dir,
        [threshold],  # Only test this one threshold
        trade_type_filter=trade_type,
        progress_callback=None  # No progress for single threshold
    )

    # Calculate metrics
    metrics = calculate_threshold_metrics(results, threshold)

    return metrics


def walk_forward_validation(trades_df: pd.DataFrame,
                           thresholds: List[float],
                           base_data_dir: str,
                           window_size_days: int = 30,
                           step_size_days: int = 15,
                           trade_type: str = 'Buy',
                           progress_callback=None) -> pd.DataFrame:
    """
    Perform walk-forward validation across time windows.

    Parameters:
    -----------
    trades_df : pd.DataFrame
        All trade data
    thresholds : list of float
        Thresholds to test
    base_data_dir : str
        Path to base_data
    window_size_days : int
        Window size in days
    step_size_days : int
        Step between windows
    trade_type : str
        'Buy' or 'Sell'
    progress_callback : callable
        Function(window_id, total_windows) for progress

    Returns:
    --------
    pd.DataFrame
        Results for each window and threshold combination
    """

    print(f"\n🔄 Walk-Forward Validation: {trade_type} Trades")
    print(f"   Window Size: {window_size_days} days")
    print(f"   Step Size: {step_size_days} days")

    # Create windows
    windows = create_rolling_windows(trades_df, window_size_days, step_size_days)
    print(f"   Created {len(windows)} windows")

    all_results = []

    for window in windows:
        if progress_callback:
            progress_callback(window['window_id'], len(windows))

        print(f"\n   Window {window['window_id']}/{len(windows)}: {window['start_date'].date()} to {window['end_date'].date()} ({window['num_trades']} trades)")

        for threshold in thresholds:
            metrics = validate_threshold_on_window(
                window['trades_df'],
                threshold,
                base_data_dir,
                trade_type
            )

            # Add window info
            metrics['window_id'] = window['window_id']
            metrics['window_start'] = window['start_date']
            metrics['window_end'] = window['end_date']
            metrics['window_num_trades'] = window['num_trades']
            metrics['trade_type'] = trade_type

            all_results.append(metrics)

        # Show best threshold for this window
        window_results = [r for r in all_results if r['window_id'] == window['window_id']]
        window_df = pd.DataFrame(window_results)

        if 'error' not in window_df.columns or not window_df['error'].notna().any():
            best_idx = window_df['win_rate_pct'].idxmax()
            best = window_df.loc[best_idx]
            print(f"      Best: {best['threshold']:.0%} | WR: {best['win_rate_pct']:.2f}% | PF: {best['profit_factor']:.2f}")

    results_df = pd.DataFrame(all_results)
    return results_df


def calculate_stability_metrics(walk_forward_results: pd.DataFrame,
                                target_threshold: float) -> Dict:
    """
    Calculate stability metrics for a specific threshold across windows.

    Parameters:
    -----------
    walk_forward_results : pd.DataFrame
        Results from walk_forward_validation
    target_threshold : float
        Threshold to analyze (e.g., 0.95)

    Returns:
    --------
    dict
        Stability metrics
    """

    # Filter to target threshold
    threshold_data = walk_forward_results[
        walk_forward_results['threshold'] == target_threshold
    ].copy()

    if len(threshold_data) == 0:
        return {'error': 'No data for threshold'}

    # Calculate metrics across windows
    win_rates = threshold_data['win_rate_pct'].values
    profit_factors = threshold_data['profit_factor'].values
    sharpe_ratios = threshold_data['sharpe_ratio'].values

    # Coefficient of Variation (lower is better, <10% is stable)
    cv_win_rate = (win_rates.std() / win_rates.mean()) * 100 if win_rates.mean() > 0 else 0
    cv_profit_factor = (profit_factors.std() / profit_factors.mean()) * 100 if profit_factors.mean() > 0 else 0
    cv_sharpe = (sharpe_ratios.std() / sharpe_ratios.mean()) * 100 if sharpe_ratios.mean() > 0 else 0

    # Consistency: % of windows where this threshold meets success criteria
    meets_wr_target = (win_rates >= 52).sum()
    meets_pf_target = (profit_factors >= 1.25).sum()
    meets_sharpe_target = (sharpe_ratios >= 1.5).sum()

    total_windows = len(threshold_data)

    consistency_rate = (
        (meets_wr_target + meets_pf_target + meets_sharpe_target) /
        (total_windows * 3)
    ) * 100

    # Check if this threshold was optimal (highest WR) in each window
    optimal_count = 0
    for window_id in threshold_data['window_id'].unique():
        window_results = walk_forward_results[
            walk_forward_results['window_id'] == window_id
        ]

        if 'error' in window_results.columns:
            window_results = window_results[~window_results['error'].notna()]

        if len(window_results) > 0:
            best_threshold = window_results.loc[window_results['win_rate_pct'].idxmax(), 'threshold']
            if best_threshold == target_threshold:
                optimal_count += 1

    optimal_rate = (optimal_count / total_windows) * 100 if total_windows > 0 else 0

    stability = {
        'threshold': target_threshold,
        'total_windows': total_windows,

        # Average performance across windows
        'avg_win_rate': win_rates.mean(),
        'avg_profit_factor': profit_factors.mean(),
        'avg_sharpe_ratio': sharpe_ratios.mean(),

        # Stability (Coefficient of Variation - lower is better)
        'cv_win_rate_pct': cv_win_rate,
        'cv_profit_factor_pct': cv_profit_factor,
        'cv_sharpe_ratio_pct': cv_sharpe,

        # Consistency (% of windows meeting targets)
        'windows_meeting_wr_target': meets_wr_target,
        'windows_meeting_pf_target': meets_pf_target,
        'windows_meeting_sharpe_target': meets_sharpe_target,
        'consistency_rate_pct': consistency_rate,

        # Optimality (% of windows where this was best)
        'windows_where_optimal': optimal_count,
        'optimal_rate_pct': optimal_rate,

        # Range
        'min_win_rate': win_rates.min(),
        'max_win_rate': win_rates.max(),
        'win_rate_range': win_rates.max() - win_rates.min(),
    }

    return stability


def print_stability_report(stability: Dict):
    """Pretty print stability metrics"""

    print("\n" + "="*70)
    print(f"STABILITY ANALYSIS: {stability['threshold']:.0%} Threshold")
    print("="*70)

    print(f"\n📊 Performance Across {stability['total_windows']} Windows:")
    print(f"   Avg Win Rate: {stability['avg_win_rate']:.2f}%")
    print(f"   Avg Profit Factor: {stability['avg_profit_factor']:.2f}")
    print(f"   Avg Sharpe Ratio: {stability['avg_sharpe_ratio']:.2f}")

    print(f"\n📉 Stability (Coefficient of Variation - target <10%):")
    print(f"   Win Rate CV: {stability['cv_win_rate_pct']:.2f}% {'✅' if stability['cv_win_rate_pct'] < 10 else '⚠️'}")
    print(f"   Profit Factor CV: {stability['cv_profit_factor_pct']:.2f}% {'✅' if stability['cv_profit_factor_pct'] < 10 else '⚠️'}")
    print(f"   Sharpe Ratio CV: {stability['cv_sharpe_ratio_pct']:.2f}% {'✅' if stability['cv_sharpe_ratio_pct'] < 10 else '⚠️'}")

    print(f"\n✅ Consistency (Windows Meeting Success Criteria):")
    print(f"   Win Rate ≥52%: {stability['windows_meeting_wr_target']}/{stability['total_windows']} ({stability['windows_meeting_wr_target']/stability['total_windows']*100:.1f}%)")
    print(f"   Profit Factor ≥1.25: {stability['windows_meeting_pf_target']}/{stability['total_windows']} ({stability['windows_meeting_pf_target']/stability['total_windows']*100:.1f}%)")
    print(f"   Sharpe Ratio ≥1.5: {stability['windows_meeting_sharpe_target']}/{stability['total_windows']} ({stability['windows_meeting_sharpe_target']/stability['total_windows']*100:.1f}%)")
    print(f"   Overall Consistency: {stability['consistency_rate_pct']:.1f}%")

    print(f"\n🎯 Optimality (Windows Where This Was Best Threshold):")
    print(f"   Optimal in: {stability['windows_where_optimal']}/{stability['total_windows']} windows ({stability['optimal_rate_pct']:.1f}%)")
    print(f"   Target: ≥70% {'✅' if stability['optimal_rate_pct'] >= 70 else '❌'}")

    print(f"\n📊 Win Rate Range:")
    print(f"   Min: {stability['min_win_rate']:.2f}%")
    print(f"   Max: {stability['max_win_rate']:.2f}%")
    print(f"   Range: {stability['win_rate_range']:.2f}%")

    print("="*70)


def assess_stability(stability: Dict) -> Tuple[bool, str]:
    """
    Assess if threshold passes stability criteria.

    Parameters:
    -----------
    stability : dict
        Stability metrics

    Returns:
    --------
    tuple
        (passes: bool, reason: str)
    """

    issues = []

    # Check CV (should be <10%)
    if stability['cv_win_rate_pct'] >= 10:
        issues.append(f"Win Rate CV too high ({stability['cv_win_rate_pct']:.1f}% ≥ 10%)")

    if stability['cv_profit_factor_pct'] >= 10:
        issues.append(f"Profit Factor CV too high ({stability['cv_profit_factor_pct']:.1f}% ≥ 10%)")

    # Check optimality (should be ≥70%)
    if stability['optimal_rate_pct'] < 70:
        issues.append(f"Not optimal in enough windows ({stability['optimal_rate_pct']:.1f}% < 70%)")

    # Check consistency (should be ≥70%)
    if stability['consistency_rate_pct'] < 70:
        issues.append(f"Consistency too low ({stability['consistency_rate_pct']:.1f}% < 70%)")

    if len(issues) == 0:
        return True, "All stability criteria passed"
    else:
        return False, " | ".join(issues)
