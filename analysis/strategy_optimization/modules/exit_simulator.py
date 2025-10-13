"""
Exit Threshold Simulator Module
================================

Purpose:
--------
Simulate alternative exit thresholds on historical trades to find optimal settings.

Key Functions:
--------------
- simulate_exit_threshold(): Test what would have happened with different threshold
- simulate_all_thresholds(): Test multiple thresholds on a trade dataset
- find_optimal_threshold(): Identify best threshold based on metrics

Methodology:
------------
For each trade:
1. Load 5-minute bars during the trade from base_data
2. Calculate MACD peak (for Buy) or valley (for Sell) after entry
3. Find when MACD histogram crosses threshold % of peak/valley
4. Calculate simulated return at that exit point
5. Compare with actual return

Author: Strategy Optimization Pipeline
Date: 2025-10-04
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def simulate_exit_threshold(trade: pd.Series,
                            base_data_dir: str,
                            threshold: float,
                            base_data_cache: Optional[Dict] = None) -> Dict:
    """
    Simulate what would have happened with a different exit threshold.

    Parameters:
    -----------
    trade : pd.Series
        Trade record with Entry Time, Exit Time, Entry Price, Trade Type, ticker
    base_data_dir : str
        Path to base_data directory
    threshold : float
        Exit threshold (0.50 to 0.95) - % of MACD peak/valley
    base_data_cache : dict, optional
        Cache for base_data to avoid repeated file reads

    Returns:
    --------
    dict
        Simulated trade outcome with new exit point and metrics
    """

    try:
        # Load base data for this ticker
        ticker = trade['ticker']

        if base_data_cache is not None and ticker in base_data_cache:
            base_data = base_data_cache[ticker]
        else:
            base_file = Path(base_data_dir) / f"{ticker}_Base_2022-01-01_to_2025-08-31.csv"
            if not base_file.exists():
                return {'error': f'No base data for {ticker}', 'threshold': threshold}

            base_data = pd.read_csv(base_file)
            base_data['timestamp'] = pd.to_datetime(base_data['timestamp']).dt.tz_localize(None)

            if base_data_cache is not None:
                base_data_cache[ticker] = base_data

        # Get bars during the trade
        entry_time = pd.to_datetime(trade['Entry Time'])
        exit_time = pd.to_datetime(trade['Exit Time'])

        trade_bars = base_data[
            (base_data['timestamp'] >= entry_time) &
            (base_data['timestamp'] <= exit_time)
        ].copy()

        if len(trade_bars) == 0:
            return {'error': 'No bars during trade', 'threshold': threshold}

        # Check if we have MACD histogram in base_data
        macd_col = None
        for col in base_data.columns:
            if 'macd' in col.lower() and 'histogram' in col.lower() and '15' in col:
                macd_col = col
                break

        if macd_col is None:
            # Try to find any MACD column
            for col in base_data.columns:
                if 'macd' in col.lower() and '15' in col:
                    macd_col = col
                    break

        if macd_col is None:
            return {'error': 'No MACD column found', 'threshold': threshold}

        # Get entry and actual exit prices
        entry_price = trade['Entry Price']
        actual_exit_price = trade['Exit Price']
        trade_type = trade['Trade Type']

        # Calculate MACD peak or valley based on trade type
        macd_values = trade_bars[macd_col].dropna()

        if len(macd_values) == 0:
            return {'error': 'No valid MACD values', 'threshold': threshold}

        if trade_type == 'Buy':
            # For Buy trades, find peak MACD (maximum)
            peak_macd = macd_values.max()
            exit_level = peak_macd * threshold

            # Find first bar after peak where MACD drops to threshold
            peak_idx = macd_values.idxmax()
            bars_after_peak = trade_bars.loc[peak_idx:][macd_col]

            # Find when MACD crosses below exit_level
            cross_bars = bars_after_peak[bars_after_peak <= exit_level]

        else:  # Sell trade
            # For Sell trades, find valley MACD (minimum, most negative)
            valley_macd = macd_values.min()
            exit_level = valley_macd * threshold  # Still multiply (valley is negative)

            # Find first bar after valley where MACD rises to threshold
            valley_idx = macd_values.idxmin()
            bars_after_valley = trade_bars.loc[valley_idx:][macd_col]

            # Find when MACD crosses above exit_level (rising from negative)
            cross_bars = bars_after_valley[bars_after_valley >= exit_level]

        # Determine simulated exit
        if len(cross_bars) > 0:
            # Found threshold crossing
            sim_exit_idx = cross_bars.index[0]
            sim_exit_bar = trade_bars.loc[sim_exit_idx]
            sim_exit_time = sim_exit_bar['timestamp']
            sim_exit_price = sim_exit_bar['close']
        else:
            # Threshold never crossed, use actual exit
            sim_exit_time = exit_time
            sim_exit_price = actual_exit_price

        # Calculate returns
        if trade_type == 'Buy':
            sim_return_pct = ((sim_exit_price - entry_price) / entry_price) * 100
        else:  # Sell
            sim_return_pct = ((entry_price - sim_exit_price) / entry_price) * 100

        actual_return_pct = trade['percentage_return']

        # Calculate duration
        sim_duration_minutes = (sim_exit_time - entry_time).total_seconds() / 60
        actual_duration_minutes = (exit_time - entry_time).total_seconds() / 60

        # Return simulation results
        result = {
            'threshold': threshold,
            'ticker': ticker,
            'trade_type': trade_type,

            # Simulated outcome
            'sim_exit_time': sim_exit_time,
            'sim_exit_price': sim_exit_price,
            'sim_return_pct': sim_return_pct,
            'sim_duration_minutes': sim_duration_minutes,

            # Actual outcome (for comparison)
            'actual_exit_time': exit_time,
            'actual_exit_price': actual_exit_price,
            'actual_return_pct': actual_return_pct,
            'actual_duration_minutes': actual_duration_minutes,

            # Deltas
            'return_delta_pct': sim_return_pct - actual_return_pct,
            'duration_delta_minutes': sim_duration_minutes - actual_duration_minutes,

            # Trade outcome change
            'was_winner': actual_return_pct > 0,
            'is_winner': sim_return_pct > 0,
            'outcome_changed': (actual_return_pct > 0) != (sim_return_pct > 0),

            # MACD info
            'macd_peak_valley': peak_macd if trade_type == 'Buy' else valley_macd,
            'macd_exit_level': exit_level,
        }

        return result

    except Exception as e:
        return {'error': str(e), 'threshold': threshold}


def simulate_all_thresholds(trades_df: pd.DataFrame,
                           base_data_dir: str,
                           thresholds: List[float],
                           trade_type_filter: Optional[str] = None,
                           progress_callback=None) -> pd.DataFrame:
    """
    Simulate all threshold combinations on a dataset of trades.

    Parameters:
    -----------
    trades_df : pd.DataFrame
        Trade data with required columns
    base_data_dir : str
        Path to base_data directory
    thresholds : list of float
        List of thresholds to test (e.g., [0.50, 0.55, 0.60, ..., 0.95])
    trade_type_filter : str, optional
        'Buy' or 'Sell' to test only specific trade type
    progress_callback : callable, optional
        Function(current, total, threshold) for progress updates

    Returns:
    --------
    pd.DataFrame
        Results for all threshold simulations
    """

    print(f"\n🔄 Simulating {len(thresholds)} thresholds on {len(trades_df):,} trades...")

    if trade_type_filter:
        trades_df = trades_df[trades_df['Trade Type'] == trade_type_filter].copy()
        print(f"   Filtered to {trade_type_filter} trades: {len(trades_df):,}")

    all_results = []
    base_data_cache = {}  # Cache base_data to avoid repeated loads

    for threshold_idx, threshold in enumerate(thresholds):
        print(f"\n   Testing threshold: {threshold:.0%} ({threshold_idx + 1}/{len(thresholds)})")

        threshold_results = []

        for idx, trade in trades_df.iterrows():
            result = simulate_exit_threshold(
                trade,
                base_data_dir,
                threshold,
                base_data_cache
            )

            # Add original trade info
            result['original_index'] = idx
            threshold_results.append(result)

            # Progress callback
            if progress_callback and (len(threshold_results) % 500 == 0):
                progress_callback(len(threshold_results), len(trades_df), threshold)

        # Convert to DataFrame
        threshold_df = pd.DataFrame(threshold_results)
        all_results.append(threshold_df)

        # Quick summary
        if 'error' in threshold_df.columns:
            valid = threshold_df[~threshold_df['error'].notna()]
        else:
            valid = threshold_df

        if len(valid) > 0:
            win_rate = (valid['is_winner'].sum() / len(valid)) * 100
            avg_return = valid['sim_return_pct'].mean()
            print(f"      ✓ Valid: {len(valid):,}/{len(threshold_df):,} | WR: {win_rate:.2f}% | Avg Return: {avg_return:.3f}%")

    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)

    print(f"\n   ✓ Completed: {len(combined):,} simulations ({len(thresholds)} thresholds × {len(trades_df):,} trades)")

    return combined


def calculate_threshold_metrics(simulation_results: pd.DataFrame,
                                threshold: float) -> Dict:
    """
    Calculate performance metrics for a specific threshold.

    Parameters:
    -----------
    simulation_results : pd.DataFrame
        Simulation results from simulate_all_thresholds
    threshold : float
        Threshold to analyze

    Returns:
    --------
    dict
        Performance metrics for this threshold
    """

    # Filter to this threshold and valid results
    threshold_mask = simulation_results['threshold'] == threshold
    if 'error' in simulation_results.columns:
        valid_mask = ~simulation_results['error'].notna()
        threshold_data = simulation_results[threshold_mask & valid_mask].copy()
    else:
        threshold_data = simulation_results[threshold_mask].copy()

    if len(threshold_data) == 0:
        return {'error': 'No valid data', 'threshold': threshold}

    # Calculate metrics
    returns = threshold_data['sim_return_pct']

    wins = returns > 0
    losses = returns < 0

    num_wins = wins.sum()
    num_losses = losses.sum()
    total_trades = len(threshold_data)

    win_rate = (num_wins / total_trades) * 100 if total_trades > 0 else 0

    avg_win = returns[wins].mean() if num_wins > 0 else 0
    avg_loss = abs(returns[losses].mean()) if num_losses > 0 else 0

    total_wins = returns[wins].sum() if num_wins > 0 else 0
    total_losses = abs(returns[losses].sum()) if num_losses > 0 else 0

    profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')

    # Sharpe ratio (annualized)
    if returns.std() > 0:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # Total return
    total_return = returns.sum()

    # Average duration
    avg_duration_minutes = threshold_data['sim_duration_minutes'].mean()

    # Outcome changes
    losers_to_winners = threshold_data[
        (threshold_data['was_winner'] == False) &
        (threshold_data['is_winner'] == True)
    ]

    winners_to_losers = threshold_data[
        (threshold_data['was_winner'] == True) &
        (threshold_data['is_winner'] == False)
    ]

    metrics = {
        'threshold': threshold,
        'total_trades': total_trades,
        'num_wins': num_wins,
        'num_losses': num_losses,
        'win_rate_pct': win_rate,
        'profit_factor': profit_factor,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'total_return_pct': total_return,
        'sharpe_ratio': sharpe_ratio,
        'avg_duration_minutes': avg_duration_minutes,

        # Outcome changes vs baseline (80%)
        'losers_to_winners_count': len(losers_to_winners),
        'winners_to_losers_count': len(winners_to_losers),
        'net_outcome_improvement': len(losers_to_winners) - len(winners_to_losers),
    }

    return metrics


def find_optimal_threshold(all_metrics: List[Dict],
                          success_criteria: Dict) -> Tuple[float, Dict]:
    """
    Find optimal threshold based on success criteria.

    Parameters:
    -----------
    all_metrics : list of dict
        Metrics for all tested thresholds
    success_criteria : dict
        Criteria that must be met (min_win_rate, min_profit_factor, etc.)

    Returns:
    --------
    tuple
        (optimal_threshold, metrics_dict)
    """

    # Filter to candidates that meet minimum criteria
    candidates = []

    for metrics in all_metrics:
        if 'error' in metrics:
            continue

        meets_criteria = True

        if 'min_win_rate' in success_criteria:
            if metrics['win_rate_pct'] < success_criteria['min_win_rate']:
                meets_criteria = False

        if 'min_profit_factor' in success_criteria:
            if metrics['profit_factor'] < success_criteria['min_profit_factor']:
                meets_criteria = False

        if 'min_sharpe_ratio' in success_criteria:
            if metrics['sharpe_ratio'] < success_criteria['min_sharpe_ratio']:
                meets_criteria = False

        if meets_criteria:
            candidates.append(metrics)

    if len(candidates) == 0:
        # No threshold meets criteria, return best by composite score
        candidates = [m for m in all_metrics if 'error' not in m]

    # Rank by composite score
    def composite_score(m):
        # Weighted score: WR + PF + Sharpe
        wr_score = m['win_rate_pct'] * 1.0
        pf_score = m['profit_factor'] * 10.0
        sharpe_score = m['sharpe_ratio'] * 10.0
        return wr_score + pf_score + sharpe_score

    best = max(candidates, key=composite_score)

    return best['threshold'], best
