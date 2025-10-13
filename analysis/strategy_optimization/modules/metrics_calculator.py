"""
Traditional Trading Metrics Calculator
=======================================

Purpose:
--------
Calculate traditional trading performance metrics for baseline and optimization.

Metrics Calculated:
-------------------
- Win Rate (%)
- Profit Factor
- Average Win / Average Loss
- Max Drawdown (%)
- Sharpe Ratio
- Total Return (%)
- Number of Trades
- Average Trade Duration
- Risk-Reward Ratio

Author: Strategy Optimization Pipeline
Date: 2025-10-04
"""

import pandas as pd
import numpy as np
from datetime import datetime


def calculate_traditional_metrics(trade_data: pd.DataFrame,
                                  initial_capital: float = 100000) -> dict:
    """
    Calculate traditional trading metrics.

    Parameters:
    -----------
    trade_data : pd.DataFrame
        Trade data with columns: percentage_return, Entry Time, Exit Time
    initial_capital : float
        Starting capital (default: 100,000)

    Returns:
    --------
    dict
        Dictionary of calculated metrics
    """

    if len(trade_data) == 0:
        return {'error': 'No trades provided', 'total_trades': 0}

    # Ensure we have the required columns
    required_cols = ['percentage_return', 'Entry Time', 'Exit Time']
    missing_cols = [col for col in required_cols if col not in trade_data.columns]
    if missing_cols:
        return {'error': f'Missing columns: {missing_cols}'}

    # Basic trade statistics
    total_trades = len(trade_data)
    returns = trade_data['percentage_return']

    # Win/Loss statistics
    winning_trades = returns[returns > 0]
    losing_trades = returns[returns < 0]
    breakeven_trades = returns[returns == 0]

    num_wins = len(winning_trades)
    num_losses = len(losing_trades)
    num_breakeven = len(breakeven_trades)

    win_rate = (num_wins / total_trades * 100) if total_trades > 0 else 0

    avg_win = winning_trades.mean() if num_wins > 0 else 0
    avg_loss = abs(losing_trades.mean()) if num_losses > 0 else 0

    # Profit Factor
    total_wins = winning_trades.sum() if num_wins > 0 else 0
    total_losses = abs(losing_trades.sum()) if num_losses > 0 else 0

    profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')

    # Total return
    total_return_pct = returns.sum()

    # Equity curve and drawdown
    equity_curve = (1 + returns / 100).cumprod() * initial_capital
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max * 100
    max_drawdown = drawdown.min()

    # Sharpe Ratio (annualized, assuming 252 trading days)
    daily_returns = returns.values
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # Trade duration statistics
    trade_data_copy = trade_data.copy()
    trade_data_copy['Entry Time'] = pd.to_datetime(trade_data_copy['Entry Time'])
    trade_data_copy['Exit Time'] = pd.to_datetime(trade_data_copy['Exit Time'])
    trade_data_copy['duration_hours'] = (
        trade_data_copy['Exit Time'] - trade_data_copy['Entry Time']
    ).dt.total_seconds() / 3600

    avg_duration_hours = trade_data_copy['duration_hours'].mean()
    median_duration_hours = trade_data_copy['duration_hours'].median()

    # Risk-Reward Ratio
    risk_reward_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0

    # Expectancy (average profit per trade)
    expectancy = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * avg_loss)

    # Consecutive wins/losses
    trade_results = (returns > 0).astype(int)
    trade_results[returns < 0] = -1
    trade_results[returns == 0] = 0

    # Calculate consecutive streaks
    streaks = []
    current_streak = 0
    for result in trade_results:
        if result == 0:
            continue
        if current_streak == 0:
            current_streak = result
        elif (current_streak > 0 and result > 0) or (current_streak < 0 and result < 0):
            current_streak += result
        else:
            streaks.append(current_streak)
            current_streak = result
    if current_streak != 0:
        streaks.append(current_streak)

    max_consecutive_wins = max([s for s in streaks if s > 0], default=0)
    max_consecutive_losses = abs(min([s for s in streaks if s < 0], default=0))

    # Final equity
    final_equity = equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital

    metrics = {
        # Basic statistics
        'total_trades': total_trades,
        'num_wins': num_wins,
        'num_losses': num_losses,
        'num_breakeven': num_breakeven,

        # Performance metrics
        'win_rate_pct': win_rate,
        'profit_factor': profit_factor,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'risk_reward_ratio': risk_reward_ratio,
        'expectancy_pct': expectancy,

        # Return metrics
        'total_return_pct': total_return_pct,
        'final_equity': final_equity,
        'return_on_capital_pct': ((final_equity - initial_capital) / initial_capital) * 100,

        # Risk metrics
        'max_drawdown_pct': max_drawdown,
        'sharpe_ratio': sharpe_ratio,

        # Duration metrics
        'avg_duration_hours': avg_duration_hours,
        'median_duration_hours': median_duration_hours,

        # Streaks
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
    }

    return metrics


def print_metrics_summary(metrics: dict, title: str = "TRADING METRICS"):
    """Pretty print trading metrics summary"""

    if 'error' in metrics:
        print(f"\n❌ {metrics['error']}")
        return

    print("\n" + "="*70)
    print(f"{title}")
    print("="*70)

    print(f"\n📊 Trade Statistics:")
    print(f"   Total Trades: {metrics['total_trades']:,}")
    print(f"   Wins: {metrics['num_wins']:,}")
    print(f"   Losses: {metrics['num_losses']:,}")
    print(f"   Breakeven: {metrics['num_breakeven']:,}")

    print(f"\n✅ Win/Loss Metrics:")
    print(f"   Win Rate: {metrics['win_rate_pct']:.2f}%")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"   Average Win: {metrics['avg_win_pct']:.2f}%")
    print(f"   Average Loss: {metrics['avg_loss_pct']:.2f}%")
    print(f"   Risk-Reward Ratio: {metrics['risk_reward_ratio']:.2f}")
    print(f"   Expectancy: {metrics['expectancy_pct']:.2f}% per trade")

    print(f"\n💰 Return Metrics:")
    print(f"   Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"   Return on Capital: {metrics['return_on_capital_pct']:.2f}%")
    print(f"   Final Equity: ${metrics['final_equity']:,.2f}")

    print(f"\n⚠️ Risk Metrics:")
    print(f"   Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

    print(f"\n⏱️ Duration Metrics:")
    print(f"   Average Duration: {metrics['avg_duration_hours']:.2f} hours")
    print(f"   Median Duration: {metrics['median_duration_hours']:.2f} hours")

    print(f"\n🔥 Streaks:")
    print(f"   Max Consecutive Wins: {metrics['max_consecutive_wins']}")
    print(f"   Max Consecutive Losses: {metrics['max_consecutive_losses']}")

    print("="*70)


def compare_metrics(baseline: dict, optimized: dict) -> dict:
    """
    Compare baseline vs optimized metrics and calculate improvements.

    Parameters:
    -----------
    baseline : dict
        Baseline metrics
    optimized : dict
        Optimized metrics

    Returns:
    --------
    dict
        Comparison results with improvements
    """

    comparison = {}

    # Calculate improvements for key metrics
    metrics_to_compare = [
        'win_rate_pct',
        'profit_factor',
        'avg_win_pct',
        'avg_loss_pct',
        'total_return_pct',
        'max_drawdown_pct',
        'sharpe_ratio',
        'expectancy_pct',
        'total_trades'
    ]

    for metric in metrics_to_compare:
        if metric in baseline and metric in optimized:
            base_val = baseline[metric]
            opt_val = optimized[metric]

            if base_val != 0:
                improvement_pct = ((opt_val - base_val) / abs(base_val)) * 100
            else:
                improvement_pct = 0

            comparison[metric] = {
                'baseline': base_val,
                'optimized': opt_val,
                'improvement_pct': improvement_pct,
                'improvement_abs': opt_val - base_val
            }

    return comparison


def print_comparison(comparison: dict, title: str = "BASELINE VS OPTIMIZED"):
    """Pretty print comparison results"""

    print("\n" + "="*70)
    print(f"{title}")
    print("="*70)

    for metric, values in comparison.items():
        base = values['baseline']
        opt = values['optimized']
        imp_pct = values['improvement_pct']
        imp_abs = values['improvement_abs']

        # Determine if improvement is good (higher is better for most metrics)
        bad_metrics = ['avg_loss_pct', 'max_drawdown_pct']  # Lower is better
        is_improvement = imp_abs > 0 if metric not in bad_metrics else imp_abs < 0

        icon = "✅" if is_improvement else "❌"

        metric_name = metric.replace('_', ' ').title()
        print(f"\n{icon} {metric_name}:")
        print(f"   Baseline: {base:.2f}")
        print(f"   Optimized: {opt:.2f}")
        print(f"   Change: {imp_abs:+.2f} ({imp_pct:+.2f}%)")

    print("="*70)
