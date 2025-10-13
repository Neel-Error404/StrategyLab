#!/usr/bin/env python3
"""
Stop-Loss Impact Analysis (Config Driven)
=========================================

Evaluates the effect of applying percentage-based stop-loss rules on historical
trades. Supports configurable thresholds via YAML, runs a base threshold
analysis, and compares multiple candidates to highlight the optimal level.

Outputs
-------
- CSV:  stop_loss_scenarios (threshold comparison table)
- JSON: stop_loss_summary   (detailed metrics + recommendation)

Usage
-----
    python 04_stop_loss_simulation.py --config ../../config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config_loader import (
    get_analysis_config,
    load_config,
    resolve_artifact_path,
    resolve_paths,
)
from modules.data_loader import load_trades

MODULE_NAME = "stop_loss_simulation"


def calculate_stop_loss_levels(df: pd.DataFrame, buy_pct: float, sell_pct: float) -> pd.DataFrame:
    """Attach stop-loss trigger prices for BUY/SELL trades."""
    df = df.copy()
    df['stop_loss_price'] = np.where(
        df['Trade Type'] == 'Buy',
        df['Entry Price'] * (1 - buy_pct / 100.0),
        df['Entry Price'] * (1 + sell_pct / 100.0),
    )
    return df


def analyze_buy_stop_loss(buy_trades: pd.DataFrame, stop_loss_pct: float) -> Dict[str, Any]:
    buy_trades = buy_trades.copy()
    buy_trades['stop_loss_triggered'] = buy_trades['Low During Trade'] <= buy_trades['stop_loss_price']
    buy_trades['stop_loss_pnl'] = np.where(
        buy_trades['stop_loss_triggered'],
        buy_trades['stop_loss_price'] - buy_trades['Entry Price'],
        buy_trades['Profit (Currency)'],
    )

    total_trades = len(buy_trades)
    stopped_trades = int(buy_trades['stop_loss_triggered'].sum())
    original_pnl = float(buy_trades['Profit (Currency)'].sum())
    stop_loss_pnl = float(buy_trades['stop_loss_pnl'].sum())
    pnl_difference = stop_loss_pnl - original_pnl

    original_win_rate = (buy_trades['Profit (Currency)'] > 0).mean() * 100 if total_trades else 0.0
    stop_loss_win_rate = (buy_trades['stop_loss_pnl'] > 0).mean() * 100 if total_trades else 0.0

    print(f"\n🟢 BUY TRADES @ {stop_loss_pct}%")
    print(f"  Total trades: {total_trades:,}")
    print(f"  Triggered: {stopped_trades:,} ({stopped_trades/total_trades*100:.1f}%)" if total_trades else "  Triggered: 0")
    print(f"  Original P&L: ₹{original_pnl:,.2f} → Stop-Loss P&L: ₹{stop_loss_pnl:,.2f} (Δ ₹{pnl_difference:,.2f})")
    print(f"  Win rate: {original_win_rate:.1f}% → {stop_loss_win_rate:.1f}% (Δ {stop_loss_win_rate-original_win_rate:+.1f}pp)")

    stopped_detail = buy_trades[buy_trades['stop_loss_triggered']].copy()
    saved_losers = int((stopped_detail['Profit (Currency)'] < 0).sum())

    return {
        'total_trades': total_trades,
        'stopped_trades': stopped_trades,
        'original_pnl': original_pnl,
        'stop_loss_pnl': stop_loss_pnl,
        'pnl_difference': pnl_difference,
        'original_win_rate': original_win_rate,
        'stop_loss_win_rate': stop_loss_win_rate,
        'saved_losers': saved_losers,
    }


def analyze_sell_stop_loss(sell_trades: pd.DataFrame, stop_loss_pct: float) -> Dict[str, Any]:
    sell_trades = sell_trades.copy()
    sell_trades['stop_loss_triggered'] = sell_trades['High During Trade'] >= sell_trades['stop_loss_price']
    sell_trades['stop_loss_pnl'] = np.where(
        sell_trades['stop_loss_triggered'],
        sell_trades['Entry Price'] - sell_trades['stop_loss_price'],
        sell_trades['Profit (Currency)'],
    )

    total_trades = len(sell_trades)
    stopped_trades = int(sell_trades['stop_loss_triggered'].sum())
    original_pnl = float(sell_trades['Profit (Currency)'].sum())
    stop_loss_pnl = float(sell_trades['stop_loss_pnl'].sum())
    pnl_difference = stop_loss_pnl - original_pnl

    original_win_rate = (sell_trades['Profit (Currency)'] > 0).mean() * 100 if total_trades else 0.0
    stop_loss_win_rate = (sell_trades['stop_loss_pnl'] > 0).mean() * 100 if total_trades else 0.0

    print(f"\n🔴 SELL TRADES @ {stop_loss_pct}%")
    print(f"  Total trades: {total_trades:,}")
    print(f"  Triggered: {stopped_trades:,} ({stopped_trades/total_trades*100:.1f}%)" if total_trades else "  Triggered: 0")
    print(f"  Original P&L: ₹{original_pnl:,.2f} → Stop-Loss P&L: ₹{stop_loss_pnl:,.2f} (Δ ₹{pnl_difference:,.2f})")
    print(f"  Win rate: {original_win_rate:.1f}% → {stop_loss_win_rate:.1f}% (Δ {stop_loss_win_rate-original_win_rate:+.1f}pp)")

    stopped_detail = sell_trades[sell_trades['stop_loss_triggered']].copy()
    saved_losers = int((stopped_detail['Profit (Currency)'] < 0).sum())

    return {
        'total_trades': total_trades,
        'stopped_trades': stopped_trades,
        'original_pnl': original_pnl,
        'stop_loss_pnl': stop_loss_pnl,
        'pnl_difference': pnl_difference,
        'original_win_rate': original_win_rate,
        'stop_loss_win_rate': stop_loss_win_rate,
        'saved_losers': saved_losers,
    }


def analyze_combined_impact(
    buy_results: Dict[str, Any],
    sell_results: Dict[str, Any],
    df: pd.DataFrame,
) -> Dict[str, Any]:
    total_original_pnl = buy_results['original_pnl'] + sell_results['original_pnl']
    total_stop_loss_pnl = buy_results['stop_loss_pnl'] + sell_results['stop_loss_pnl']
    total_pnl_difference = total_stop_loss_pnl - total_original_pnl

    total_trades = len(df)
    total_stopped = buy_results['stopped_trades'] + sell_results['stopped_trades']

    if total_original_pnl == 0:
        pct_change = 0.0
    else:
        pct_change = (total_pnl_difference / abs(total_original_pnl)) * 100

    print(f"\n⚖️ COMBINED IMPACT")
    print(f"  Total stopped trades: {total_stopped:,} ({total_stopped/total_trades*100:.1f}%)")
    print(f"  Original P&L: ₹{total_original_pnl:,.2f} → Stop-Loss P&L: ₹{total_stop_loss_pnl:,.2f} (Δ ₹{total_pnl_difference:,.2f}; {pct_change:+.1f}%)")

    recommendation = "AVOID STOP-LOSS - Reduces performance"
    if total_pnl_difference > 100_000:
        recommendation = "IMPLEMENT STOP-LOSS - Significant improvement"
    elif total_pnl_difference > 0:
        recommendation = "CONSIDER STOP-LOSS - Marginal improvement"

    return {
        'total_pnl_difference': total_pnl_difference,
        'total_stopped': total_stopped,
        'recommendation': recommendation,
        'pct_change': pct_change,
    }


def simulate_stop_loss_impact(df: pd.DataFrame, stop_loss_pct: float) -> Dict[str, Any]:
    """Run stop-loss simulation for a single threshold."""
    df_levels = calculate_stop_loss_levels(df, stop_loss_pct, stop_loss_pct)
    buy_trades = df_levels[df_levels['Trade Type'] == 'Buy']
    sell_trades = df_levels[df_levels['Trade Type'] == 'Sell']

    buy_results = analyze_buy_stop_loss(buy_trades, stop_loss_pct)
    sell_results = analyze_sell_stop_loss(sell_trades, stop_loss_pct)
    combined_results = analyze_combined_impact(buy_results, sell_results, df_levels)

    return {
        'buy': buy_results,
        'sell': sell_results,
        'combined': combined_results,
        'threshold': stop_loss_pct,
    }


def multi_threshold_analysis(df: pd.DataFrame, thresholds: List[float]) -> Tuple[pd.DataFrame, float]:
    """Evaluate multiple thresholds and return summary + best threshold."""
    summaries = []
    for threshold in thresholds:
        print(f"\n--- Testing {threshold}% stop-loss ---")
        result = simulate_stop_loss_impact(df, threshold)
        summaries.append({
            'threshold': threshold,
            'buy_pnl_diff': result['buy']['pnl_difference'],
            'sell_pnl_diff': result['sell']['pnl_difference'],
            'total_pnl_diff': result['combined']['total_pnl_difference'],
            'total_stopped': result['combined']['total_stopped'],
            'buy_stopped_pct': (result['buy']['stopped_trades'] / result['buy']['total_trades'] * 100)
                if result['buy']['total_trades'] else 0.0,
            'sell_stopped_pct': (result['sell']['stopped_trades'] / result['sell']['total_trades'] * 100)
                if result['sell']['total_trades'] else 0.0,
        })

    summary_df = pd.DataFrame(summaries)
    if summary_df.empty:
        return summary_df, 0.0

    optimal_idx = summary_df['total_pnl_diff'].idxmax()
    optimal_threshold = float(summary_df.loc[optimal_idx, 'threshold'])
    print(f"\n🏆 Optimal stop-loss threshold: {optimal_threshold}%")
    print(f"💰 Maximum improvement: ₹{summary_df.loc[optimal_idx, 'total_pnl_diff']:,.2f}")
    return summary_df, optimal_threshold


def save_stop_loss_results(
    config: Dict[str, Any],
    base_result: Dict[str, Any],
    summary_df: pd.DataFrame,
    optimal_threshold: float,
) -> None:
    """Persist summary JSON and comparison CSV using config-driven paths."""
    summary_path = Path(resolve_artifact_path(config, MODULE_NAME, 'stop_loss_summary', artifact_type='json'))
    scenarios_path = Path(resolve_artifact_path(config, MODULE_NAME, 'stop_loss_scenarios', artifact_type='csv'))

    payload = {
        'analysis_date': datetime.now().isoformat(),
        'base_threshold': base_result['threshold'],
        'base_results': base_result,
        'optimal_threshold': optimal_threshold,
        'threshold_comparison': summary_df.to_dict(orient='records'),
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n💾 Saved stop-loss summary → {summary_path}")

    scenarios_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(scenarios_path, index=False)
    print(f"💾 Saved threshold comparison CSV → {scenarios_path}")


def parse_thresholds(module_cfg: Dict[str, Any]) -> List[float]:
    thresholds = module_cfg.get('thresholds')
    if isinstance(thresholds, list) and thresholds:
        return [float(t) for t in thresholds]
    # Default levels if none supplied
    return [1.0, 1.5, 2.0, 2.5, 3.0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Stop-loss impact analysis")
    parser.add_argument("--config", required=True, help="Path to analysis YAML config")
    parser.add_argument("--sample", type=int, help="Optional sample size for quick iteration")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = resolve_paths(config)
    module_cfg = get_analysis_config(config, MODULE_NAME) or {}

    sample_size = args.sample or module_cfg.get('sample_size')
    df = load_trades(config, paths, sample_size=sample_size)

    thresholds = parse_thresholds(module_cfg)
    base_threshold = float(module_cfg.get('base_threshold', thresholds[0] if thresholds else 2.0))
    if base_threshold not in thresholds:
        thresholds = [base_threshold] + thresholds
    thresholds = sorted(set(thresholds))

    print("=" * 80)
    print(f"STOP-LOSS ANALYSIS | Thresholds: {thresholds}")
    print("=" * 80)

    base_result = simulate_stop_loss_impact(df, base_threshold)
    summary_df, optimal_threshold = multi_threshold_analysis(df, thresholds)
    save_stop_loss_results(config, base_result, summary_df, optimal_threshold)

    best_msg = (
        f"Optimal threshold {optimal_threshold}% "
        f"(Δ ₹{summary_df.loc[summary_df['threshold'] == optimal_threshold, 'total_pnl_diff'].iloc[0]:,.2f})"
        if not summary_df.empty else "No improvement detected"
    )

    print("\n✅ STOP-LOSS ANALYSIS COMPLETE")
    print(f"   Base threshold: {base_threshold}%")
    print(f"   Recommendation: {base_result['combined']['recommendation']}")
    print(f"   {best_msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
