#!/usr/bin/env python3
"""
Risk-Adjusted Pattern Analysis (Config Driven)
==============================================

Computes risk metrics for key trade patterns (e.g. cascade categories) using the
merged trade dataset. Outputs a CSV table and optional Markdown summary that
highlight profit factor, risk-reward, drawdown, volatility, and sharpe-like
measures for each pattern.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config_loader import (
    get_analysis_config,
    load_config,
    resolve_artifact_path,
    resolve_paths,
)
from modules.data_loader import load_trades

MODULE_NAME = "risk_adjusted_patterns"


def categorize_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Adds cascade-style category labels to a trade dataframe."""
    df = df.sort_values(['ticker', 'Entry Time']).reset_index(drop=True)
    df['prev_ticker'] = df['ticker'].shift(1)
    df['prev_entry_date'] = df['Entry Time'].dt.date.shift(1)
    df['prev_trade_type'] = df['Trade Type'].shift(1)
    df['prev_exit_time'] = df['Exit Time'].shift(1)
    df['Entry Date'] = df['Entry Time'].dt.date

    def classify(row: pd.Series) -> str:
        if pd.isna(row['prev_ticker']):
            return 'FIRST_TRADE_OVERALL'
        if row['ticker'] != row['prev_ticker']:
            return 'FIRST_TRADE_FOR_TICKER'
        if row['Entry Date'] != row['prev_entry_date']:
            return 'FIRST_TRADE_OF_DAY'
        if row['Trade Type'] == row['prev_trade_type']:
            return 'CONSECUTIVE_SAME_DIRECTION'
        return 'CONSECUTIVE_OPPOSITE_DIRECTION'

    df['trade_category'] = df.apply(classify, axis=1)
    return df


def calculate_risk_metrics(profits: pd.Series) -> Dict[str, Any]:
    profits = profits.dropna()
    if profits.empty:
        return {
            'trade_count': 0,
            'total_profit': 0.0,
            'avg_profit': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'risk_reward_ratio': 0.0,
            'max_drawdown': 0.0,
            'volatility': 0.0,
            'sharpe_like': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
        }

    total_profit = profits.sum()
    avg_profit = profits.mean()
    trade_count = len(profits)

    wins = profits[profits > 0]
    losses = profits[profits <= 0]

    win_rate = (len(wins) / trade_count) * 100 if trade_count else 0.0
    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = abs(losses.mean()) if not losses.empty else 0.0

    gross_profit = wins.sum() if not wins.empty else 0.0
    gross_loss = abs(losses.sum()) if not losses.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)

    risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else (float('inf') if avg_win > 0 else 0.0)

    volatility = profits.std()
    sharpe_like = avg_profit / volatility if volatility and volatility > 0 else 0.0

    cumulative = profits.cumsum()
    rolling_max = cumulative.expanding().max()
    drawdown = cumulative - rolling_max
    max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0.0

    largest_win = profits.max()
    largest_loss = abs(profits.min())

    return {
        'trade_count': trade_count,
        'total_profit': float(total_profit),
        'avg_profit': float(avg_profit),
        'win_rate': float(win_rate),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'profit_factor': float(profit_factor),
        'risk_reward_ratio': float(risk_reward_ratio),
        'max_drawdown': float(max_drawdown),
        'volatility': float(volatility) if not np.isnan(volatility) else 0.0,
        'sharpe_like': float(sharpe_like),
        'largest_win': float(largest_win),
        'largest_loss': float(largest_loss),
    }


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    return [col for col in columns if col not in df.columns]


def render_markdown(table: pd.DataFrame) -> str:
    lines = [
        "# Risk-Adjusted Pattern Summary",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        table.to_markdown(index=False),
        "",
        "_Metrics include average profit, win rate, profit factor, risk-reward ratio, "
        "drawdown, sharpe-like ratio, and extremes._",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Risk-adjusted pattern analysis")
    parser.add_argument("--config", required=True, help="Path to analysis YAML config")
    parser.add_argument("--sample", type=int, help="Optional sample size for quick testing")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = resolve_paths(config)
    module_cfg = get_analysis_config(config, MODULE_NAME) or {}

    sample_size = args.sample or module_cfg.get('sample_size')
    trades_df = load_trades(config, paths, sample_size=sample_size)

    missing_cols = ensure_columns(trades_df, ['Profit (Currency)', 'Trade Type', 'Entry Time', 'Exit Time'])
    if missing_cols:
        print(f"❌ Missing columns required for risk analysis: {missing_cols}")
        return 1

    categorized_df = categorize_trades(trades_df)

    grouping_columns = module_cfg.get('grouping_columns') or ['trade_category']
    valid_group_cols = [col for col in grouping_columns if col in categorized_df.columns]
    if not valid_group_cols:
        print("⚠️  No valid grouping columns found; defaulting to 'trade_category'.")
        valid_group_cols = ['trade_category']

    results: List[Dict[str, Any]] = []
    for keys, subset in categorized_df.groupby(valid_group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(valid_group_cols, keys))
        metrics = calculate_risk_metrics(subset['Profit (Currency)'])
        results.append({**key_dict, **metrics})

    if not results:
        print("⚠️  No pattern metrics computed.")
        return 0

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(valid_group_cols).reset_index(drop=True)

    csv_path = Path(resolve_artifact_path(config, MODULE_NAME, 'pattern_metrics', artifact_type='csv'))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved pattern metrics → {csv_path}")

    # Optional markdown output if configured in YAML (artifact defined)
    try:
        md_path = Path(resolve_artifact_path(config, MODULE_NAME, 'pattern_report', artifact_type='markdown'))
    except Exception:
        md_path = None

    if md_path:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown(result_df))
        print(f"📝 Saved pattern summary → {md_path}")

    print("\n✅ Risk-adjusted pattern analysis complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
