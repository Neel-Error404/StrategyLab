#!/usr/bin/env python3
"""
Top-N Pattern Breakdown (Config Driven)
======================================

Produces detailed pattern metrics for the top-N tickers (by ranking output).
Useful for drilling deeper into cascade behaviour within the high-performing
subset.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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

MODULE_NAME = "top50_pattern_breakdown"


def ranking_csv(config: Dict[str, Any]) -> Path:
    return Path(resolve_artifact_path(config, 'ticker_ranking', 'ticker_scores', category='generic', artifact_type='csv'))


def select_top_tickers(config: Dict[str, Any], top_n: int) -> List[str]:
    csv_path = ranking_csv(config)
    if not csv_path.exists():
        raise FileNotFoundError(f"Ranking CSV missing at {csv_path}")
    ranking_df = pd.read_csv(csv_path)
    order_col = 'rank' if 'rank' in ranking_df.columns else 'composite_score'
    ranking_df = ranking_df.sort_values(order_col, ascending=True)
    return ranking_df.head(top_n)['ticker'].tolist()


def categorize_trades(df: pd.DataFrame) -> pd.DataFrame:
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


def compute_metrics(profits: pd.Series) -> Dict[str, Any]:
    profits = profits.dropna()
    if profits.empty:
        return {
            'trade_count': 0,
            'win_count': 0,
            'loss_count': 0,
            'total_profit': 0.0,
            'avg_profit': 0.0,
            'accuracy': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'rrr': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'max_win': 0.0,
            'max_loss': 0.0,
            'volatility': 0.0,
            'max_drawdown': 0.0,
        }

    wins = profits[profits > 0]
    losses = profits[profits <= 0]

    trade_count = len(profits)
    win_count = len(wins)
    loss_count = len(losses)

    avg_profit = profits.mean()
    accuracy = (win_count / trade_count) * 100 if trade_count else 0.0

    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = abs(losses.mean()) if not losses.empty else 0.0
    rrr = avg_win / avg_loss if avg_loss > 0 else (float('inf') if avg_win > 0 else 0.0)

    gross_profit = wins.sum() if not wins.empty else 0.0
    gross_loss = abs(losses.sum()) if not losses.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)

    volatility = profits.std()
    sharpe = avg_profit / volatility if volatility and volatility > 0 else 0.0

    cumulative = profits.cumsum()
    rolling_max = cumulative.expanding().max()
    drawdown = cumulative - rolling_max
    max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0.0

    return {
        'trade_count': trade_count,
        'win_count': win_count,
        'loss_count': loss_count,
        'total_profit': float(profits.sum()),
        'avg_profit': float(avg_profit),
        'accuracy': float(accuracy),
        'profit_factor': float(profit_factor),
        'sharpe_ratio': float(sharpe),
        'rrr': float(rrr),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'max_win': float(profits.max()),
        'max_loss': float(abs(profits.min())),
        'volatility': float(volatility) if not np.isnan(volatility) else 0.0,
        'max_drawdown': float(max_drawdown),
    }


def save_outputs(config: Dict[str, Any], metrics_df: pd.DataFrame) -> None:
    csv_path = Path(resolve_artifact_path(config, MODULE_NAME, 'top50_pattern_metrics', artifact_type='csv'))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved pattern metrics → {csv_path}")

    md_path = Path(resolve_artifact_path(config, MODULE_NAME, 'top50_pattern_summary', artifact_type='markdown'))
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_lines = [
        "# Top-N Pattern Breakdown",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        metrics_df.to_markdown(index=False),
    ]
    md_path.write_text("\n".join(md_lines))
    print(f"📝 Saved pattern summary → {md_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Detailed pattern breakdown for top-N tickers")
    parser.add_argument("--config", required=True, help="Path to analysis YAML config")
    parser.add_argument("--sample", type=int, help="Optional trade sample size")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = resolve_paths(config)
    module_cfg = get_analysis_config(config, MODULE_NAME) or {}

    top_n = int(module_cfg.get('top_n', 50))
    patterns = module_cfg.get('patterns') or [
        'CONSECUTIVE_SAME_DIRECTION',
        'CONSECUTIVE_OPPOSITE_DIRECTION',
        'FIRST_TRADE_OF_DAY',
    ]

    top_tickers = select_top_tickers(config, top_n)
    sample_size = args.sample or module_cfg.get('sample_size')
    trades_df = load_trades(config, paths, sample_size=sample_size)
    top_df = trades_df[trades_df['ticker'].isin(top_tickers)].copy()
    if top_df.empty:
        print("❌ No trades available for top tickers.")
        return 1

    categorized = categorize_trades(top_df)
    rows: List[Dict[str, Any]] = []
    for pattern in patterns:
        subset = categorized[categorized['trade_category'] == pattern]
        metrics = compute_metrics(subset['Profit (Currency)'])
        rows.append({'pattern': pattern, **metrics})

    metrics_df = pd.DataFrame(rows)
    save_outputs(config, metrics_df)
    print("\n✅ Top-N pattern breakdown complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
