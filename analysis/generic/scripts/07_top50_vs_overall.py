#!/usr/bin/env python3
"""
Top-N vs Overall Comparative Analysis
=====================================

Compares risk metrics between the overall dataset and the top-N tickers (as
ranked by the ticker_ranking module). Produces a CSV and Markdown summary that
highlight differences in profit factor, win rate, drawdown, and other stats by
pattern.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

MODULE_NAME = "top50_vs_overall"


def get_ranking_path(config: Dict[str, Any]) -> Path:
    return Path(resolve_artifact_path(config, 'ticker_ranking', 'ticker_scores', category='generic', artifact_type='csv'))


def load_top_tickers(config: Dict[str, Any], top_n: int) -> List[str]:
    ranking_path = get_ranking_path(config)
    if not ranking_path.exists():
        raise FileNotFoundError(f"Ranking output not found at {ranking_path}. Run ticker_ranking first.")
    ranking_df = pd.read_csv(ranking_path)
    if 'ticker' not in ranking_df.columns:
        raise ValueError("Ranking CSV missing 'ticker' column.")
    ranking_df = ranking_df.sort_values('rank' if 'rank' in ranking_df.columns else 'composite_score', ascending=True)
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
        return {'trade_count': 0, 'total_profit': 0.0, 'avg_return_per_trade': 0.0, 'win_rate': 0.0,
                'profit_factor': 0.0, 'risk_reward_ratio': 0.0, 'max_drawdown': 0.0,
                'volatility': 0.0, 'sharpe_like': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
                'largest_win': 0.0, 'largest_loss': 0.0}

    total_profit = profits.sum()
    avg_return_per_trade = profits.mean()
    wins = profits[profits > 0]
    losses = profits[profits <= 0]

    win_rate = (len(wins) / len(profits)) * 100 if len(profits) else 0.0
    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = abs(losses.mean()) if not losses.empty else 0.0

    gross_profit = wins.sum() if not wins.empty else 0.0
    gross_loss = abs(losses.sum()) if not losses.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)
    risk_reward = avg_win / avg_loss if avg_loss > 0 else (float('inf') if avg_win > 0 else 0.0)

    volatility = profits.std()
    sharpe_like = avg_return_per_trade / volatility if volatility and volatility > 0 else 0.0

    cumulative = profits.cumsum()
    rolling_max = cumulative.expanding().max()
    drawdown = cumulative - rolling_max
    max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0.0

    return {
        'trade_count': len(profits),
        'total_profit': float(total_profit),
        'avg_return_per_trade': float(avg_return_per_trade),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor),
        'risk_reward_ratio': float(risk_reward),
        'max_drawdown': float(max_drawdown),
        'volatility': float(volatility) if not np.isnan(volatility) else 0.0,
        'sharpe_like': float(sharpe_like),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'largest_win': float(profits.max()),
        'largest_loss': float(abs(profits.min())),
    }


def analyse_dataset(df: pd.DataFrame, label: str, patterns: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    categorized = categorize_trades(df)
    for pattern in patterns:
        subset = categorized[categorized['trade_category'] == pattern]
        metrics = compute_metrics(subset['Profit (Currency)'])
        rows.append({'dataset': label, 'pattern': pattern, **metrics})
    return rows


def save_outputs(config: Dict[str, Any], summary_df: pd.DataFrame) -> Tuple[Path, Path]:
    csv_path = Path(resolve_artifact_path(config, MODULE_NAME, 'top50_comparison', artifact_type='csv'))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved comparison CSV → {csv_path}")

    md_path = Path(resolve_artifact_path(config, MODULE_NAME, 'top50_report', artifact_type='markdown'))
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_lines = [
        "# Top-N vs Overall Comparison",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        summary_df.to_markdown(index=False),
    ]
    md_path.write_text("\n".join(md_lines))
    print(f"📝 Saved comparison report → {md_path}")
    return csv_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare top-N tickers vs overall dataset")
    parser.add_argument("--config", required=True, help="Path to analysis YAML config")
    parser.add_argument("--sample", type=int, help="Optional sample size for quick testing")
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

    top_tickers = load_top_tickers(config, top_n)
    if not top_tickers:
        print("❌ No tickers available from ranking output.")
        return 1

    sample_size = args.sample or module_cfg.get('sample_size')
    trades_df = load_trades(config, paths, sample_size=sample_size)

    overall_rows = analyse_dataset(trades_df, 'Overall', patterns)
    top_df = trades_df[trades_df['ticker'].isin(top_tickers)].copy()
    top_rows = analyse_dataset(top_df, f'Top {top_n}', patterns)

    summary_df = pd.DataFrame(overall_rows + top_rows)
    save_outputs(config, summary_df)
    print("\n✅ Top-N vs overall analysis complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
