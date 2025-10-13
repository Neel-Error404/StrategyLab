#!/usr/bin/env python3
"""
Validation Check Analysis (Config Driven)
=========================================

Summarises key sanity checks on the merged trade dataset and samples
consecutive trade patterns (same-direction vs opposite-direction) to ensure
the cascade analysis is capturing the intended behaviour.

Outputs
-------
- Markdown report stored via the configured artifact template.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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

MODULE_NAME = "validation_check"


def validate_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
    else:
        print("✅ All required columns present.")
    return missing


def compute_overall_stats(df: pd.DataFrame) -> Dict[str, Any]:
    total = len(df)
    wins = (df['Profit (Currency)'] > 0).sum()
    losses = (df['Profit (Currency)'] < 0).sum()
    breakeven = total - wins - losses
    win_rate = (wins / total * 100) if total else 0.0
    pnl = df['Profit (Currency)'].sum()
    print(f"\n📈 Overall stats: Trades={total:,}, WinRate={win_rate:.2f}%, P&L=₹{pnl:,.2f}")
    return {
        'total_trades': total,
        'winning_trades': int(wins),
        'losing_trades': int(losses),
        'breakeven_trades': int(breakeven),
        'win_rate': win_rate,
        'total_pnl': float(pnl),
    }


def prepare_consecutive_pairs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'Entry Time']).reset_index(drop=True)
    df['next_ticker'] = df['ticker'].shift(-1)
    df['next_trade_type'] = df['Trade Type'].shift(-1)
    df['next_entry_time'] = df['Entry Time'].shift(-1)
    df['next_exit_time'] = df['Exit Time'].shift(-1)
    df['pair_gap_minutes'] = (df['next_entry_time'] - df['Exit Time']).dt.total_seconds() / 60
    return df


def sample_pairs(df: pd.DataFrame, same_direction: bool, sample_size: int) -> pd.DataFrame:
    mask = (
        (df['ticker'] == df['next_ticker']) &
        (
            (df['Trade Type'] == df['next_trade_type']) if same_direction
            else (df['Trade Type'] != df['next_trade_type'])
        )
    )
    samples = df[mask].head(sample_size).copy()
    if samples.empty:
        return samples
    samples['current_summary'] = samples.apply(
        lambda row: f"{row['Trade Type']} {row['Entry Time'].strftime('%H:%M')}→{row['Exit Time'].strftime('%H:%M')} (₹{row['Profit (Currency)']:.0f})",
        axis=1,
    )
    samples['next_summary'] = samples.apply(
        lambda row: f"{row['next_trade_type']} {row['next_entry_time'].strftime('%H:%M')}→{row['next_exit_time'].strftime('%H:%M')}",
        axis=1,
    )
    samples['gap_minutes'] = samples['pair_gap_minutes'].round(1)
    return samples[['ticker', 'Entry Time', 'current_summary', 'next_summary', 'gap_minutes']]


def pick_sample_ticker(df: pd.DataFrame, preferred: Optional[str] = None) -> Optional[str]:
    if preferred and preferred in df['ticker'].unique():
        return preferred
    counts = df['ticker'].value_counts()
    return counts.idxmax() if not counts.empty else None


def sample_day_trades(df: pd.DataFrame, ticker: str, max_examples: int = 10) -> pd.DataFrame:
    ticker_df = df[df['ticker'] == ticker].copy()
    if ticker_df.empty:
        return ticker_df
    ticker_df['Entry Date'] = ticker_df['Entry Time'].dt.date
    top_day = (
        ticker_df.groupby('Entry Date')
        .size()
        .sort_values(ascending=False)
        .index[0]
    )
    day_trades = ticker_df[ticker_df['Entry Date'] == top_day].sort_values('Entry Time').head(max_examples)
    return day_trades[['Entry Time', 'Exit Time', 'Trade Type', 'Profit (Currency)']].copy()


def export_report(
    config: Dict[str, Any],
    stats: Dict[str, Any],
    missing_cols: List[str],
    warnings_list: List[str],
    same_dir_samples: pd.DataFrame,
    opp_dir_samples: pd.DataFrame,
    sample_trades: pd.DataFrame,
    sample_ticker: Optional[str],
) -> Path:
    report_path = Path(resolve_artifact_path(config, MODULE_NAME, 'validation_report', artifact_type='markdown'))
    lines: List[str] = [
        "# Validation Check Report",
        f"**Generated**: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"**Strategy**: {config['run']['strategy']}",
        f"**Run ID**: {config['run']['run_id']}",
        "",
        "## Overall Stats",
        f"- Total Trades: {stats['total_trades']:,}",
        f"- Win Rate: {stats['win_rate']:.2f}%",
        f"- Total P&L: ₹{stats['total_pnl']:,.2f}",
        f"- Winning Trades: {stats['winning_trades']:,}",
        f"- Losing Trades: {stats['losing_trades']:,}",
        f"- Breakeven Trades: {stats['breakeven_trades']:,}",
        "",
    ]

    if missing_cols:
        lines.append("## Missing Columns ⚠️")
        for col in missing_cols:
            lines.append(f"- `{col}`")
        lines.append("")

    if warnings_list:
        lines.append("## Warnings ⚠️")
        for warning in warnings_list:
            lines.append(f"- {warning}")
        lines.append("")

    lines.append("## Consecutive Trade Samples")
    if same_dir_samples.empty:
        lines.append("_No same-direction consecutive trades available in sample._")
    else:
        lines.append("### Same Direction")
        lines.append(same_dir_samples.to_markdown(index=False))
    lines.append("")

    if opp_dir_samples.empty:
        lines.append("_No opposite-direction consecutive trades available in sample._")
    else:
        lines.append("### Opposite Direction")
        lines.append(opp_dir_samples.to_markdown(index=False))
    lines.append("")

    if not sample_trades.empty and sample_ticker:
        lines.append(f"## Sample Day Trades for `{sample_ticker}`")
        sample_trades_fmt = sample_trades.copy()
        sample_trades_fmt['Entry Time'] = sample_trades_fmt['Entry Time'].dt.strftime("%Y-%m-%d %H:%M")
        sample_trades_fmt['Exit Time'] = sample_trades_fmt['Exit Time'].dt.strftime("%Y-%m-%d %H:%M")
        lines.append(sample_trades_fmt.to_markdown(index=False))
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(f"\n📝 Validation report saved → {report_path}")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validation check report generator")
    parser.add_argument("--config", required=True, help="Path to analysis YAML config")
    parser.add_argument("--sample", type=int, help="Optional sample size for quick runs")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = resolve_paths(config)
    module_cfg = get_analysis_config(config, MODULE_NAME) or {}

    sample_size = args.sample or module_cfg.get('sample_size')
    df = load_trades(config, paths, sample_size=sample_size)

    warnings_list: List[str] = []
    required_columns = module_cfg.get('required_columns', [])
    missing_cols = validate_columns(df, required_columns)

    stats = compute_overall_stats(df)
    if module_cfg.get('warn_on_zero_win_rate', True) and stats['win_rate'] == 0:
        warning = "Win rate is 0%; verify trade merges and filters."
        warnings_list.append(warning)
        print(f"⚠️  {warning}")

    pairs_df = prepare_consecutive_pairs(df)
    same_dir_samples = sample_pairs(pairs_df, same_direction=True, sample_size=module_cfg.get('sample_rows', 5))
    opp_dir_samples = sample_pairs(pairs_df, same_direction=False, sample_size=module_cfg.get('sample_rows', 5))

    sample_ticker = pick_sample_ticker(df, module_cfg.get('sample_ticker'))
    sample_day_df = sample_day_trades(df, sample_ticker) if sample_ticker else pd.DataFrame()

    export_report(
        config,
        stats,
        missing_cols,
        warnings_list,
        same_dir_samples,
        opp_dir_samples,
        sample_day_df,
        sample_ticker,
    )
    print("\n✅ Validation check complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
