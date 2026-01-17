#!/usr/bin/env python3
"""Aggregate exit reasons across runs in an experiment folder."""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

EXIT_COLUMNS = ['Exit Reason', 'PnL', 'Profit (%)']


def load_run_info(run_dir: Path) -> Dict:
    run_info_file = run_dir / 'run_info.json'
    if not run_info_file.exists():
        raise FileNotFoundError(f"run_info.json missing in {run_dir}")
    with run_info_file.open() as f:
        data = json.load(f)
    data['strategy_run_dir'] = str(run_dir)
    return data


def _infer_ticker_from_filename(trade_file: Path) -> str:
    """Extract ticker symbol from flat file naming convention."""
    stem = trade_file.stem
    return stem.split('_')[0]


def collect_trade_files(run_dir: Path, trade_source: str) -> List[Tuple[Path, str]]:
    """
    Gather trade files for the provided run directory.

    Supports both the legacy structure (`tickers/<TICKER>/strategy_trades.csv`)
    and the new v2 layout (`data/strategy_trades/<TICKER>_StrategyTrades_*.csv`).
    """
    collected: List[Tuple[Path, str]] = []
    filename = 'strategy_trades.csv' if trade_source == 'strategy' else 'risk_approved_trades.csv'

    # Legacy directory structure
    tickers_dir = run_dir / 'tickers'
    if tickers_dir.exists():
        for ticker_dir in tickers_dir.iterdir():
            if not ticker_dir.is_dir():
                continue
            trade_file = ticker_dir / filename
            if trade_file.exists():
                collected.append((trade_file, ticker_dir.name))

    # New directory structure under data/<category>/
    data_dir = run_dir / 'data'
    if data_dir.exists():
        category = 'strategy_trades' if trade_source == 'strategy' else 'risk_approved_trades'
        flat_dir = data_dir / category
        if flat_dir.exists():
            for trade_file in sorted(flat_dir.glob('*.csv')):
                collected.append((trade_file, _infer_ticker_from_filename(trade_file)))

    return collected


def summarize_exit_reason(rows: List[Dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=['run_label', 'exit_reason', 'trades', 'avg_pnl', 'avg_profit_pct'])
    df = pd.DataFrame(rows)
    grouped = df.groupby(['run_label', 'exit_reason']).agg(
        trades=('PnL', 'count'),
        avg_pnl=('PnL', 'mean'),
        avg_profit_pct=('Profit (%)', 'mean')
    ).reset_index()
    return grouped


def analyze_experiment(experiment_dir: Path, trade_source: str) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(experiment_dir.rglob('run_info.json')):
        run_dir = run_dir.parent
        run_info = load_run_info(run_dir)
        run_label = run_info.get('run_label') or run_info.get('timestamp')
        exit_template = run_info.get('exit_template')
        trade_files = collect_trade_files(Path(run_dir), trade_source)
        if not trade_files:
            continue
        for trade_file, ticker in trade_files:
            df = pd.read_csv(trade_file)
            if 'Exit Reason' not in df.columns:
                continue
            df = df.copy()
            df['run_label'] = run_label
            df['exit_template'] = exit_template
            df['ticker'] = ticker
            rows.extend(df[['run_label', 'ticker', 'exit_template', 'Exit Reason', 'PnL', 'Profit (%)']]
                        .rename(columns={'Exit Reason': 'exit_reason'}).to_dict('records'))
    if not rows:
        return pd.DataFrame()
    summary = summarize_exit_reason(rows)
    summary['experiment_dir'] = str(experiment_dir)
    summary = summary.sort_values(['run_label', 'exit_reason']).reset_index(drop=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description='Summarize exit reasons across experiment runs')
    parser.add_argument('experiment_dir', type=str, help='Path to outputs/experiment folder')
    parser.add_argument('--output', type=str, help='Optional output CSV path')
    parser.add_argument('--trade-source', choices=['strategy', 'risk'], default='strategy',
                        help='Use strategy_trades (default) or risk_approved_trades')
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    trade_source = args.trade_source
    summary = analyze_experiment(exp_dir, trade_source)
    if summary.empty:
        print('No trade data found for exit reason analysis.')
        return

    print(summary)

    suffix = 'strategy' if trade_source == 'strategy' else 'risk'
    default_path = exp_dir / 'analysis_reports' / f'exit_reason_summary_{suffix}.csv'
    output_path = Path(args.output) if args.output else default_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Saved exit reason summary to {output_path}")

if __name__ == '__main__':
    main()
