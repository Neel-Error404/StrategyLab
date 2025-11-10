import pandas as pd
from pathlib import Path

SCENARIO_THRESHOLDS = {
    'mse_exit_drop_001': 0.99,
    'mse_exit_drop_005': 0.95,
    'mse_exit_drop_020': 0.80,
    'mse_exit_drop_055': 0.45,
    'mse_exit_drop_080': 0.20,
    'mse_exit_drop_095': 0.05,
}

def discover_latest_run(scenario_dir: Path) -> Path | None:
    run_dirs = [d for d in scenario_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda d: d.stat().st_mtime)

def find_date_dir(run_dir: Path) -> Path | None:
    strat_root = run_dir / 'mse_strategy_backtesting'
    if not strat_root.exists():
        return None
    date_dirs = [d for d in strat_root.iterdir() if d.is_dir()]
    if not date_dirs:
        return None
    return max(date_dirs, key=lambda d: d.stat().st_mtime)

def merge_trades():
    base_dir = Path('outputs')
    merged_root = base_dir / 'mse_exit_merged'
    merged_root.mkdir(parents=True, exist_ok=True)
    summary = []

    for scenario, threshold in SCENARIO_THRESHOLDS.items():
        scenario_dir = base_dir / scenario
        if not scenario_dir.exists():
            print(f"[WARN] {scenario} not found; skipping.")
            continue

        latest_run = discover_latest_run(scenario_dir)
        if latest_run is None:
            print(f"[WARN] {scenario} has no run directories; skipping.")
            continue

        date_dir = find_date_dir(latest_run)
        if date_dir is None:
            print(f"[WARN] {scenario} run {latest_run.name} missing date-range data; skipping.")
            continue

        trade_dir = date_dir / 'data' / 'strategy_trades'
        if not trade_dir.exists():
            print(f"[WARN] {scenario} missing strategy_trades at {trade_dir}; skipping.")
            continue

        csv_paths = sorted(trade_dir.glob('*_StrategyTrades_*.csv'))
        if not csv_paths:
            print(f"[WARN] {scenario} has no strategy trade CSVs in {trade_dir}; skipping.")
            continue

        frames = []
        for csv_path in csv_paths:
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                print(f"[WARN] Failed to read {csv_path}: {exc}")
                continue
            if df.empty:
                continue

            if 'Ticker' not in df.columns:
                df['Ticker'] = csv_path.stem.split('_')[0]
            if 'ticker' in df.columns and 'Ticker' not in df.columns:
                df.rename(columns={'ticker': 'Ticker'}, inplace=True)

            df['scenario'] = scenario
            df['exit_threshold'] = threshold
            df['drop_pct'] = round(1 - threshold, 4)
            df['date_range'] = date_dir.name
            frames.append(df)

        if not frames:
            print(f"[WARN] {scenario} produced no non-empty trade frames.")
            continue

        merged = pd.concat(frames, ignore_index=True)
        out_path = merged_root / f"{scenario}_merged_trades.csv"
        merged.to_csv(out_path, index=False)
        summary.append((scenario, len(merged), str(out_path)))
        print(f"[INFO] {scenario}: wrote {len(merged)} rows -> {out_path}")

    if summary:
        print('\nSummary:')
        for scenario, rows, out_path in summary:
            print(f"  {scenario}: {rows} trades | {out_path}")
    else:
        print("No merged trade files were created.")

if __name__ == '__main__':
    merge_trades()
