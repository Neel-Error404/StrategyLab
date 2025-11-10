import pandas as pd
from pathlib import Path
import math

MERGED_DIR = Path('outputs/mse_exit_merged')
OUTPUT_CSV = MERGED_DIR / 'exit_scenario_summary.csv'

def compute_scenario_metrics():
    files = sorted(MERGED_DIR.glob('mse_exit_drop_*_merged_trades.csv'))
    if not files:
        print("[WARN] No merged files found. Run merge_exit_trades.py first.")
        return

    summary_rows = []
    for file in files:
        df = pd.read_csv(file)
        if df.empty:
            print(f"[WARN] {file} is empty; skipping.")
            continue

        scenario = df['scenario'].iloc[0]
        exit_threshold = df['exit_threshold'].iloc[0]
        drop_pct = df['drop_pct'].iloc[0]
        trade_count = len(df)

        win_rate = (df['Profit (Currency)'] > 0).mean() * 100 if trade_count else float('nan')
        avg_duration = df['Trade Duration (min)'].mean()
        avg_profit_currency = df['Profit (Currency)'].mean()
        avg_profit_pct = df['Profit (%)'].mean()

        returns = (df['Profit (%)'].dropna() / 100.0)
        if len(returns) > 1 and returns.std() != 0:
            sharpe = (returns.mean() / returns.std()) * math.sqrt(len(returns))
        else:
            sharpe = float('nan')

        summary_rows.append({
            'scenario': scenario,
            'drop_pct': drop_pct,
            'exit_threshold': exit_threshold,
            'trades': trade_count,
            'win_rate_pct': win_rate,
            'avg_duration_min': avg_duration,
            'avg_profit_currency': avg_profit_currency,
            'avg_profit_pct': avg_profit_pct,
            'sharpe_per_trade': sharpe,
            'source_file': str(file)
        })

    summary_df = pd.DataFrame(summary_rows).sort_values('drop_pct', ascending=False)
    summary_df.to_csv(OUTPUT_CSV, index=False)
    print(summary_df[['scenario','drop_pct','trades','win_rate_pct','avg_duration_min','avg_profit_currency','avg_profit_pct','sharpe_per_trade']]
          .to_string(index=False, float_format=lambda x: f"{x:0.2f}"))
    print(f"\nSummary written to {OUTPUT_CSV}")

if __name__ == '__main__':
    compute_scenario_metrics()
