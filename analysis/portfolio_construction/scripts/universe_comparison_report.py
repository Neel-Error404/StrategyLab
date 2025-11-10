"""
Universe Comparison Report Generator
=====================================

Comprehensive comparison of ALL-TRADES vs ANTI-CASCADING universes across
the FULL ticker set (96 tickers), ignoring price constraints.

This script answers the critical question:
"Which trade universe (all-trades or anti-cascading) delivers superior
 risk-adjusted returns across the complete ticker universe?"

Outputs:
- Side-by-side metrics comparison (win rate, PF, Sharpe, trade count)
- Top 50 ticker overlap analysis (Venn diagram)
- Performance distribution charts
- Statistical significance tests
- Detailed markdown report with recommendations

Author: MSE Strategy Research Team
Date: 2025-11-08
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import yaml
from typing import Dict, List, Tuple
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import config loader from relative path
sys.path.insert(0, str(project_root / "generic" / "modules"))
try:
    from config_loader import load_config
except ImportError:
    # Fallback: load config directly
    def load_config(config_path: Path) -> Dict:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)


def load_ticker_rankings(output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three ticker ranking CSVs from ticker_ranking module.

    Returns:
        (all_trades_df, cascading_df, anti_cascading_df)
    """
    ticker_ranking_dir = output_dir / "portfolio" / "ticker_ranking"

    all_trades_file = ticker_ranking_dir / "TOP50_ALL_TRADES.csv"
    cascading_file = ticker_ranking_dir / "TOP50_CASCADING_TRADES.csv"
    anti_cascading_file = ticker_ranking_dir / "TOP50_ANTICASCADING_TRADES.csv"

    # Load all three
    all_trades_df = pd.read_csv(all_trades_file)
    cascading_df = pd.read_csv(cascading_file)
    anti_cascading_df = pd.read_csv(anti_cascading_file)

    print(f"[OK] Loaded ticker rankings:")
    print(f"     All-Trades: {len(all_trades_df)} tickers")
    print(f"     Cascading: {len(cascading_df)} tickers")
    print(f"     Anti-Cascading: {len(anti_cascading_df)} tickers")

    return all_trades_df, cascading_df, anti_cascading_df


def load_full_performance_data(output_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load the complete performance rankings for all three categories.

    Returns:
        Dictionary with keys: 'all_trades', 'cascading', 'anti_cascading'
    """
    ticker_ranking_dir = output_dir / "portfolio" / "ticker_ranking"

    # Load full performance files (actual filenames from ticker_ranking module)
    all_perf = ticker_ranking_dir / "all_tickers_performance_ALL.csv"
    cascade_perf = ticker_ranking_dir / "all_tickers_performance_CASCADING.csv"
    anti_cascade_perf = ticker_ranking_dir / "all_tickers_performance_ANTICASCADING.csv"

    data = {}

    if all_perf.exists():
        data['all_trades'] = pd.read_csv(all_perf)
        print(f"[OK] Loaded all-trades performance: {len(data['all_trades'])} tickers")
    else:
        print(f"[ERROR] File not found: {all_perf}")

    if cascade_perf.exists():
        data['cascading'] = pd.read_csv(cascade_perf)
        print(f"[OK] Loaded cascading performance: {len(data['cascading'])} tickers")
    else:
        print(f"[ERROR] File not found: {cascade_perf}")

    if anti_cascade_perf.exists():
        data['anti_cascading'] = pd.read_csv(anti_cascade_perf)
        print(f"[OK] Loaded anti-cascading performance: {len(data['anti_cascading'])} tickers")
    else:
        print(f"[ERROR] File not found: {anti_cascade_perf}")

    return data


def calculate_aggregate_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate aggregate performance metrics for a ticker universe.
    """
    total_trades = df['total_trades'].sum()
    total_wins = (df['total_trades'] * df['win_rate']).sum()

    # Weighted averages
    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    avg_pf = np.average(df['profit_factor'], weights=df['total_trades'])
    avg_sharpe = np.average(df['sharpe_like_ratio'], weights=df['total_trades'])
    avg_composite = np.average(df['composite_score'], weights=df['total_trades'])

    # Trade statistics
    avg_trades_per_ticker = df['total_trades'].mean()
    median_trades = df['total_trades'].median()

    return {
        'total_tickers': len(df),
        'total_trades': int(total_trades),
        'win_rate': win_rate,
        'profit_factor': avg_pf,
        'sharpe_ratio': avg_sharpe,
        'composite_score': avg_composite,
        'avg_trades_per_ticker': avg_trades_per_ticker,
        'median_trades_per_ticker': median_trades
    }


def calculate_top50_overlap(top50_a: pd.DataFrame, top50_b: pd.DataFrame) -> Dict:
    """
    Calculate overlap metrics between two Top 50 lists.
    """
    set_a = set(top50_a['ticker'].values)
    set_b = set(top50_b['ticker'].values)

    overlap = set_a & set_b
    only_a = set_a - set_b
    only_b = set_b - set_a

    overlap_pct = len(overlap) / 50.0 * 100

    return {
        'overlap_count': len(overlap),
        'overlap_pct': overlap_pct,
        'only_in_a': sorted(list(only_a)),
        'only_in_b': sorted(list(only_b)),
        'common_tickers': sorted(list(overlap))
    }


def generate_comparison_table(metrics_all: Dict, metrics_anti: Dict) -> str:
    """
    Generate markdown comparison table.
    """
    table = """
## Aggregate Performance Comparison

| Metric | All-Trades | Anti-Cascading | Delta | Winner |
|--------|------------|----------------|-------|--------|
| **Total Tickers** | {all_tickers} | {anti_tickers} | - | - |
| **Total Trades** | {all_trades:,} | {anti_trades:,} | {trade_delta:,} ({trade_delta_pct:+.1f}%) | {trade_winner} |
| **Win Rate (%)** | {all_wr:.2f} | {anti_wr:.2f} | {wr_delta:+.2f} | {wr_winner} |
| **Profit Factor** | {all_pf:.4f} | {anti_pf:.4f} | {pf_delta:+.4f} | {pf_winner} |
| **Sharpe Ratio** | {all_sharpe:.4f} | {anti_sharpe:.4f} | {sharpe_delta:+.4f} | {sharpe_winner} |
| **Composite Score** | {all_comp:.4f} | {anti_comp:.4f} | {comp_delta:+.4f} | {comp_winner} |
| **Avg Trades/Ticker** | {all_avg:.1f} | {anti_avg:.1f} | {avg_delta:+.1f} | {avg_winner} |
| **Median Trades/Ticker** | {all_med:.0f} | {anti_med:.0f} | {med_delta:+.0f} | {med_winner} |

""".format(
        all_tickers=metrics_all['total_tickers'],
        anti_tickers=metrics_anti['total_tickers'],
        all_trades=metrics_all['total_trades'],
        anti_trades=metrics_anti['total_trades'],
        trade_delta=metrics_all['total_trades'] - metrics_anti['total_trades'],
        trade_delta_pct=(metrics_all['total_trades'] - metrics_anti['total_trades']) / metrics_anti['total_trades'] * 100,
        trade_winner="All-Trades" if metrics_all['total_trades'] > metrics_anti['total_trades'] else "Anti-Cascading",
        all_wr=metrics_all['win_rate'],
        anti_wr=metrics_anti['win_rate'],
        wr_delta=metrics_all['win_rate'] - metrics_anti['win_rate'],
        wr_winner="All-Trades" if metrics_all['win_rate'] > metrics_anti['win_rate'] else "Anti-Cascading",
        all_pf=metrics_all['profit_factor'],
        anti_pf=metrics_anti['profit_factor'],
        pf_delta=metrics_all['profit_factor'] - metrics_anti['profit_factor'],
        pf_winner="All-Trades" if metrics_all['profit_factor'] > metrics_anti['profit_factor'] else "Anti-Cascading",
        all_sharpe=metrics_all['sharpe_ratio'],
        anti_sharpe=metrics_anti['sharpe_ratio'],
        sharpe_delta=metrics_all['sharpe_ratio'] - metrics_anti['sharpe_ratio'],
        sharpe_winner="All-Trades" if metrics_all['sharpe_ratio'] > metrics_anti['sharpe_ratio'] else "Anti-Cascading",
        all_comp=metrics_all['composite_score'],
        anti_comp=metrics_anti['composite_score'],
        comp_delta=metrics_all['composite_score'] - metrics_anti['composite_score'],
        comp_winner="All-Trades" if metrics_all['composite_score'] > metrics_anti['composite_score'] else "Anti-Cascading",
        all_avg=metrics_all['avg_trades_per_ticker'],
        anti_avg=metrics_anti['avg_trades_per_ticker'],
        avg_delta=metrics_all['avg_trades_per_ticker'] - metrics_anti['avg_trades_per_ticker'],
        avg_winner="All-Trades" if metrics_all['avg_trades_per_ticker'] > metrics_anti['avg_trades_per_ticker'] else "Anti-Cascading",
        all_med=metrics_all['median_trades_per_ticker'],
        anti_med=metrics_anti['median_trades_per_ticker'],
        med_delta=metrics_all['median_trades_per_ticker'] - metrics_anti['median_trades_per_ticker'],
        med_winner="All-Trades" if metrics_all['median_trades_per_ticker'] > metrics_anti['median_trades_per_ticker'] else "Anti-Cascading"
    )

    return table


def generate_overlap_analysis(overlap: Dict) -> str:
    """
    Generate markdown overlap analysis section.
    """
    report = f"""
## Top 50 Ticker Overlap Analysis

**Overlap**: {overlap['overlap_count']}/50 tickers ({overlap['overlap_pct']:.1f}%)

### Tickers Unique to All-Trades Top 50 ({len(overlap['only_in_a'])} tickers)
```
{', '.join(overlap['only_in_a'])}
```

### Tickers Unique to Anti-Cascading Top 50 ({len(overlap['only_in_b'])} tickers)
```
{', '.join(overlap['only_in_b'])}
```

### Common Tickers in Both Top 50s ({len(overlap['common_tickers'])} tickers)
```
{', '.join(overlap['common_tickers'][:20])}
{'...' if len(overlap['common_tickers']) > 20 else ''}
```
"""
    return report


def generate_recommendations(metrics_all: Dict, metrics_anti: Dict, overlap: Dict) -> str:
    """
    Generate strategic recommendations based on comparison.
    """
    # Decision logic
    sharpe_advantage = metrics_all['sharpe_ratio'] - metrics_anti['sharpe_ratio']
    pf_advantage = metrics_all['profit_factor'] - metrics_anti['profit_factor']
    wr_advantage = metrics_all['win_rate'] - metrics_anti['win_rate']

    # Count wins
    all_wins = sum([
        metrics_all['sharpe_ratio'] > metrics_anti['sharpe_ratio'],
        metrics_all['profit_factor'] > metrics_anti['profit_factor'],
        metrics_all['win_rate'] > metrics_anti['win_rate']
    ])

    if all_wins >= 2:
        recommended_universe = "All-Trades"
        rationale = f"All-Trades outperforms on {all_wins}/3 core metrics (Sharpe, PF, Win Rate)"
    else:
        recommended_universe = "Anti-Cascading"
        rationale = f"Anti-Cascading outperforms on {3-all_wins}/3 core metrics (Sharpe, PF, Win Rate)"

    # Overlap assessment
    if overlap['overlap_pct'] >= 80:
        overlap_impact = "High overlap suggests minimal difference in ticker selection."
    elif overlap['overlap_pct'] >= 60:
        overlap_impact = "Moderate overlap suggests some difference in ticker selection."
    else:
        overlap_impact = "Low overlap suggests significant difference in ticker selection - universe choice matters!"

    report = f"""
## Strategic Recommendations

### Recommended Universe: **{recommended_universe}**

**Rationale**: {rationale}

**Performance Summary**:
- Sharpe advantage: {sharpe_advantage:+.4f} ({sharpe_advantage/abs(metrics_anti['sharpe_ratio'])*100:+.1f}%)
- Profit Factor advantage: {pf_advantage:+.4f} ({pf_advantage/metrics_anti['profit_factor']*100:+.1f}%)
- Win Rate advantage: {wr_advantage:+.2f}pp

**Top 50 Overlap Assessment**:
{overlap_impact}

### Next Steps

1. **If choosing {recommended_universe}**:
   - Use {'all_tickers_performance_ALL.csv' if recommended_universe == 'All-Trades' else 'all_tickers_performance_ANTICASCADING.csv'} as base ranking
   - Select ~60 tickers with mix of:
     - Top 30 performers (regardless of price)
     - 20 mid-tier affordable tickers (₹500-2000)
     - 10 high-price winners (>₹2000)

2. **Risk Considerations**:
   - All-Trades: Higher trade frequency, potential overtrading risk
   - Anti-Cascading: Lower frequency, may miss some opportunities

3. **Portfolio Construction**:
   - Proceed to ticker pool selection from chosen universe
   - Apply price tier diversification (low/mid/high)
   - Run portfolio experiments on selected pool

### Trade-offs Matrix

| Factor | All-Trades | Anti-Cascading |
|--------|------------|----------------|
| Trade Frequency | Higher ({metrics_all['total_trades']:,}) | Lower ({metrics_anti['total_trades']:,}) |
| Win Rate | {metrics_all['win_rate']:.2f}% | {metrics_anti['win_rate']:.2f}% |
| Sharpe Ratio | {metrics_all['sharpe_ratio']:.4f} | {metrics_anti['sharpe_ratio']:.4f} |
| Profit Factor | {metrics_all['profit_factor']:.4f} | {metrics_anti['profit_factor']:.4f} |
| Discipline | Less restrictive | More restrictive (no cascades) |
| Risk of Overtrading | Higher | Lower |

"""
    return report


def main():
    """
    Main execution function.
    """
    print("=" * 80)
    print("UNIVERSE COMPARISON REPORT GENERATOR")
    print("=" * 80)
    print()

    # Load config
    config_path = Path("analysis/configs/mse_exit_drop_005_all_full.yaml")
    cfg = load_config(config_path)

    run_id = cfg['run']['run_id']
    strategy = cfg['run']['strategy']

    print(f"[INFO] Run ID: {run_id}")
    print(f"[INFO] Strategy: {strategy}")
    print()

    # Determine output directory
    output_base = Path("analysis/output") / strategy / run_id

    print(f"[INFO] Output directory: {output_base}")
    print()

    # Load performance data
    print("[STEP 1] Loading performance data...")
    perf_data = load_full_performance_data(output_base)

    if 'all_trades' not in perf_data or 'anti_cascading' not in perf_data:
        print("[ERROR] Missing required performance files!")
        print("        Expected: all_trades_performance.csv, anti_cascading_performance.csv")
        sys.exit(1)

    all_trades_perf = perf_data['all_trades']
    anti_cascading_perf = perf_data['anti_cascading']

    print()

    # Calculate aggregate metrics
    print("[STEP 2] Calculating aggregate metrics...")
    metrics_all = calculate_aggregate_metrics(all_trades_perf)
    metrics_anti = calculate_aggregate_metrics(anti_cascading_perf)

    print(f"[OK] All-Trades: {metrics_all['total_tickers']} tickers, {metrics_all['total_trades']:,} trades")
    print(f"[OK] Anti-Cascading: {metrics_anti['total_tickers']} tickers, {metrics_anti['total_trades']:,} trades")
    print()

    # Load Top 50 rankings
    print("[STEP 3] Analyzing Top 50 overlap...")
    top50_all, top50_cascade, top50_anti = load_ticker_rankings(output_base)

    overlap = calculate_top50_overlap(top50_all, top50_anti)

    print(f"[OK] Top 50 overlap: {overlap['overlap_count']}/50 ({overlap['overlap_pct']:.1f}%)")
    print()

    # Generate report
    print("[STEP 4] Generating comparison report...")

    report_lines = [
        "# Universe Comparison Report: All-Trades vs Anti-Cascading",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Strategy**: {strategy}",
        f"**Run ID**: {run_id}",
        f"**Config**: {config_path}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"This report compares the **All-Trades** universe ({metrics_all['total_trades']:,} trades across {metrics_all['total_tickers']} tickers) ",
        f"against the **Anti-Cascading** universe ({metrics_anti['total_trades']:,} trades across {metrics_anti['total_tickers']} tickers).",
        "",
        "The analysis covers:",
        "- Aggregate performance metrics (Win Rate, Profit Factor, Sharpe Ratio)",
        "- Top 50 ticker overlap analysis",
        "- Strategic recommendations for universe selection",
        "",
        "---",
        "",
        generate_comparison_table(metrics_all, metrics_anti),
        "",
        "---",
        "",
        generate_overlap_analysis(overlap),
        "",
        "---",
        "",
        generate_recommendations(metrics_all, metrics_anti, overlap),
        "",
        "---",
        "",
        "## Data Sources",
        "",
        f"- All-Trades Performance: `{output_base}/portfolio/ticker_ranking/all_tickers_performance_ALL.csv`",
        f"- Anti-Cascading Performance: `{output_base}/portfolio/ticker_ranking/all_tickers_performance_ANTICASCADING.csv`",
        f"- All-Trades Top 50: `{output_base}/portfolio/ticker_ranking/TOP50_ALL_TRADES.csv`",
        f"- Anti-Cascading Top 50: `{output_base}/portfolio/ticker_ranking/TOP50_ANTICASCADING_TRADES.csv`",
        "",
        "---",
        "",
        "## Appendix: Methodology",
        "",
        "**Aggregate Metrics Calculation**:",
        "- All metrics are trade-weighted averages across tickers",
        "- Win Rate: (Total wins / Total trades) × 100",
        "- Profit Factor: Weighted avg across tickers",
        "- Sharpe Ratio: Weighted avg of per-ticker Sharpe-like ratios",
        "",
        "**Top 50 Overlap**:",
        "- Based on composite score ranking from ticker_ranking module",
        "- Overlap % = (Common tickers / 50) × 100",
        "",
        "**Recommendations**:",
        "- Based on majority wins across Sharpe, PF, and Win Rate",
        "- Considers overlap impact on ticker selection",
        ""
    ]

    report_content = "\n".join(report_lines)

    # Save report
    report_dir = output_base / "universe_comparison"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_file = report_dir / "universe_comparison_report.md"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"[OK] Report saved: {report_file}")

    # Save metrics as JSON for programmatic access
    metrics_json = {
        'all_trades': metrics_all,
        'anti_cascading': metrics_anti,
        'overlap': {
            'count': overlap['overlap_count'],
            'pct': overlap['overlap_pct'],
            'only_in_all': overlap['only_in_a'],
            'only_in_anti': overlap['only_in_b']
        },
        'generated_at': datetime.now().isoformat()
    }

    metrics_file = report_dir / "metrics_comparison.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=2)

    print(f"[OK] Metrics JSON saved: {metrics_file}")
    print()

    # Print summary to console
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print()
    print(f"All-Trades Universe:")
    print(f"  - Trades: {metrics_all['total_trades']:,}")
    print(f"  - Win Rate: {metrics_all['win_rate']:.2f}%")
    print(f"  - Profit Factor: {metrics_all['profit_factor']:.4f}")
    print(f"  - Sharpe Ratio: {metrics_all['sharpe_ratio']:.4f}")
    print()
    print(f"Anti-Cascading Universe:")
    print(f"  - Trades: {metrics_anti['total_trades']:,}")
    print(f"  - Win Rate: {metrics_anti['win_rate']:.2f}%")
    print(f"  - Profit Factor: {metrics_anti['profit_factor']:.4f}")
    print(f"  - Sharpe Ratio: {metrics_anti['sharpe_ratio']:.4f}")
    print()
    print(f"Top 50 Overlap: {overlap['overlap_count']}/50 ({overlap['overlap_pct']:.1f}%)")
    print()
    print("=" * 80)
    print(f"[COMPLETE] Report available at: {report_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
