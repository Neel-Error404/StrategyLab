"""
Ticker Pool Selector - Top 60 Performers
=========================================

Selects the top 60 performing tickers from the chosen universe (Anti-Cascading)
based purely on composite score ranking.

Selection Criteria:
- Ranked by composite score (descending)
- Top 60 performers selected
- Price preference: ≤₹1,500 (informational only, NOT a filter)
- No price-based bifurcation - pure performance ranking

Outputs:
- selected_ticker_pool_60.csv (60 tickers with full metadata)
- ticker_pool_analysis.md (price distribution, performance stats)

Author: MSE Strategy Research Team
Date: 2025-11-08
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml
from typing import Dict, List
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_config(config_path: Path) -> Dict:
    """Load YAML config file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_performance_data(performance_file: Path) -> pd.DataFrame:
    """
    Load ticker performance data from Anti-Cascading universe.
    """
    if not performance_file.exists():
        raise FileNotFoundError(f"Performance file not found: {performance_file}")

    df = pd.read_csv(performance_file)

    print(f"[OK] Loaded performance data: {len(df)} tickers")
    print(f"     Columns: {', '.join(df.columns.tolist())}")

    return df


def get_last_prices(output_base: Path, config: Dict) -> Dict[str, float]:
    """
    Extract last traded exit prices from the merged trades dataset.
    Uses the exit price from the most recent trade for each ticker.
    """
    # First try metadata file (has 26 tickers)
    metadata_file = output_base / "portfolio" / "anti_cascade_filter" / "affordable_tickers_metadata.csv"
    price_dict = {}

    if metadata_file.exists():
        metadata_df = pd.read_csv(metadata_file)
        for _, row in metadata_df.iterrows():
            ticker = row['ticker']
            price = row['current_price']
            price_dict[ticker] = price
        print(f"[OK] Loaded {len(price_dict)} prices from metadata file")

    # Now get prices for remaining tickers from actual trade data
    run_id = config['run']['run_id']
    strategy = config['run']['strategy']
    merged_file = Path(f"analysis/output/{strategy}/{run_id}/data/all_trades_merged.csv")

    if not merged_file.exists():
        print(f"[WARN] Merged trades file not found: {merged_file}")
        print(f"[WARN] Only {len(price_dict)} tickers will have price data")
        return price_dict

    print(f"[INFO] Loading merged trades to extract last exit prices...")
    trades_df = pd.read_csv(merged_file)

    # Get last trade for each ticker (sorted by Exit Time)
    trades_df['Exit Time'] = pd.to_datetime(trades_df['Exit Time'])
    last_trades = trades_df.sort_values('Exit Time').groupby('ticker').last().reset_index()

    # Extract exit prices for tickers not in metadata
    new_prices = 0
    for _, row in last_trades.iterrows():
        ticker = row['ticker']
        if ticker not in price_dict:  # Only add if not already present
            exit_price = row['Exit Price']
            if pd.notna(exit_price) and exit_price > 0:
                price_dict[ticker] = exit_price
                new_prices += 1

    print(f"[OK] Extracted {new_prices} additional prices from trade data")
    print(f"[OK] Total price data available: {len(price_dict)} tickers")

    return price_dict


def categorize_price(price: float) -> str:
    """Categorize ticker by price tier."""
    if pd.isna(price):
        return "Unknown"
    elif price <= 500:
        return "Under ₹500"
    elif price <= 1000:
        return "₹500-1000"
    elif price <= 1500:
        return "₹1000-1500"
    elif price <= 2000:
        return "₹1500-2000"
    else:
        return "Over ₹2000"


def select_top_60(performance_df: pd.DataFrame, price_dict: Dict[str, float], pool_size: int = 60) -> pd.DataFrame:
    """
    Select top N performers based on composite score ranking.

    Pure performance-based selection - NO price filtering.
    """
    # Sort by composite_score descending (already has rank column, but recalculate to be safe)
    sorted_df = performance_df.sort_values('composite_score', ascending=False).reset_index(drop=True)

    # Take top N
    top_n = sorted_df.head(pool_size).copy()

    # Add price data
    top_n['current_price'] = top_n['ticker'].map(price_dict)
    top_n['price_category'] = top_n['current_price'].apply(categorize_price)

    # Add selection rank (1-60)
    top_n['selection_rank'] = range(1, len(top_n) + 1)

    print(f"[OK] Selected top {pool_size} performers")

    return top_n


def analyze_pool(pool_df: pd.DataFrame, price_preference: float = 1500) -> Dict:
    """
    Analyze the selected ticker pool.
    """
    total_tickers = len(pool_df)

    # Price distribution
    has_price = pool_df['current_price'].notna().sum()
    no_price = total_tickers - has_price

    if has_price > 0:
        within_preference = (pool_df['current_price'] <= price_preference).sum()
        above_preference = has_price - within_preference
        pct_within = within_preference / total_tickers * 100

        avg_price = pool_df['current_price'].mean()
        median_price = pool_df['current_price'].median()
        min_price = pool_df['current_price'].min()
        max_price = pool_df['current_price'].max()
    else:
        within_preference = 0
        above_preference = 0
        pct_within = 0
        avg_price = median_price = min_price = max_price = 0

    # Performance stats
    avg_composite = pool_df['composite_score'].mean()
    avg_sharpe = pool_df['sharpe_like_ratio'].mean()
    avg_pf = pool_df['profit_factor'].mean()
    avg_wr = pool_df['win_rate'].mean()

    total_trades = pool_df['total_trades'].sum()
    avg_trades_per_ticker = pool_df['total_trades'].mean()

    # Price category breakdown
    price_breakdown = pool_df['price_category'].value_counts().to_dict()

    return {
        'total_tickers': total_tickers,
        'has_price_data': has_price,
        'no_price_data': no_price,
        'within_preference': within_preference,
        'above_preference': above_preference,
        'pct_within_preference': pct_within,
        'avg_price': avg_price,
        'median_price': median_price,
        'min_price': min_price,
        'max_price': max_price,
        'avg_composite_score': avg_composite,
        'avg_sharpe': avg_sharpe,
        'avg_profit_factor': avg_pf,
        'avg_win_rate': avg_wr,
        'total_trades': total_trades,
        'avg_trades_per_ticker': avg_trades_per_ticker,
        'price_breakdown': price_breakdown
    }


def generate_analysis_report(pool_df: pd.DataFrame, analysis: Dict, price_preference: float) -> str:
    """
    Generate markdown analysis report.
    """
    lines = [
        "# Ticker Pool Selection Analysis",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Pool Size**: {analysis['total_tickers']} tickers",
        f"**Selection Method**: Top performers by composite score (Anti-Cascading universe)",
        f"**Price Preference**: ≤₹{price_preference:,.0f} (informational only, NOT a filter)",
        "",
        "---",
        "",
        "## Selection Summary",
        "",
        f"Selected the **top {analysis['total_tickers']} performing tickers** based on composite score ranking.",
        "Pure performance-based selection with NO price filtering or bifurcation.",
        "",
        "## Price Distribution Analysis",
        "",
        f"**Tickers with price data**: {analysis['has_price_data']}/{analysis['total_tickers']}",
        f"**Tickers ≤₹{price_preference:,.0f}**: {analysis['within_preference']} ({analysis['pct_within_preference']:.1f}%)",
        f"**Tickers >₹{price_preference:,.0f}**: {analysis['above_preference']} ({100-analysis['pct_within_preference']:.1f}%)",
        "",
        "### Price Statistics",
        "",
        f"- **Average Price**: ₹{analysis['avg_price']:,.2f}",
        f"- **Median Price**: ₹{analysis['median_price']:,.2f}",
        f"- **Min Price**: ₹{analysis['min_price']:,.2f}",
        f"- **Max Price**: ₹{analysis['max_price']:,.2f}",
        "",
        "### Price Category Breakdown",
        "",
    ]

    # Add price breakdown table
    lines.append("| Price Category | Count | Percentage |")
    lines.append("|----------------|-------|------------|")

    for category in ["Under ₹500", "₹500-1000", "₹1000-1500", "₹1500-2000", "Over ₹2000", "Unknown"]:
        count = analysis['price_breakdown'].get(category, 0)
        pct = count / analysis['total_tickers'] * 100
        lines.append(f"| {category} | {count} | {pct:.1f}% |")

    lines.extend([
        "",
        "---",
        "",
        "## Performance Statistics",
        "",
        f"**Total Trades**: {analysis['total_trades']:,}",
        f"**Avg Trades/Ticker**: {analysis['avg_trades_per_ticker']:.0f}",
        "",
        "### Aggregate Metrics (Trade-Weighted)",
        "",
        f"- **Win Rate**: {analysis['avg_win_rate']:.2f}%",
        f"- **Profit Factor**: {analysis['avg_profit_factor']:.4f}",
        f"- **Sharpe Ratio**: {analysis['avg_sharpe']:.4f}",
        f"- **Composite Score**: {analysis['avg_composite_score']:.4f}",
        "",
        "---",
        "",
        "## Top 10 Tickers in Pool",
        "",
        "| Rank | Ticker | Price | Category | Win Rate | PF | Sharpe | Composite |",
        "|------|--------|-------|----------|----------|----|----|-----------|"
    ])

    # Add top 10
    top_10 = pool_df.head(10)
    for _, row in top_10.iterrows():
        price_str = f"₹{row['current_price']:,.2f}" if pd.notna(row['current_price']) else "N/A"
        category = row['price_category']
        lines.append(
            f"| {int(row['selection_rank'])} | {row['ticker']} | {price_str} | {category} | "
            f"{row['win_rate']:.2f}% | {row['profit_factor']:.4f} | {row['sharpe_like_ratio']:.4f} | "
            f"{row['composite_score']:.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Price Preference Assessment",
        "",
        f"**Target**: ₹{price_preference:,.0f} or less for substantial capital allocation",
        f"**Actual**: {analysis['within_preference']}/{analysis['total_tickers']} tickers ({analysis['pct_within_preference']:.1f}%) meet preference",
        "",
    ])

    if analysis['pct_within_preference'] >= 80:
        lines.append(f"✅ **EXCELLENT**: {analysis['pct_within_preference']:.1f}% of top performers are within price preference!")
    elif analysis['pct_within_preference'] >= 60:
        lines.append(f"✅ **GOOD**: {analysis['pct_within_preference']:.1f}% within preference. Acceptable capital allocation feasibility.")
    elif analysis['pct_within_preference'] >= 40:
        lines.append(f"⚠️ **MODERATE**: Only {analysis['pct_within_preference']:.1f}% within preference. May need capital adjustments.")
    else:
        lines.append(f"❌ **LOW**: Only {analysis['pct_within_preference']:.1f}% within preference. Consider capital constraints.")

    lines.extend([
        "",
        "### Recommendation",
        "",
        "This is a **pure performance-based pool**. If capital constraints require stricter price limits:",
        "1. Review tickers >₹1,500 for individual merit",
        "2. Consider weighted allocation (more capital to affordable, less to expensive)",
        "3. Or proceed with all 60 and adjust portfolio sizing in experiments",
        "",
        "---",
        "",
        "## Next Steps",
        "",
        "1. Review `selected_ticker_pool_60.csv` for complete ticker list",
        "2. Proceed to portfolio construction with this pool",
        "3. Run portfolio experiments (size, sector, correlation sweeps)",
        ""
    ])

    return "\n".join(lines)


def main():
    """
    Main execution function.
    """
    print("=" * 80)
    print("TICKER POOL SELECTOR - TOP 60 PERFORMERS")
    print("=" * 80)
    print()

    # Configuration
    config_path = Path("analysis/configs/mse_exit_drop_005_all_full.yaml")
    cfg = load_config(config_path)

    run_id = cfg['run']['run_id']
    strategy = cfg['run']['strategy']

    output_base = Path("analysis/output") / strategy / run_id

    print(f"[INFO] Run ID: {run_id}")
    print(f"[INFO] Strategy: {strategy}")
    print(f"[INFO] Output directory: {output_base}")
    print()

    # Load Anti-Cascading performance data
    print("[STEP 1] Loading Anti-Cascading performance data...")
    perf_file = output_base / "portfolio" / "ticker_ranking" / "all_tickers_performance_ANTICASCADING.csv"
    performance_df = load_performance_data(perf_file)
    print()

    # Load price data
    print("[STEP 2] Loading price data...")
    price_dict = get_last_prices(output_base, cfg)
    print()

    # Select top 60
    print("[STEP 3] Selecting top 60 performers...")
    pool_size = 60
    price_preference = 1500

    pool_df = select_top_60(performance_df, price_dict, pool_size)
    print()

    # Analyze pool
    print("[STEP 4] Analyzing selected pool...")
    analysis = analyze_pool(pool_df, price_preference)

    print(f"[OK] Pool analysis complete:")
    print(f"     {analysis['within_preference']}/{analysis['total_tickers']} tickers ≤₹{price_preference:,.0f} ({analysis['pct_within_preference']:.1f}%)")
    print(f"     Avg Price: ₹{analysis['avg_price']:,.2f}")
    print(f"     Avg Composite Score: {analysis['avg_composite_score']:.4f}")
    print()

    # Generate report
    print("[STEP 5] Generating analysis report...")
    report = generate_analysis_report(pool_df, analysis, price_preference)

    # Save outputs
    output_dir = output_base / "ticker_pool_selection"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save pool CSV
    pool_file = output_dir / "selected_ticker_pool_60.csv"
    pool_df.to_csv(pool_file, index=False)
    print(f"[OK] Saved ticker pool: {pool_file}")

    # Save analysis report
    report_file = output_dir / "ticker_pool_analysis.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"[OK] Saved analysis report: {report_file}")
    print()

    # Summary
    print("=" * 80)
    print("SELECTION COMPLETE")
    print("=" * 80)
    print()
    print(f"Selected: {pool_size} top performers from Anti-Cascading universe")
    print(f"Price preference (₹1,500): {analysis['within_preference']}/{pool_size} tickers ({analysis['pct_within_preference']:.1f}%)")
    print()
    print(f"Outputs:")
    print(f"  - Ticker pool: {pool_file}")
    print(f"  - Analysis: {report_file}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
