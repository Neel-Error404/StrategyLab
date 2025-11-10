#!/usr/bin/env python3
"""
Top-50 Trade Universe Filter (config driven)
===========================================

Creates the trade universe that feeds downstream portfolio modules.
Originally this script only produced the "anti-cascading" subset (i.e.,
trades that are not consecutive same-direction entries). It now also
supports pulling the Cascading or All-Trades top 50 lists and optionally
skipping the same-direction exclusion so we can benchmark the impact.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PORTFOLIO_DIR = SCRIPT_DIR.parent
ANALYSIS_DIR = PORTFOLIO_DIR.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from generic.modules.config_loader import (
    get_module_spec,
    get_output_dir,
    load_config,
    resolve_paths,
)
from generic.modules.data_loader import load_trades, validate_trade_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top-50 trade universe filter")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    return parser.parse_args()


def load_top50_universe(
    config: Dict,
    variant: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Load the requested Top-50 list from the ticker-ranking output directory.
    """

    variant_key = (variant or "anti_cascading").lower()
    file_map = {
        "anti": ("TOP50_ANTICASCADING_TRADES.csv", "Anti-Cascading Top 50"),
        "anti_cascading": ("TOP50_ANTICASCADING_TRADES.csv", "Anti-Cascading Top 50"),
        "cascading": ("TOP50_CASCADING_TRADES.csv", "Cascading Top 50"),
        "all": ("TOP50_ALL_TRADES.csv", "All-Trades Top 50"),
        "all_trades": ("TOP50_ALL_TRADES.csv", "All-Trades Top 50"),
    }
    if variant_key not in file_map:
        raise ValueError(f"Unsupported top50_variant '{variant}'. Valid options: {list(file_map.keys())}")

    file_name, label = file_map[variant_key]
    output_dir = Path(get_output_dir(config, "ticker_ranking", category="portfolio"))
    top50_file = output_dir / file_name
    top50_df = pd.read_csv(top50_file)
    tickers = top50_df["ticker"].tolist()

    print(f"\n[INFO] Loaded {label}: {len(tickers)} tickers")
    print(f"[INFO] Source file: {top50_file}")
    print(f"[INFO] Top 10: {', '.join(tickers[:10])}")

    return top50_df, tickers, label


def filter_trades_for_top50(trades_df: pd.DataFrame, tickers: list[str], label: str) -> pd.DataFrame:
    """
    Restrict merged trades to the requested Top-50 ticker list.
    """

    print("\n[STEP] Filtering to Top-50 universe")
    print(f"[INFO] Trades before filter: {len(trades_df):,}")

    subset = trades_df[trades_df["ticker"].isin(tickers)].copy()
    print(f"[INFO] Trades after ({label}): {len(subset):,}")
    return subset


def identify_affordable_tickers(
    trades_top50: pd.DataFrame,
    universe_df: pd.DataFrame,
    price_threshold: float,
) -> pd.DataFrame:
    """
    Keep only tickers with last-trade price below the configured threshold.
    """

    print(f"\n[STEP] Identifying tickers under ₹{price_threshold:,.0f}")
    last_trades = trades_top50.groupby("ticker").last().reset_index()

    rows = []
    for _, trade in last_trades.iterrows():
        ticker = trade["ticker"]
        current_price = trade["Exit Price"]
        last_exit_time = trade["Exit Time"]
        last_date = last_exit_time.date() if isinstance(last_exit_time, datetime) else last_exit_time

        metrics = universe_df[universe_df["ticker"] == ticker]
        metrics_row = metrics.iloc[0] if not metrics.empty else {}

        if current_price < 500:
            price_category = "Under ₹500"
        elif current_price < 1000:
            price_category = "Under ₹1000"
        elif current_price < price_threshold:
            price_category = f"Under ₹{int(price_threshold)}"
        else:
            price_category = f"Over ₹{int(price_threshold)}"

        anticascading_rank = metrics_row.get("rank", 0)
        rows.append(
            {
                "ticker": ticker,
                "current_price": current_price,
                "last_trade_date": last_date,
                "anticascading_rank": anticascading_rank,
                "rank": anticascading_rank,
                "profit_factor": metrics_row.get("profit_factor", 0.0),
                "sharpe_like_ratio": metrics_row.get("sharpe_like_ratio", 0.0),
                "composite_score": metrics_row.get("composite_score", 0.0),
                "price_category": price_category,
                "under_threshold": current_price < price_threshold,
            }
        )

    price_df = pd.DataFrame(rows).sort_values("current_price")
    affordable = price_df[price_df["current_price"] < price_threshold].reset_index(drop=True)

    print(f"[INFO] Total tickers in universe: {len(price_df)}")
    print(f"[INFO] Affordable tickers (< ₹{price_threshold:,.0f}): {len(affordable)}")
    if not affordable.empty:
        for _, row in affordable.iterrows():
            print(
                f"   - {row['ticker']:12}  price=₹{row['current_price']:8.2f} "
                f"rank={int(row['rank']):3d}  PF={row['profit_factor']:5.2f}  "
                f"Sharpe≈{row['sharpe_like_ratio']:5.3f}"
            )

    return affordable


def apply_anti_cascading_filter(
    trades_top50: pd.DataFrame,
    affordable_tickers: pd.DataFrame,
    exclude_same_direction: bool,
) -> pd.DataFrame:
    """
    Optionally remove consecutive same-direction trades.
    """

    subset = trades_top50[trades_top50["ticker"].isin(affordable_tickers["ticker"])].copy()
    print(f"\n[STEP] Cascade filter -> starting trades: {len(subset):,}")

    if not exclude_same_direction:
        print("[INFO] exclude_same_direction=False, retaining every trade.")
        subset["trade_category"] = "UNFILTERED"
        return subset

    subset = subset.sort_values(["ticker", "Entry Time"]).reset_index(drop=True)
    subset["Entry Date"] = subset["Entry Time"].dt.date
    subset["prev_ticker"] = subset["ticker"].shift(1)
    subset["prev_entry_date"] = subset["Entry Date"].shift(1)
    subset["prev_trade_type"] = subset["Trade Type"].shift(1)

    def categorize(row):
        if pd.isna(row["prev_ticker"]):
            return "FIRST_TRADE_OVERALL"
        if row["ticker"] != row["prev_ticker"]:
            return "FIRST_TRADE_FOR_TICKER"
        if row["Entry Date"] != row["prev_entry_date"]:
            return "FIRST_TRADE_OF_DAY"
        if row["Trade Type"] == row["prev_trade_type"]:
            return "CONSECUTIVE_SAME_DIRECTION"
        return "CONSECUTIVE_OPPOSITE_DIRECTION"

    subset["trade_category"] = subset.apply(categorize, axis=1)
    summary = subset["trade_category"].value_counts()
    print("[INFO] Cascade categories:")
    for cat, count in summary.items():
        pct = 100 * count / len(subset)
        print(f"   - {cat:30} {count:8,d} ({pct:5.1f}%)")

    mask = subset["trade_category"] != "CONSECUTIVE_SAME_DIRECTION"
    filtered = subset[mask].copy()
    print(f"[INFO] Removed {len(subset) - len(filtered):,} cascading trades ({100 * (1 - len(filtered)/len(subset)):.1f}%)")
    return filtered


def save_filtered_dataset(
    config: Dict,
    trades: pd.DataFrame,
    affordable_tickers: pd.DataFrame,
) -> Path:
    """
    Persist filtered trades, metadata, and summary.
    """

    output_dir = Path(get_output_dir(config, "anti_cascade_filter", category="portfolio"))
    output_dir.mkdir(parents=True, exist_ok=True)

    trades_file = output_dir / "anti_cascading_trades_filtered.csv"
    metadata_file = output_dir / "affordable_tickers_metadata.csv"
    summary_file = output_dir / "anti_cascade_filter_summary.md"

    trades.to_csv(trades_file, index=False)
    affordable_tickers.to_csv(metadata_file, index=False)

    with summary_file.open("w", encoding="utf-8") as fh:
        fh.write("# FILTER SUMMARY\n\n")
        fh.write(f"- Total trades after filter: {len(trades):,}\n")
        fh.write(f"- Affordable tickers: {len(affordable_tickers)}\n")
        fh.write(
            f"- Date range: {trades['Entry Time'].min()} to {trades['Exit Time'].max()}\n\n"
        )

    print(f"[INFO] Saved trades -> {trades_file}")
    print(f"[INFO] Saved metadata -> {metadata_file}")
    return trades_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> Dict | None:
    args = parse_args()
    config = load_config(args.config)
    paths = resolve_paths(config)
    module_spec = get_module_spec(config, "anti_cascade_filter", category="portfolio") or {}
    module_cfg = module_spec.get("config", {}) if isinstance(module_spec, dict) else {}

    price_threshold = module_cfg.get("price_threshold", 2000)
    top50_variant = module_cfg.get("top50_variant", "anti_cascading")
    exclude_same_direction = module_cfg.get("exclude_same_direction", True)

    print("=" * 80)
    print("[RUN] Top-50 trade universe filter")
    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Strategy: {config['run']['strategy']}")
    print(f"[INFO] Date range: {config['run']['date_range']}")
    print(f"[INFO] Variant: {top50_variant} | Exclude cascades: {exclude_same_direction}")
    print("=" * 80)

    try:
        top50_df, tickers, label = load_top50_universe(config, top50_variant)
        trades_df = load_trades(config, paths)
        validation = validate_trade_data(trades_df)
        if not validation["valid"]:
            raise ValueError(f"Trade validation failed: {validation['errors']}")

        trades_top50 = filter_trades_for_top50(trades_df, tickers, label)
        affordable = identify_affordable_tickers(trades_top50, top50_df, price_threshold)
        filtered_trades = apply_anti_cascading_filter(
            trades_top50,
            affordable,
            exclude_same_direction=exclude_same_direction,
        )
        output_file = save_filtered_dataset(config, filtered_trades, affordable)

        print("\n[OK] Universe filtering complete.")
        return {
            "output_file": str(output_file),
            "affordable_tickers": len(affordable),
            "trades": len(filtered_trades),
        }
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return None


if __name__ == "__main__":
    main()
