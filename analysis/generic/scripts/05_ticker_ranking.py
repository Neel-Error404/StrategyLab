#!/usr/bin/env python3
"""
Ticker Ranking Analysis (Config Driven)
=======================================

Ranks tickers by composite performance scores derived from profitability,
risk, consistency, efficiency, and liquidity metrics. Outputs comprehensive
CSV tables plus supporting summaries for downstream portfolio construction.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config_loader import (
    get_analysis_config,
    load_config,
    resolve_artifact_path,
    resolve_paths,
)
from modules.data_loader import load_trades

MODULE_NAME = "ticker_ranking"


def load_trade_data(config: Dict[str, Any], paths: Dict[str, str], sample_size: Optional[int] = None) -> pd.DataFrame:
    print("📊 Loading trade data for ticker ranking analysis...")
    df = load_trades(config, paths, sample_size=sample_size)
    print(f"✅ Loaded {len(df):,} trades covering {df['ticker'].nunique()} tickers")
    return df


def filter_by_min_trades(df: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    if min_trades <= 0:
        return df
    ticker_counts = df['ticker'].value_counts()
    valid_tickers = ticker_counts[ticker_counts >= min_trades].index
    filtered = df[df['ticker'].isin(valid_tickers)]
    removed = set(df['ticker'].unique()) - set(valid_tickers)
    if removed:
        print(f"⚠️  Excluding {len(removed)} tickers with fewer than {min_trades} trades.")
    return filtered


def calculate_ticker_metrics(df: pd.DataFrame) -> pd.DataFrame:
    print("\n📊 Calculating ticker-level metrics...")
    metrics = []
    grouped = df.groupby('ticker')

    for ticker, ticker_data in grouped:
        total_trades = len(ticker_data)
        buy_data = ticker_data[ticker_data['Trade Type'] == 'Buy']
        sell_data = ticker_data[ticker_data['Trade Type'] == 'Sell']

        total_pnl = ticker_data['Profit (Currency)'].sum()
        avg_profit = ticker_data['Profit (Currency)'].mean()
        win_rate = (ticker_data['Profit (Currency)'] > 0).mean() * 100

        max_drawdown = ticker_data['Drawdown (%)'].max()
        avg_drawdown = ticker_data['Drawdown (%)'].mean()
        profit_std = ticker_data['Profit (Currency)'].std()
        sharpe_like = avg_profit / profit_std if profit_std and profit_std > 0 else 0.0

        gross_profit = ticker_data[ticker_data['Profit (Currency)'] > 0]['Profit (Currency)'].sum()
        gross_loss = abs(ticker_data[ticker_data['Profit (Currency)'] < 0]['Profit (Currency)'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        profit_consistency = ticker_data.groupby(ticker_data['Entry Time'].dt.to_period('M'))['Profit (Currency)'].sum()
        profit_consistency = (profit_consistency > 0).mean() * 100 if len(profit_consistency) else 0.0

        # Capture efficiency approximations
        def capture_efficiency(subset: pd.DataFrame, high_col: str, low_col: str, is_buy: bool) -> float:
            if subset.empty:
                return 0.0
            if is_buy:
                potential = subset[high_col] - subset['Entry Price']
                actual = subset['Exit Price'] - subset['Entry Price']
            else:
                potential = subset['Entry Price'] - subset[low_col]
                actual = subset['Entry Price'] - subset['Exit Price']
            eff = np.where(potential > 0, (actual / potential) * 100, np.nan)
            return float(np.nanmean(eff)) if len(eff) else 0.0

        buy_eff = capture_efficiency(buy_data, 'High During Trade', 'Low During Trade', True)
        sell_eff = capture_efficiency(sell_data, 'High During Trade', 'Low During Trade', False)

        metrics.append({
            'ticker': ticker,
            'total_trades': total_trades,
            'buy_trades': len(buy_data),
            'sell_trades': len(sell_data),
            'total_pnl': float(total_pnl),
            'avg_profit_per_trade': float(avg_profit),
            'win_rate': float(win_rate),
            'buy_win_rate': float((buy_data['Profit (Currency)'] > 0).mean() * 100 if not buy_data.empty else 0.0),
            'sell_win_rate': float((sell_data['Profit (Currency)'] > 0).mean() * 100 if not sell_data.empty else 0.0),
            'max_drawdown': float(max_drawdown) if not pd.isna(max_drawdown) else 0.0,
            'avg_drawdown': float(avg_drawdown) if not pd.isna(avg_drawdown) else 0.0,
            'profit_factor': float(profit_factor),
            'sharpe_like_ratio': float(sharpe_like),
            'avg_duration_min': float(ticker_data['Trade Duration (min)'].mean()),
            'avg_recovery_time': float(ticker_data['Recovery Time (min)'].mean()),
            'profit_consistency': float(profit_consistency),
            'avg_capture_efficiency': float(np.nanmean([buy_eff, sell_eff])),
        })

    metrics_df = pd.DataFrame(metrics)
    print(f"✅ Metrics computed for {len(metrics_df)} tickers.")
    return metrics_df


def normalise_scores(metrics_df: pd.DataFrame, config_weights: Dict[str, float]) -> pd.DataFrame:
    df = metrics_df.copy()

    positive_metrics = ['total_pnl', 'avg_profit_per_trade', 'win_rate', 'profit_factor',
                        'sharpe_like_ratio', 'profit_consistency', 'avg_capture_efficiency']
    negative_metrics = ['max_drawdown', 'avg_drawdown', 'avg_recovery_time']

    df['profit_factor'] = df['profit_factor'].replace([np.inf, -np.inf], df['profit_factor'][np.isfinite(df['profit_factor'])].max())

    def scale_column(series: pd.Series, invert: bool = False) -> pd.Series:
        if series.empty:
            return pd.Series(dtype=float)
        min_val, max_val = series.min(), series.max()
        if max_val == min_val:
            return pd.Series(50.0, index=series.index)
        if invert:
            return ((max_val - series) / (max_val - min_val)) * 100
        return ((series - min_val) / (max_val - min_val)) * 100

    for metric in positive_metrics:
        if metric in df.columns:
            df[f'{metric}_score'] = scale_column(df[metric])
    for metric in negative_metrics:
        if metric in df.columns:
            df[f'{metric}_score'] = scale_column(df[metric], invert=True)

    # Component scores
    df['profitability_score'] = (
        df['total_pnl_score'] * 0.4 +
        df['avg_profit_per_trade_score'] * 0.4 +
        df['win_rate_score'] * 0.2
    )
    df['risk_score'] = (
        df['max_drawdown_score'] * 0.4 +
        df['avg_drawdown_score'] * 0.3 +
        df['sharpe_like_ratio_score'] * 0.3
    )
    df['consistency_score'] = (
        df['profit_consistency_score'] * 0.6 +
        df['profit_factor_score'] * 0.4
    )
    df['efficiency_score'] = df['avg_capture_efficiency_score']

    optimal_trades = 2500
    df['frequency_score'] = 100 - (abs(df['total_trades'] - optimal_trades) / optimal_trades * 100)
    df['frequency_score'] = df['frequency_score'].clip(0, 100)

    weights = {
        'profitability': config_weights.get('profitability', 0.30),
        'risk_management': config_weights.get('risk_management', 0.25),
        'consistency': config_weights.get('consistency', 0.20),
        'efficiency': config_weights.get('efficiency', 0.15),
        'frequency': config_weights.get('frequency', 0.10),
    }

    df['composite_score'] = (
        df['profitability_score'] * weights['profitability'] +
        df['risk_score'] * weights['risk_management'] +
        df['consistency_score'] * weights['consistency'] +
        df['efficiency_score'] * weights['efficiency'] +
        df['frequency_score'] * weights['frequency']
    )

    print("\n🧮 Applied composite score weights:")
    for key, value in weights.items():
        print(f"   • {key.replace('_', ' ').title()}: {value*100:.0f}%")

    return df


def classify_tiers(scoring_df: pd.DataFrame) -> pd.DataFrame:
    ranking_df = scoring_df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    total = len(ranking_df)

    tier1 = max(75, ranking_df['composite_score'].quantile(0.80))
    tier2 = max(55, ranking_df['composite_score'].quantile(0.50))
    tier3 = max(35, ranking_df['composite_score'].quantile(0.20))

    def assign(score: float) -> str:
        if score >= tier1:
            return "Tier 1 - Excellent"
        if score >= tier2:
            return "Tier 2 - Good"
        if score >= tier3:
            return "Tier 3 - Average"
        return "Tier 4 - Poor"

    ranking_df['tier'] = ranking_df['composite_score'].apply(assign)
    ranking_df['rank'] = range(1, total + 1)

    print("\n🏆 Tier distribution:")
    for tier, count in ranking_df['tier'].value_counts().sort_index().items():
        print(f"   • {tier}: {count} tickers ({count/total*100:.1f}%)")

    return ranking_df


def analyse_top_bottom(ranking_df: pd.DataFrame, top_n: int) -> Dict[str, pd.DataFrame]:
    top_df = ranking_df.head(top_n)
    bottom_df = ranking_df.tail(top_n)
    print(f"\n🥇 Top {top_n} tickers preview:")
    for _, row in top_df.head(10).iterrows():
        print(f"   {row['rank']:2d}. {row['ticker']} | score {row['composite_score']:.1f} | P&L ₹{row['total_pnl']:,.0f}")
    print(f"\n🔻 Bottom {top_n} tickers preview:")
    for _, row in bottom_df.head(10).iterrows():
        print(f"   {row['rank']:2d}. {row['ticker']} | score {row['composite_score']:.1f} | P&L ₹{row['total_pnl']:,.0f}")
    return {'top': top_df, 'bottom': bottom_df}


def tier_summary(ranking_df: pd.DataFrame) -> pd.DataFrame:
    summary = ranking_df.groupby('tier').agg({
        'total_pnl': ['count', 'mean', 'sum'],
        'avg_profit_per_trade': 'mean',
        'win_rate': 'mean',
        'max_drawdown': 'mean',
        'profit_factor': 'mean',
        'avg_capture_efficiency': 'mean',
        'total_trades': 'mean',
        'composite_score': ['mean', 'min', 'max'],
    }).round(2)
    print("\n📊 Tier summary:")
    print(summary)
    return summary


def liquidity_analysis(ranking_df: pd.DataFrame) -> pd.DataFrame:
    ranking_df = ranking_df.copy()
    ranking_df['liquidity_category'] = pd.cut(
        ranking_df['total_trades'],
        bins=[0, 1000, 2000, 3000, 5000, float('inf')],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    summary = ranking_df.groupby('liquidity_category').agg({
        'ticker': 'count',
        'total_pnl': 'mean',
        'avg_profit_per_trade': 'mean',
        'win_rate': 'mean',
        'composite_score': 'mean',
    }).round(2)
    summary.columns = ['Ticker_Count', 'Avg_Total_PnL', 'Avg_Profit_Per_Trade', 'Avg_Win_Rate', 'Avg_Composite_Score']
    print("\n💧 Liquidity analysis:")
    print(summary)
    return summary


def save_outputs(
    config: Dict[str, Any],
    ranking_df: pd.DataFrame,
    top_df: pd.DataFrame,
    bottom_df: pd.DataFrame,
    tier_df: pd.DataFrame,
    liquidity_df: pd.DataFrame,
) -> None:
    base_path = Path(resolve_artifact_path(config, MODULE_NAME, 'ticker_scores', artifact_type='csv'))
    output_dir = base_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    ranking_columns = [
        'rank', 'ticker', 'tier', 'composite_score', 'total_trades', 'total_pnl',
        'avg_profit_per_trade', 'win_rate', 'buy_win_rate', 'sell_win_rate',
        'max_drawdown', 'avg_drawdown', 'profit_factor', 'avg_capture_efficiency',
        'profitability_score', 'risk_score', 'consistency_score', 'efficiency_score', 'frequency_score'
    ]
    ranking_df[ranking_columns].round(3).to_csv(base_path, index=False)
    print(f"\n💾 Saved ranking table → {base_path}")

    top_df.to_csv(output_dir / "top_performers.csv", index=False)
    bottom_df.to_csv(output_dir / "bottom_performers.csv", index=False)
    tier_df.to_csv(output_dir / "tier_analysis.csv")
    liquidity_df.to_csv(output_dir / "liquidity_analysis.csv")

    summary_payload = {
        'analysis_date': datetime.now().isoformat(),
        'total_tickers': len(ranking_df),
        'tier_distribution': ranking_df['tier'].value_counts().to_dict(),
        'top_10': ranking_df.head(10)[['ticker', 'composite_score', 'total_pnl']].to_dict('records'),
        'bottom_10': ranking_df.tail(10)[['ticker', 'composite_score', 'total_pnl']].to_dict('records'),
        'best_ticker': ranking_df.iloc[0]['ticker'],
        'best_score': float(ranking_df.iloc[0]['composite_score']),
    }
    summary_path = output_dir / "ticker_analysis_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, default=str))
    print(f"💾 Saved summary JSON → {summary_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Comprehensive ticker ranking analysis")
    parser.add_argument("--config", required=True, help="Path to analysis YAML config")
    parser.add_argument("--sample", type=int, help="Optional trade sample size for quick iteration")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = resolve_paths(config)
    module_cfg = get_analysis_config(config, MODULE_NAME) or {}

    sample_size = args.sample or module_cfg.get('sample_size')
    df = load_trade_data(config, paths, sample_size=sample_size)
    df = filter_by_min_trades(df, module_cfg.get('min_trades', 20))
    if df.empty:
        print("❌ No tickers meet the minimum trade count requirement.")
        return 1

    metrics_df = calculate_ticker_metrics(df)
    scoring_df = normalise_scores(metrics_df, module_cfg.get('weighting_scheme', {}))
    ranking_df = classify_tiers(scoring_df)
    extremes = analyse_top_bottom(ranking_df, top_n=module_cfg.get('top_n', 20))
    tier_df = tier_summary(ranking_df)
    liquidity_df = liquidity_analysis(ranking_df)
    save_outputs(config, ranking_df, extremes['top'], extremes['bottom'], tier_df, liquidity_df)

    print("\n✅ TICKER RANKING ANALYSIS COMPLETE")
    print(f"   Total tickers analysed: {len(ranking_df)}")
    print(f"   Top performer: {ranking_df.iloc[0]['ticker']} (score {ranking_df.iloc[0]['composite_score']:.1f})")
    print(f"   Results directory: {Path(resolve_artifact_path(config, MODULE_NAME, 'ticker_scores')).parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
