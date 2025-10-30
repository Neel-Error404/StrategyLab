"""
Metrics computation for options replay results.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .models import ReplayTradeResult


def _safe_mean(values: Iterable[float]) -> float:
    numeric = [float(v) for v in values if v is not None]
    if not numeric:
        return 0.0
    return float(np.mean(numeric))


def _compute_slippage(event) -> Tuple[Optional[float], Optional[float]]:
    """
    Estimate slippage versus synthetic price model.
    """
    synthetic_price = event.notes.get("synthetic_price") if event.notes else None
    if synthetic_price is None:
        return None, None
    synthetic_price = float(synthetic_price)
    if np.isclose(synthetic_price, 0.0):
        return None, None
    raw = float(event.price) - synthetic_price
    bps = (raw / synthetic_price) * 10_000.0
    return raw, bps


@dataclass
class ReplayMetrics:
    summary: Dict[str, float]
    per_ticker: Dict[str, Dict[str, float]]
    equity_curve: pd.DataFrame
    diagnostics: Dict[str, object]


def _compute_equity_curve(trades: Iterable[ReplayTradeResult], initial_capital: float) -> pd.DataFrame:
    rows = []
    realized = 0.0
    for trade in sorted(trades, key=lambda t: t.exit.timestamp):
        realized += trade.realized_pnl
        rows.append(
            {
                "timestamp": trade.exit.timestamp,
                "cumulative_pnl": realized,
                "equity": initial_capital + realized,
            }
        )
    if not rows:
        rows.append({"timestamp": pd.Timestamp.utcnow(), "cumulative_pnl": 0.0, "equity": initial_capital})
    df = pd.DataFrame(rows)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _max_drawdown(series: pd.Series) -> float:
    running_max = series.cummax()
    drawdowns = (series - running_max) / running_max.replace(0, np.nan)
    if drawdowns.empty:
        return 0.0
    return float(drawdowns.min())


def _sharpe_ratio(returns: Iterable[float], risk_free_rate: float = 0.0) -> float:
    arr = np.array(list(returns), dtype=float)
    if arr.size == 0:
        return 0.0
    if np.isclose(arr.std(ddof=1), 0):
        return 0.0
    excess = arr - risk_free_rate
    sharpe = np.mean(excess) / np.std(excess, ddof=1)
    return float(sharpe * math.sqrt(252))


def compute_replay_metrics(
    trades: List[ReplayTradeResult],
    initial_capital: float,
) -> ReplayMetrics:
    """
    Calculate replay level metrics.
    """
    total_trades = len(trades)
    total_pnl = sum(trade.realized_pnl for trade in trades)
    wins = sum(1 for trade in trades if trade.realized_pnl > 0)
    avg_hold_hours = np.mean(
        [
            (trade.exit.timestamp - trade.entry.timestamp).total_seconds() / 3600.0
            for trade in trades
        ]
    ) if trades else 0.0
    returns = [
        trade.return_pct for trade in trades if not np.isnan(trade.return_pct)
    ]
    sharpe = _sharpe_ratio(returns)
    equity_curve = _compute_equity_curve(trades, initial_capital)
    max_dd = _max_drawdown(equity_curve["equity"])
    entry_slippages: List[float] = []
    exit_slippages: List[float] = []
    entry_slippages_bp: List[float] = []
    exit_slippages_bp: List[float] = []
    fallback_trades = 0
    actual_entry_count = 0
    actual_exit_count = 0

    net_return_samples: List[float] = []
    total_costs = 0.0
    total_slippage_costs = 0.0
    net_total_pnl = 0.0
    delta_peak_samples: List[float] = []
    delta_drift_samples: List[float] = []
    theta_cumulative_samples: List[float] = []
    gamma_peak_samples: List[float] = []
    forced_close_count = 0
    forced_close_before_expiry_count = 0
    forced_close_reason_counts: Counter[str] = Counter()
    exit_override_count = 0
    exit_override_reason_counts: Counter[str] = Counter()

    summary = {
        "total_trades": total_trades,
        "win_rate_pct": (wins / total_trades * 100.0) if total_trades else 0.0,
        "total_pnl": total_pnl,
        "realized_pnl": total_pnl,
        "unrealized_pnl": 0.0,
        "return_on_capital_pct": (total_pnl / initial_capital * 100.0) if initial_capital else 0.0,
        "average_hold_hours": avg_hold_hours,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd * 100.0,
    }
    per_ticker: Dict[str, Dict[str, object]] = {}
    for trade in trades:
        ticker = trade.equity_trade.ticker
        ticker_stats = per_ticker.setdefault(
            ticker,
            {
                "trades": 0,
                "pnl": 0.0,
                "net_pnl": 0.0,
                "total_costs": 0.0,
                "total_slippage": 0.0,
                "wins": 0,
                "fallback_count": 0,
                "actual_entry_count": 0,
                "actual_exit_count": 0,
                "entry_slippage": [],
                "exit_slippage": [],
                "entry_slippage_bp": [],
                "exit_slippage_bp": [],
                "net_returns": [],
                "delta_peaks": [],
                "delta_drifts": [],
                "theta_cumulatives": [],
                "gamma_peaks": [],
                "forced_close_count": 0,
                "forced_close_before_expiry_count": 0,
                "exit_override_count": 0,
            },
        )
        ticker_stats["trades"] += 1
        ticker_stats["pnl"] += trade.realized_pnl
        ticker_stats["net_pnl"] += trade.net_realized_pnl
        ticker_stats["total_costs"] += trade.costs.total
        ticker_stats["total_slippage"] += trade.costs.total_slippage
        if trade.realized_pnl > 0:
            ticker_stats["wins"] += 1
        if trade.pricing_fallbacks:
            fallback_trades += 1
            ticker_stats["fallback_count"] += 1
        if trade.entry.pricing_mode == "actual":
            actual_entry_count += 1
            ticker_stats["actual_entry_count"] += 1
        if trade.exit.pricing_mode == "actual":
            actual_exit_count += 1
            ticker_stats["actual_exit_count"] += 1
        entry_raw, entry_bp = _compute_slippage(trade.entry)
        exit_raw, exit_bp = _compute_slippage(trade.exit)
        if entry_raw is not None:
            entry_slippages.append(entry_raw)
            ticker_stats["entry_slippage"].append(entry_raw)
        if exit_raw is not None:
            exit_slippages.append(exit_raw)
            ticker_stats["exit_slippage"].append(exit_raw)
        if entry_bp is not None:
            entry_slippages_bp.append(entry_bp)
            ticker_stats["entry_slippage_bp"].append(entry_bp)
        if exit_bp is not None:
            exit_slippages_bp.append(exit_bp)
            ticker_stats["exit_slippage_bp"].append(exit_bp)
        net_return_samples.append(trade.net_return_pct)
        ticker_stats["net_returns"].append(trade.net_return_pct)
        total_costs += trade.costs.total
        total_slippage_costs += trade.costs.total_slippage
        net_total_pnl += trade.net_realized_pnl
        lifecycle = trade.lifecycle_metrics or {}
        delta_peak = float(lifecycle.get("delta_peak_abs", 0.0))
        delta_drift = float(lifecycle.get("delta_drift", 0.0))
        theta_cumulative = float(lifecycle.get("theta_cumulative", 0.0))
        gamma_peak = float(lifecycle.get("gamma_peak_abs", 0.0))
        delta_peak_samples.append(delta_peak)
        delta_drift_samples.append(delta_drift)
        theta_cumulative_samples.append(theta_cumulative)
        gamma_peak_samples.append(gamma_peak)
        ticker_stats["delta_peaks"].append(delta_peak)
        ticker_stats["delta_drifts"].append(delta_drift)
        ticker_stats["theta_cumulatives"].append(theta_cumulative)
        ticker_stats["gamma_peaks"].append(gamma_peak)
        if getattr(trade, "forced_close", False):
            forced_close_count += 1
            ticker_stats["forced_close_count"] += 1
            if getattr(trade, "expiry_forced_close", False):
                forced_close_before_expiry_count += 1
                ticker_stats["forced_close_before_expiry_count"] += 1
            reason = getattr(trade, "forced_close_reason", None)
            if reason:
                forced_close_reason_counts[str(reason)] += 1
        if getattr(trade, "exit_override_reason", None):
            exit_override_count += 1
            ticker_stats["exit_override_count"] += 1
            exit_override_reason_counts[str(trade.exit_override_reason)] += 1

    for ticker, stats in per_ticker.items():
        trades_count = stats["trades"]
        stats["win_rate_pct"] = (stats["wins"] / stats["trades"] * 100.0) if stats["trades"] else 0.0
        stats["return_pct"] = (stats["pnl"] / initial_capital * 100.0) if initial_capital else 0.0
        stats["net_return_pct"] = (stats["net_pnl"] / initial_capital * 100.0) if initial_capital else 0.0
        stats["fallback_rate_pct"] = (
            (stats["fallback_count"] / trades_count * 100.0) if trades_count else 0.0
        )
        stats["actual_entry_ratio_pct"] = (
            (stats["actual_entry_count"] / trades_count * 100.0) if trades_count else 0.0
        )
        stats["actual_exit_ratio_pct"] = (
            (stats["actual_exit_count"] / trades_count * 100.0) if trades_count else 0.0
        )
        stats["entry_slippage_mean"] = _safe_mean(stats.pop("entry_slippage"))
        stats["exit_slippage_mean"] = _safe_mean(stats.pop("exit_slippage"))
        stats["entry_slippage_bp_mean"] = _safe_mean(stats.pop("entry_slippage_bp"))
        stats["exit_slippage_bp_mean"] = _safe_mean(stats.pop("exit_slippage_bp"))
        stats["net_return_pct_mean"] = _safe_mean(stats.pop("net_returns"))
        stats["average_cost_per_trade"] = (stats["total_costs"] / trades_count) if trades_count else 0.0
        stats["average_slippage_per_trade"] = (stats["total_slippage"] / trades_count) if trades_count else 0.0
        stats["average_delta_peak_abs"] = _safe_mean(stats.pop("delta_peaks"))
        stats["average_delta_drift"] = _safe_mean(stats.pop("delta_drifts"))
        stats["average_theta_cumulative"] = _safe_mean(stats.pop("theta_cumulatives"))
        stats["average_gamma_peak_abs"] = _safe_mean(stats.pop("gamma_peaks"))
        stats["forced_close_share_pct"] = (
            (stats["forced_close_count"] / trades_count * 100.0) if trades_count else 0.0
        )
        stats["exit_override_share_pct"] = (
            (stats["exit_override_count"] / trades_count * 100.0) if trades_count else 0.0
        )

    summary.update(
        {
            "fallback_rate_pct": (fallback_trades / total_trades * 100.0) if total_trades else 0.0,
            "actual_entry_ratio_pct": (actual_entry_count / total_trades * 100.0) if total_trades else 0.0,
            "actual_exit_ratio_pct": (actual_exit_count / total_trades * 100.0) if total_trades else 0.0,
            "average_entry_slippage": _safe_mean(entry_slippages),
            "average_exit_slippage": _safe_mean(exit_slippages),
            "average_entry_slippage_bp": _safe_mean(entry_slippages_bp),
            "average_exit_slippage_bp": _safe_mean(exit_slippages_bp),
            "total_transaction_costs": total_costs,
            "total_slippage_costs": total_slippage_costs,
            "net_realized_pnl": net_total_pnl,
            "net_return_on_capital_pct": (net_total_pnl / initial_capital * 100.0) if initial_capital else 0.0,
            "average_cost_per_trade": (total_costs / total_trades) if total_trades else 0.0,
            "average_net_return_pct": _safe_mean(net_return_samples),
            "average_delta_peak_abs": _safe_mean(delta_peak_samples),
            "average_delta_drift": _safe_mean(delta_drift_samples),
            "average_theta_cumulative": _safe_mean(theta_cumulative_samples),
            "average_gamma_peak_abs": _safe_mean(gamma_peak_samples),
            "forced_close_count": forced_close_count,
            "forced_close_before_expiry_count": forced_close_before_expiry_count,
            "forced_close_share_pct": (forced_close_count / total_trades * 100.0) if total_trades else 0.0,
            "forced_close_reason_counts": dict(forced_close_reason_counts),
            "exit_override_count": exit_override_count,
            "exit_override_share_pct": (exit_override_count / total_trades * 100.0) if total_trades else 0.0,
            "exit_override_reason_counts": dict(exit_override_reason_counts),
        }
    )

    diagnostics = {
        "fallback_trades": fallback_trades,
        "actual_entry_count": actual_entry_count,
        "actual_exit_count": actual_exit_count,
        "entry_slippage_samples": len(entry_slippages),
        "exit_slippage_samples": len(exit_slippages),
        "total_transaction_costs": total_costs,
        "total_slippage_costs": total_slippage_costs,
        "forced_close_count": forced_close_count,
        "forced_close_before_expiry_count": forced_close_before_expiry_count,
        "exit_override_count": exit_override_count,
    }

    return ReplayMetrics(
        summary=summary,
        per_ticker=per_ticker,
        equity_curve=equity_curve,
        diagnostics=diagnostics,
    )
