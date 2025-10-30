"""
High-level orchestration for the Phase 3 options replay engine.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, date
from hashlib import sha256
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.core.options.pricing.costs import TradeCostSummary
from src.core.options.replay.config import OptionsReplayConfig
from src.core.options.replay.models import ReplayTradeResult, ReplayRunArtifacts
from src.core.options.replay.data_loader import (
    OptionDataStore,
    find_option_data_dir,
    load_equity_trades,
    load_underlying_data,
    parse_date_range,
)
from src.core.options.replay.pricing import HybridPricingEngine, LiquidityFilterError
from src.core.options.replay.trade_mapper import map_trade_to_option
from src.core.options.replay.control_board import ControlBoardBuilder, ControlBoardResult
from src.core.options.replay.metrics import compute_replay_metrics
from src.core.options.replay.risk import RiskManager


def _compile_risk_flags(
    *,
    override_flag: Optional[str],
    forced_close_reason: Optional[str],
    expiry_forced_close: bool,
    assignment_flag: bool,
) -> List[str]:
    """
    Normalise risk flag emission so that override-driven exits do not double count.
    """
    flags: List[str] = []
    if override_flag:
        flags.append(override_flag)
    if expiry_forced_close:
        flags.append("force_closed_before_expiry")
    elif forced_close_reason:
        flags.append("force_closed_before_exit_signal")
    if assignment_flag:
        flags.append("assignment_risk")
    return flags


@dataclass
class TickerContext:
    """Container for per-ticker replay artefacts."""

    ticker: str
    pricer: HybridPricingEngine
    underlying_cache: Dict[str, pd.DataFrame]
    option_store: OptionDataStore
    date_ranges: Sequence[str]
    coverage_start: Optional[date]
    coverage_end: Optional[date]


class OptionsReplayEngine:
    """
    End-to-end orchestrator that consumes equity trades, maps them to
    options, applies hybrid pricing, enforces risk, and emits artefacts.
    """

    def __init__(
        self,
        config: OptionsReplayConfig,
        logger: Optional[logging.Logger] = None,
        control_board: Optional[ControlBoardBuilder] = None,
    ) -> None:
        self.config = config
        self.logger = logger or logging.getLogger("OptionsReplayEngine")
        self.logger.setLevel(getattr(logging, config.logging.level.upper(), logging.INFO))
        self.run_id = self._build_run_id()
        self.output_dir = self._prepare_output_dir()
        self.hash_records: Dict[str, str] = {}
        self.structured_log_path = self.output_dir / "logs.jsonl"
        self._structured_log_buffer: List[Dict[str, object]] = []
        self.control_board = control_board

        random.seed(config.inputs.seed)
        np.random.seed(config.inputs.seed)

        if self.control_board:
            self.control_board.attach_run_context(self.run_id, self.output_dir)

    def _build_run_id(self) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{self.config.inputs.run_label}"

    def _prepare_output_dir(self) -> Path:
        out_dir = self.config.output.output_dir / self.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _log_structured(self, level: str, message: str, **kwargs: object) -> None:
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level.upper(),
            "message": message,
            **kwargs,
        }
        self._structured_log_buffer.append(record)
        getattr(self.logger, level.lower(), self.logger.info)(message)
        if self.control_board:
            self.control_board.record_structured_log(record)

    def _flush_structured_log(self) -> None:
        if not self._structured_log_buffer:
            return
        with self.structured_log_path.open("w", encoding="utf-8") as fh:
            for record in self._structured_log_buffer:
                fh.write(json.dumps(record, default=str) + "\n")

    def _normalise_tickers(self, tickers: Optional[Sequence[str] | str]) -> Tuple[Optional[List[str]], str]:
        """
        Normalise user-supplied ticker arguments.

        Returns:
            (resolved_tickers, selection_mode)
            selection_mode: 'manual' | 'auto'
        """
        if tickers is None:
            return None, "auto"
        if isinstance(tickers, str):
            token = tickers.strip()
            if not token or token.lower() == "auto":
                return None, "auto"
            return [token.upper()], "manual"
        resolved: List[str] = []
        mode = "manual"
        for item in tickers:
            if item is None:
                continue
            token = str(item).strip()
            if not token:
                continue
            if token.lower() == "auto":
                mode = "auto"
                continue
            resolved.append(token.upper())
        if mode == "auto" and not resolved:
            return None, "auto"
        return (resolved or None), mode

    def _prepare_ticker_contexts(
        self,
        trades_by_ticker: Dict[str, List[Tuple[object, str]]],
    ) -> Tuple[Dict[str, TickerContext], int]:
        """
        Build pricing contexts for each ticker, optionally in parallel.

        Returns:
            (contexts, workers_used)
        """
        tasks: List[Tuple[str, List[str]]] = []
        for ticker, items in trades_by_ticker.items():
            date_labels = sorted({label for _, label in items})
            tasks.append((ticker, date_labels))

        contexts: Dict[str, TickerContext] = {}
        parallel_cfg = self.config.performance.parallel
        use_parallel = (
            parallel_cfg.enabled and parallel_cfg.max_workers > 1 and len(tasks) > 1
        )
        workers_used = 1

        if use_parallel:
            workers_used = min(len(tasks), parallel_cfg.max_workers)
            with ThreadPoolExecutor(max_workers=parallel_cfg.max_workers) as executor:
                future_map = {
                    executor.submit(self._build_pricers, ticker, date_ranges): (ticker, date_ranges)
                    for ticker, date_ranges in tasks
                }
                for future in as_completed(future_map):
                    ticker, date_ranges = future_map[future]
                    pricer, underlying_cache, option_store = future.result()
                    coverage_start, coverage_end = option_store.coverage_bounds()
                    contexts[ticker] = TickerContext(
                        ticker=ticker,
                        pricer=pricer,
                        underlying_cache=underlying_cache,
                        option_store=option_store,
                        date_ranges=date_ranges,
                        coverage_start=coverage_start,
                        coverage_end=coverage_end,
                    )
                    self._log_structured(
                        "info",
                        "ticker_context_ready",
                        ticker=ticker,
                        date_ranges=date_ranges,
                        coverage_start=coverage_start.isoformat() if coverage_start else None,
                        coverage_end=coverage_end.isoformat() if coverage_end else None,
                        parallel_mode="threaded",
                    )
        else:
            for ticker, date_ranges in tasks:
                pricer, underlying_cache, option_store = self._build_pricers(ticker, date_ranges)
                coverage_start, coverage_end = option_store.coverage_bounds()
                contexts[ticker] = TickerContext(
                    ticker=ticker,
                    pricer=pricer,
                    underlying_cache=underlying_cache,
                    option_store=option_store,
                    date_ranges=date_ranges,
                    coverage_start=coverage_start,
                    coverage_end=coverage_end,
                )
                self._log_structured(
                    "info",
                    "ticker_context_ready",
                    ticker=ticker,
                    date_ranges=date_ranges,
                    coverage_start=coverage_start.isoformat() if coverage_start else None,
                    coverage_end=coverage_end.isoformat() if coverage_end else None,
                    parallel_mode="serial",
                )

        return contexts, workers_used

    def _filter_trades(
        self,
        trades,
        tickers: Optional[Sequence[str]],
        date_ranges: Sequence[str],
    ):
        tz = self.config.inputs.timezone
        bounds = []
        for dr in date_ranges:
            start, end = parse_date_range(dr)
            if start.tzinfo is None:
                start = start.tz_localize(tz)
            else:
                start = start.tz_convert(tz)
            if end.tzinfo is None:
                end = end.tz_localize(tz)
            else:
                end = end.tz_convert(tz)
            bounds.append((dr, start, end))
        selected = []
        tickers_upper = {t.upper() for t in tickers} if tickers else None
        for trade in trades:
            if tickers_upper and trade.ticker not in tickers_upper:
                continue
            for label, start, end in bounds:
                entry_day = trade.entry_time.tz_convert(tz).normalize()
                if start <= entry_day <= end:
                    selected.append((trade, label))
                    break
        return selected

    def _build_pricers(
        self,
        ticker: str,
        date_ranges: Sequence[str],
    ) -> Tuple[HybridPricingEngine, Dict[str, pd.DataFrame], OptionDataStore]:
        bounds = [parse_date_range(rng) for rng in date_ranges]
        start = min(start for start, _ in bounds)
        end = max(end for _, end in bounds)
        option_dir = find_option_data_dir(
            self.config.inputs.options_data_root,
            ticker,
            start,
            end,
        )
        option_store = OptionDataStore(
            option_dir,
            timeframe=self.config.inputs.options_timeframe,
            timeframes=self.config.inputs.options_timeframes,
        )
        underlying_cache: Dict[str, pd.DataFrame] = {}
        underlying_frames: List[pd.DataFrame] = []
        for date_range in date_ranges:
            df = load_underlying_data(
                root=self.config.inputs.equity_data_root,
                date_range=date_range,
                timeframe=self.config.inputs.underlying_timeframe,
                ticker=ticker,
                tz=self.config.inputs.timezone,
            )
            underlying_cache[date_range] = df
            underlying_frames.append(df)
        consolidated = pd.concat(underlying_frames, ignore_index=True).drop_duplicates(subset=["timestamp"])
        consolidated.sort_values("timestamp", inplace=True)
        pricer = HybridPricingEngine(self.config, consolidated, option_store)
        return pricer, underlying_cache, option_store

    def _resolve_exit_override(
        self,
        entry_event,
        lifecycle: List["OptionPositionSnapshot"],
        entry_time: pd.Timestamp,
        default_exit_time: pd.Timestamp,
    ) -> Tuple[Optional[pd.Timestamp], Optional[str], Optional[Dict[str, object]], Optional[int]]:
        """
        Evaluate stop-loss / take-profit / time-based overrides and return the earliest trigger.
        """
        exit_cfg = self.config.position_management.exit
        if not lifecycle or (not exit_cfg.stop_loss.enabled and not exit_cfg.take_profit.enabled and not exit_cfg.time_based.enabled):
            return None, None, None, None

        base_price = float(entry_event.price)
        if base_price <= 0:
            base_price = None

        def _pct_change(current_price: float) -> Optional[float]:
            if base_price is None or base_price == 0:
                return None
            return (current_price - base_price) / base_price

        triggered_time: Optional[pd.Timestamp] = None
        reason: Optional[str] = None
        details: Optional[Dict[str, object]] = None
        index: Optional[int] = None

        for idx, snapshot in enumerate(lifecycle[1:], start=1):
            snap_ts = snapshot.timestamp
            if snap_ts >= default_exit_time:
                break
            hold_hours = (snap_ts - entry_time).total_seconds() / 3600.0
            pct_change = _pct_change(snapshot.option_price)

            if exit_cfg.stop_loss.enabled and pct_change is not None:
                if pct_change <= exit_cfg.stop_loss.threshold_pct:
                    triggered_time = snap_ts
                    reason = "stop_loss"
                    details = {
                        "pct_change": pct_change,
                        "threshold_pct": exit_cfg.stop_loss.threshold_pct,
                        "hold_hours": hold_hours,
                    }
                    index = idx
                    break

            if exit_cfg.take_profit.enabled and pct_change is not None:
                if pct_change >= exit_cfg.take_profit.threshold_pct:
                    triggered_time = snap_ts
                    reason = "take_profit"
                    details = {
                        "pct_change": pct_change,
                        "threshold_pct": exit_cfg.take_profit.threshold_pct,
                        "hold_hours": hold_hours,
                    }
                    index = idx
                    break

            if exit_cfg.time_based.enabled and exit_cfg.time_based.max_hold_hours > 0:
                if hold_hours >= exit_cfg.time_based.max_hold_hours:
                    triggered_time = snap_ts
                    reason = "time_based"
                    details = {
                        "hold_hours": hold_hours,
                        "threshold_hours": exit_cfg.time_based.max_hold_hours,
                    }
                    index = idx
                    break

        return triggered_time, reason, details, index

    def _option_symbol(self, contract) -> str:
        expiry = contract.expiry.tz_convert(self.config.inputs.timezone)
        expiry_code = expiry.strftime("%y%m%d")
        strike_code = f"{int(round(contract.strike)):05d}"
        return f"{contract.ticker}{expiry_code}{strike_code}{contract.option_type.value}"

    def _emit_trade_dataframe(self, trades: List[ReplayTradeResult]) -> pd.DataFrame:
        rows = []
        for trade in trades:
            meta = getattr(trade, "mapping_metadata", {}) or {}
            expiry_meta = meta.get("expiry_info", {}) if isinstance(meta, dict) else {}
            strike_meta = meta.get("strike_info", {}) if isinstance(meta, dict) else {}
            candidates = strike_meta.get("candidates_considered") if isinstance(strike_meta, dict) else None
            candidate_count = len(candidates) if isinstance(candidates, list) else strike_meta.get("candidate_count")
            entry_costs = trade.costs.entry
            exit_costs = trade.costs.exit
            entry_notes = trade.entry.notes or {}
            exit_notes = trade.exit.notes or {}
            fallback_list = [fb for fb in trade.pricing_fallbacks if fb]
            fallback_serialised = ";".join(sorted(set(fallback_list)))
            rows.append(
                {
                    "trade_id": trade.equity_trade.trade_id,
                    "ticker": trade.equity_trade.ticker,
                    "option_symbol": self._option_symbol(trade.contract),
                    "expiry": trade.contract.expiry.isoformat(),
                    "strike": trade.contract.strike,
                    "option_type": trade.contract.option_type.value,
                    "lots": trade.lots,
                    "quantity": trade.quantity,
                    "entry_time": trade.entry.timestamp.isoformat(),
                    "exit_time": trade.exit.timestamp.isoformat(),
                    "entry_price": trade.entry.price,
                    "exit_price": trade.exit.price,
                    "pricing_mode_entry": trade.entry.pricing_mode,
                    "pricing_mode_exit": trade.exit.pricing_mode,
                    "entry_implied_vol": trade.entry.implied_vol,
                    "exit_implied_vol": trade.exit.implied_vol,
                    "entry_synthetic_vol": entry_notes.get("synthetic_vol"),
                    "exit_synthetic_vol": exit_notes.get("synthetic_vol"),
                    "entry_time_to_expiry_days": entry_notes.get("time_to_expiry_days"),
                    "exit_time_to_expiry_days": exit_notes.get("time_to_expiry_days"),
                    "entry_calibration_error": entry_notes.get("calibration_error"),
                    "exit_calibration_error": exit_notes.get("calibration_error"),
                    "entry_fallback_reason": entry_notes.get("fallback_reason"),
                    "exit_fallback_reason": exit_notes.get("fallback_reason"),
                    "exit_override_reason": trade.exit_override_reason,
                    "exit_override_timestamp": trade.exit_override_timestamp.isoformat()
                    if trade.exit_override_timestamp
                    else None,
                    "exit_override_details": json.dumps(trade.exit_override_details, sort_keys=True, default=str)
                    if trade.exit_override_details
                    else None,
                    "forced_close": trade.forced_close,
                    "forced_close_reason": trade.forced_close_reason,
                    "forced_close_before_expiry": trade.expiry_forced_close,
                    "realized_pnl": trade.realized_pnl,
                    "return_pct": trade.return_pct,
                    "net_realized_pnl": trade.net_realized_pnl,
                    "net_return_pct": trade.net_return_pct,
                    "entry_cost_total": entry_costs.total,
                    "entry_cost_brokerage": entry_costs.brokerage,
                    "entry_cost_exchange": entry_costs.exchange_transaction,
                    "entry_cost_stt": entry_costs.stt,
                    "entry_cost_stamp": entry_costs.stamp_duty,
                    "entry_cost_sebi": entry_costs.sebi_fee,
                    "entry_cost_clearing": entry_costs.clearing_fee,
                    "entry_cost_gst": entry_costs.gst,
                    "entry_cost_slippage": entry_costs.slippage,
                    "exit_cost_total": exit_costs.total,
                    "exit_cost_brokerage": exit_costs.brokerage,
                    "exit_cost_exchange": exit_costs.exchange_transaction,
                    "exit_cost_stt": exit_costs.stt,
                    "exit_cost_stamp": exit_costs.stamp_duty,
                    "exit_cost_sebi": exit_costs.sebi_fee,
                    "exit_cost_clearing": exit_costs.clearing_fee,
                    "exit_cost_gst": exit_costs.gst,
                    "exit_cost_slippage": exit_costs.slippage,
                    "total_transaction_costs": trade.costs.total,
                    "total_slippage_costs": trade.costs.total_slippage,
                    "max_drawdown_pct": trade.max_drawdown_pct,
                    "risk_flags": ";".join(trade.risk_flags),
                    "pricing_fallbacks": fallback_serialised,
                    "expiry_selection_method": expiry_meta.get("method"),
                    "expiry_selection_reason": expiry_meta.get("fallback_reason"),
                    "expiry_selection_dte": expiry_meta.get("dte_days"),
                    "expiry_type": expiry_meta.get("expiry_type"),
                    "strike_selection_method": strike_meta.get("method"),
                    "strike_selection_status": strike_meta.get("selection_status"),
                    "strike_target_delta": strike_meta.get("target_delta"),
                    "strike_selected_delta": strike_meta.get("selected_delta"),
                    "strike_delta_diff": strike_meta.get("delta_diff"),
                    "strike_candidate_count": candidate_count,
                    "strike_fallback_reason": strike_meta.get("fallback_reason"),
                    "strike_target_premium": strike_meta.get("target_premium"),
                    "strike_selected_premium": strike_meta.get("selected_premium"),
                    "strike_premium_diff": strike_meta.get("premium_diff"),
                    "lifecycle_delta_peak_abs": trade.lifecycle_metrics.get("delta_peak_abs"),
                    "lifecycle_delta_drift": trade.lifecycle_metrics.get("delta_drift"),
                    "lifecycle_theta_cumulative": trade.lifecycle_metrics.get("theta_cumulative"),
                    "lifecycle_gamma_peak_abs": trade.lifecycle_metrics.get("gamma_peak_abs"),
                    "average_delta_exposure": trade.lifecycle_metrics.get("average_delta_exposure"),
                    "max_unrealized_drawdown": trade.lifecycle_metrics.get("max_unrealized_drawdown"),
                }
            )
        return pd.DataFrame(rows)

    def _emit_positions_dataframe(self, trades: List[ReplayTradeResult]) -> pd.DataFrame:
        rows = []
        for trade in trades:
            symbol = self._option_symbol(trade.contract)
            assignment_reason = trade.lifecycle_metrics.get("assignment_reason")
            for snapshot in trade.lifecycle:
                rows.append(
                    {
                        "trade_id": trade.equity_trade.trade_id,
                        "option_symbol": symbol,
                        "timestamp": snapshot.timestamp.isoformat(),
                        "option_price": snapshot.option_price,
                        "underlying_price": snapshot.underlying_price,
                        "dte": snapshot.dte,
                        "pricing_mode": snapshot.pricing_mode,
                        "position_value": snapshot.position_value,
                        "unrealized_pnl": snapshot.unrealized_pnl,
                        "delta_exposure": snapshot.delta_exposure,
                        "gamma_exposure": snapshot.gamma_exposure,
                        "theta_exposure": snapshot.theta_exposure,
                        **{f"greek_{k}": v for k, v in snapshot.greeks.items()},
                    }
                )
        return pd.DataFrame(rows)

    def _emit_health_cards(self, trades: List[ReplayTradeResult]) -> pd.DataFrame:
        rows = []
        for trade in trades:
            lifecycle = trade.lifecycle
            entry_snapshot = lifecycle[0] if lifecycle else None
            exit_snapshot = lifecycle[-1] if lifecycle else None
            duration_hours = (trade.exit.timestamp - trade.entry.timestamp).total_seconds() / 3600.0
            assignment_reason = trade.lifecycle_metrics.get("assignment_reason")
            rows.append(
                {
                    "trade_id": trade.equity_trade.trade_id,
                    "ticker": trade.equity_trade.ticker,
                    "entry_time": trade.entry.timestamp.isoformat(),
                    "exit_time": trade.exit.timestamp.isoformat(),
                    "duration_hours": duration_hours,
                    "entry_dte": entry_snapshot.dte if entry_snapshot else None,
                    "exit_dte": exit_snapshot.dte if exit_snapshot else None,
                    "realized_pnl": trade.realized_pnl,
                    "net_realized_pnl": trade.net_realized_pnl,
                    "max_drawdown_pct": trade.max_drawdown_pct,
                    "max_unrealized_drawdown": trade.lifecycle_metrics.get("max_unrealized_drawdown"),
                    "delta_peak_abs": trade.lifecycle_metrics.get("delta_peak_abs"),
                    "delta_drift": trade.lifecycle_metrics.get("delta_drift"),
                    "theta_cumulative": trade.lifecycle_metrics.get("theta_cumulative"),
                    "gamma_peak_abs": trade.lifecycle_metrics.get("gamma_peak_abs"),
                    "average_delta_exposure": trade.lifecycle_metrics.get("average_delta_exposure"),
                    "assignment_risk": trade.lifecycle_metrics.get("assignment_risk", False),
                    "assignment_reason": assignment_reason,
                    "assignment_intrinsic": trade.lifecycle_metrics.get("assignment_intrinsic"),
                    "assignment_dte_hours": trade.lifecycle_metrics.get("assignment_dte_hours"),
                    "pricing_fallbacks": ";".join(sorted(set(trade.pricing_fallbacks))),
                    "risk_flags": ";".join(sorted(set(trade.risk_flags))),
                    "forced_close": trade.forced_close,
                    "forced_close_reason": trade.forced_close_reason,
                    "forced_close_before_expiry": trade.expiry_forced_close,
                    "exit_override_reason": trade.exit_override_reason,
                    "exit_override_timestamp": trade.exit_override_timestamp.isoformat()
                    if trade.exit_override_timestamp
                    else None,
                    "exit_override_details": json.dumps(trade.exit_override_details, sort_keys=True, default=str)
                    if trade.exit_override_details
                    else None,
                }
            )
        return pd.DataFrame(rows)

    def _write_output(self, df: pd.DataFrame, filename: str) -> Optional[Path]:
        if df.empty:
            return None
        path = self.output_dir / filename
        compression = {"method": "gzip"} if self.config.output.compress else {}
        if filename.endswith(".csv"):
            df.to_csv(path, index=False, compression=compression or None)
        elif filename.endswith(".parquet"):
            df.to_parquet(path, index=False)
        else:
            raise ValueError(f"Unsupported output format for {filename}")
        self.hash_records[filename] = sha256(path.read_bytes()).hexdigest()
        return path

    def _write_json(self, payload: Dict[str, object], filename: str) -> Path:
        path = self.output_dir / filename
        path.write_text(json.dumps(payload, indent=2, default=str))
        self.hash_records[filename] = sha256(path.read_bytes()).hexdigest()
        return path

    def run(
        self,
        tickers: Optional[Sequence[str] | str],
        date_ranges: Sequence[str],
        verify_hash: bool = False,
    ) -> ReplayRunArtifacts:
        """
        Execute the replay engine for the provided tickers/date ranges.
        """
        run_start = datetime.utcnow()
        if self.control_board:
            self.control_board.mark_run_start(run_start)

        self._log_structured("info", "Starting replay run", run_id=self.run_id, config_hash=self.config.config_hash)
        resolved_tickers, ticker_selection_mode = self._normalise_tickers(tickers)
        self._log_structured(
            "info",
            "Replay scope resolved",
            tickers=resolved_tickers or "auto",
            ticker_mode=ticker_selection_mode,
            date_ranges=list(date_ranges),
        )

        trades = load_equity_trades(
            trades_path=self.config.inputs.equity_trades_path,
            tz=self.config.inputs.timezone,
            ticker_whitelist=self.config.inputs.ticker_whitelist,
        )
        filtered = self._filter_trades(trades, resolved_tickers, date_ranges)
        self._log_structured("info", "Filtered equity trades", total=len(filtered))

        trades_by_ticker: Dict[str, List[Tuple[object, str]]] = defaultdict(list)
        for trade, label in filtered:
            trades_by_ticker[trade.ticker].append((trade, label))

        ticker_order = sorted(trades_by_ticker.keys())
        contexts: Dict[str, TickerContext] = {}
        workers_used = 0
        if ticker_order:
            contexts, workers_used = self._prepare_ticker_contexts(
                {ticker: trades_by_ticker[ticker] for ticker in ticker_order}
            )
            self._log_structured(
                "info",
                "Context build complete",
                tickers=len(contexts),
                workers_used=workers_used,
                parallel_enabled=self.config.performance.parallel.enabled,
            )
        else:
            self._log_structured(
                "warning",
                "No trades available for replay",
                reason="empty_filter",
                tickers=resolved_tickers or "auto",
            )

        ordered_trades = sorted(filtered, key=lambda item: (item[0].entry_time, item[0].trade_id))
        processed: List[ReplayTradeResult] = []
        skipped: List[Dict[str, object]] = []
        risk_manager = RiskManager(self.config.risk)
        risk_events: List = []
        ticker_summary: Dict[str, Dict[str, object]] = defaultdict(
            lambda: {
                "processed": 0,
                "skipped": 0,
                "realized_pnl": 0.0,
                "net_realized_pnl": 0.0,
                "total_costs": 0.0,
                "lifecycle_metrics": [],
                "fallback_counts": Counter(),
            }
        )

        for equity_trade, label in ordered_trades:
            ticker = equity_trade.ticker
            summary = ticker_summary[ticker]
            context = contexts.get(ticker)
            if context is None:
                skipped.append(
                    {
                        "trade_id": equity_trade.trade_id,
                        "ticker": ticker,
                        "date_range": label,
                        "reason": "context_missing",
                    }
                )
                summary["skipped"] += 1
                self._log_structured(
                    "warning",
                    "trade_skipped",
                    trade_id=equity_trade.trade_id,
                    ticker=ticker,
                    reason="context_missing",
                )
                continue

            tz = self.config.inputs.timezone
            entry_date = equity_trade.entry_time.tz_convert(tz).date()
            exit_date = equity_trade.exit_time.tz_convert(tz).date()
            if context.coverage_start and entry_date < context.coverage_start:
                skipped.append(
                    {
                        "trade_id": equity_trade.trade_id,
                        "ticker": ticker,
                        "date_range": label,
                        "reason": "pre_coverage_window",
                        "coverage_start": context.coverage_start.isoformat(),
                    }
                )
                summary["skipped"] += 1
                self._log_structured(
                    "warning",
                    "trade_skipped",
                    trade_id=equity_trade.trade_id,
                    ticker=ticker,
                    reason="pre_coverage_window",
                    coverage_start=context.coverage_start.isoformat(),
                )
                continue
            if context.coverage_end and exit_date > context.coverage_end:
                skipped.append(
                    {
                        "trade_id": equity_trade.trade_id,
                        "ticker": ticker,
                        "date_range": label,
                        "reason": "post_coverage_window",
                        "coverage_end": context.coverage_end.isoformat(),
                    }
                )
                summary["skipped"] += 1
                self._log_structured(
                    "warning",
                    "trade_skipped",
                    trade_id=equity_trade.trade_id,
                    ticker=ticker,
                    reason="post_coverage_window",
                    coverage_end=context.coverage_end.isoformat(),
                )
                continue

            underlying_df = context.underlying_cache.get(label)
            if underlying_df is None:
                skipped.append(
                    {
                        "trade_id": equity_trade.trade_id,
                        "ticker": ticker,
                        "date_range": label,
                        "reason": "missing_underlying_slice",
                    }
                )
                summary["skipped"] += 1
                self._log_structured(
                    "warning",
                    "trade_skipped",
                    trade_id=equity_trade.trade_id,
                    ticker=ticker,
                    reason="missing_underlying_slice",
                )
                continue

            entry_row = underlying_df[underlying_df["timestamp"] >= equity_trade.entry_time]
            if entry_row.empty:
                skipped.append(
                    {
                        "trade_id": equity_trade.trade_id,
                        "ticker": ticker,
                        "date_range": label,
                        "reason": "no_underlying_bars",
                    }
                )
                summary["skipped"] += 1
                self._log_structured(
                    "warning",
                    "trade_skipped",
                    trade_id=equity_trade.trade_id,
                    ticker=ticker,
                    reason="no_underlying_bars",
                )
                continue
            entry_close = float(entry_row.iloc[0]["close"])

            try:
                mapping = map_trade_to_option(
                    config=self.config,
                    option_store=context.option_store,
                    trade=equity_trade,
                    underlying_entry_price=entry_close,
                )
            except Exception as exc:
                skipped.append(
                    {
                        "trade_id": equity_trade.trade_id,
                        "ticker": ticker,
                        "date_range": label,
                        "reason": "mapping_failed",
                        "details": str(exc),
                    }
                )
                summary["skipped"] += 1
                self._log_structured(
                    "error",
                    "trade_mapping_failed",
                    trade_id=equity_trade.trade_id,
                    ticker=ticker,
                    error=str(exc),
                )
                continue

            contract = mapping.contract
            lots = self.config.lot_sizing.fixed.lots_per_trade
            quantity = lots * contract.lot_size

            original_exit_time = equity_trade.exit_time
            exit_time = original_exit_time
            forced_close = False
            expiry_forced_close = False
            forced_close_reason: Optional[str] = None
            force_cfg = self.config.position_management.exit.force_close_before_expiry
            if force_cfg.enabled:
                expiry_cutoff = contract.expiry - pd.Timedelta(hours=force_cfg.hours_before)
                if exit_time > expiry_cutoff:
                    exit_time = expiry_cutoff
                    forced_close = True
                    expiry_forced_close = True
                    forced_close_reason = "expiry_cutoff"
            if exit_time <= equity_trade.entry_time:
                skipped.append(
                    {
                        "trade_id": equity_trade.trade_id,
                        "ticker": ticker,
                        "date_range": label,
                        "reason": "exit_before_entry",
                    }
                )
                summary["skipped"] += 1
                self._log_structured(
                    "warning",
                    "trade_skipped",
                    trade_id=equity_trade.trade_id,
                    ticker=ticker,
                    reason="exit_before_entry",
                )
                continue
            if exit_time < original_exit_time:
                forced_close = True
                if forced_close_reason is None:
                    forced_close_reason = "exit_signal"
                if forced_close_reason == "expiry_cutoff":
                    expiry_forced_close = True
                equity_trade.exit_time = exit_time

            slice_mask = (underlying_df["timestamp"] >= equity_trade.entry_time) & (
                underlying_df["timestamp"] <= exit_time
            )
            slice_df = underlying_df.loc[slice_mask]
            if slice_df.empty:
                skipped.append(
                    {
                        "trade_id": equity_trade.trade_id,
                        "ticker": ticker,
                        "date_range": label,
                        "reason": "no_underlying_bars",
                    }
                )
                summary["skipped"] += 1
                self._log_structured(
                    "warning",
                    "trade_skipped",
                    trade_id=equity_trade.trade_id,
                    ticker=ticker,
                    reason="no_underlying_bars",
                )
                continue

            entry_close = float(slice_df.iloc[0]["close"])
            try:
                entry_event, entry_fallback = context.pricer.price(
                    contract, equity_trade.entry_time, entry_close, allow_skip=True
                )
            except LiquidityFilterError as exc:
                skipped.append(
                    {
                        "trade_id": equity_trade.trade_id,
                        "ticker": ticker,
                        "date_range": label,
                        "reason": "liquidity_filter",
                        "details": exc.reasons,
                    }
                )
                summary["skipped"] += 1
                self._log_structured(
                    "warning",
                    "trade_skipped",
                    trade_id=equity_trade.trade_id,
                    ticker=ticker,
                    reason="liquidity_filter",
                    details=exc.reasons,
                )
                continue
            entry_leg_costs = context.pricer.compute_leg_costs(
                price=entry_event.price,
                quantity=quantity,
                side="buy",
            )
            entry_cost = entry_event.price * quantity
            allowed, reasons = risk_manager.evaluate_entry(
                equity_trade.trade_id,
                entry_cost,
                equity_trade.entry_time,
                ticker=ticker,
            )
            if not allowed:
                skipped.append(
                    {
                        "trade_id": equity_trade.trade_id,
                        "ticker": ticker,
                        "date_range": label,
                        "reason": "risk_rejection",
                        "details": reasons,
                    }
                )
                summary["skipped"] += 1
                self._log_structured(
                    "warning",
                    "trade_skipped",
                    trade_id=equity_trade.trade_id,
                    ticker=ticker,
                    reason="risk_rejection",
                    details=reasons,
                )
                continue

            risk_manager.register_entry(
                equity_trade.trade_id,
                entry_cost,
                quantity,
                equity_trade.entry_time,
                ticker=ticker,
            )
            timestamps = slice_df["timestamp"].tolist()
            prices = slice_df["close"].astype(float).tolist()
            lifecycle, fallbacks = context.pricer.price_path(
                contract=contract,
                timestamps=timestamps,
                underlying_prices=prices,
                include_greeks=self.config.greeks.calculate,
            )
            override_flag: Optional[str] = None
            override_reason: Optional[str] = None
            override_details: Optional[Dict[str, object]] = None
            override_timestamp, override_reason, override_details, override_index = self._resolve_exit_override(
                entry_event=entry_event,
                lifecycle=lifecycle,
                entry_time=equity_trade.entry_time,
                default_exit_time=exit_time,
            )
            if override_timestamp and override_timestamp < exit_time:
                forced_close = True
                override_flag = f"exit_{override_reason}" if override_reason else "exit_override"
                forced_close_reason = (
                    f"override_{override_reason}" if override_reason else "override"
                )
                expiry_forced_close = False
                exit_time = override_timestamp
                if exit_time < equity_trade.exit_time:
                    equity_trade.exit_time = exit_time
                if override_index is not None:
                    timestamps = timestamps[: override_index + 1]
                    prices = prices[: override_index + 1]
                    lifecycle = lifecycle[: override_index + 1]
                    slice_df = slice_df.iloc[: override_index + 1].copy()
                self._log_structured(
                    "info",
                    "exit_override_applied",
                    trade_id=equity_trade.trade_id,
                    ticker=ticker,
                    reason=override_reason,
                    timestamp=override_timestamp.isoformat(),
                )
            exit_underlying_price = prices[-1]
            exit_event, exit_fallback = context.pricer.price(
                contract, exit_time, exit_underlying_price, allow_skip=False
            )
            exit_leg_costs = context.pricer.compute_leg_costs(
                price=exit_event.price,
                quantity=quantity,
                side="sell",
            )
            exit_notes = dict(exit_event.notes)
            if override_reason:
                exit_notes["override_reason"] = override_reason
                if override_details:
                    exit_notes["override_details"] = override_details
                if override_timestamp:
                    exit_notes["override_timestamp"] = override_timestamp.isoformat()
            if forced_close:
                exit_notes["forced_close"] = True
                exit_notes["forced_close_reason"] = forced_close_reason or "exit_signal"
                exit_notes["forced_close_before_expiry"] = bool(expiry_forced_close)
            exit_event = replace(exit_event, notes=exit_notes)
            exit_snapshot = lifecycle[-1] if lifecycle else None
            assignment_flag, assignment_reason, assignment_intrinsic, assignment_dte_hours = _evaluate_assignment_risk(
                self.config.risk.assignment_risk,
                contract,
                exit_event,
                exit_snapshot,
            )
            realized_pnl = (exit_event.price - entry_event.price) * quantity
            return_pct = realized_pnl / entry_cost if entry_cost else 0.0
            mtm_curve: List[Dict[str, object]] = []
            position_values = []
            max_drawdown = 0.0
            peak_value = None
            delta_samples: List[float] = []
            gamma_samples: List[float] = []
            theta_samples: List[float] = []
            for snap in lifecycle:
                position_value = snap.option_price * quantity
                unrealized_pnl = position_value - entry_cost
                snap.position_value = position_value
                snap.unrealized_pnl = unrealized_pnl
                delta_exposure = snap.greeks.get("delta", 0.0) * quantity
                gamma_exposure = snap.greeks.get("gamma", 0.0) * quantity
                theta_exposure = snap.greeks.get("theta", 0.0) * quantity
                snap.delta_exposure = delta_exposure
                snap.gamma_exposure = gamma_exposure
                snap.theta_exposure = theta_exposure
                delta_samples.append(delta_exposure)
                gamma_samples.append(gamma_exposure)
                theta_samples.append(theta_exposure)
                position_values.append(position_value)
                mtm_curve.append(
                    {
                        "timestamp": snap.timestamp.isoformat(),
                        "position_value": position_value,
                        "unrealized_pnl": unrealized_pnl,
                        "delta_exposure": delta_exposure,
                        "gamma_exposure": gamma_exposure,
                        "theta_exposure": theta_exposure,
                    }
                )
                if peak_value is None:
                    peak_value = position_value
                else:
                    peak_value = max(peak_value, position_value)
                if peak_value and peak_value > 0:
                    drawdown = (position_value - peak_value) / peak_value
                    max_drawdown = min(max_drawdown, drawdown)
            max_drawdown_pct = max_drawdown * 100.0
            if forced_close and not forced_close_reason:
                forced_close_reason = "exit_signal"
            risk_flags = _compile_risk_flags(
                override_flag=override_flag,
                forced_close_reason=forced_close_reason,
                expiry_forced_close=expiry_forced_close,
                assignment_flag=bool(assignment_flag),
            )
            pricing_fallbacks = [fb for fb in fallbacks if fb] + ([entry_fallback] if entry_fallback else []) + (
                [exit_fallback] if exit_fallback else []
            )
            trade_costs = TradeCostSummary(entry=entry_leg_costs, exit=exit_leg_costs)
            total_costs = trade_costs.total
            net_realized_pnl = realized_pnl - total_costs
            entry_total_outlay = entry_cost + entry_leg_costs.total
            net_return_pct = net_realized_pnl / entry_total_outlay if entry_total_outlay else 0.0
            max_unrealized_drawdown = max_drawdown
            lifecycle_metrics = {
                "delta_peak_abs": max((abs(x) for x in delta_samples), default=0.0),
                "delta_drift": (max(delta_samples) - min(delta_samples)) if delta_samples else 0.0,
                "theta_cumulative": float(sum(theta_samples)),
                "gamma_peak_abs": max((abs(x) for x in gamma_samples), default=0.0),
                "average_delta_exposure": float(sum(delta_samples) / len(delta_samples)) if delta_samples else 0.0,
                "max_unrealized_drawdown": float(max_unrealized_drawdown),
                "assignment_risk": assignment_flag,
                "assignment_intrinsic": assignment_intrinsic,
                "assignment_dte_hours": assignment_dte_hours,
                "assignment_reason": assignment_reason,
            }
            risk_manager.record_lifecycle_metrics(
                ticker=ticker,
                delta_peak=lifecycle_metrics["delta_peak_abs"],
                delta_drift=lifecycle_metrics["delta_drift"],
                theta_cumulative=lifecycle_metrics["theta_cumulative"],
                gamma_peak=lifecycle_metrics["gamma_peak_abs"],
                unrealized_drawdown=lifecycle_metrics["max_unrealized_drawdown"],
                assignment_risk=assignment_flag,
                assignment_intrinsic=assignment_intrinsic,
                assignment_dte_hours=assignment_dte_hours,
                timestamp=exit_time,
                trade_id=equity_trade.trade_id,
            )

            ticker_events = risk_manager.register_exit(
                equity_trade.trade_id,
                realized_pnl=realized_pnl,
                timestamp=exit_time,
                ticker=ticker,
            )
            for event in ticker_events:
                event.details.setdefault("ticker", ticker)
            risk_events.extend(ticker_events)

            processed.append(
                ReplayTradeResult(
                    equity_trade=equity_trade,
                    contract=contract,
                    lots=lots,
                    quantity=quantity,
                    entry=entry_event,
                    exit=exit_event,
                    lifecycle=lifecycle,
                    realized_pnl=realized_pnl,
                    return_pct=return_pct * 100.0,
                    costs=trade_costs,
                    net_realized_pnl=net_realized_pnl,
                    net_return_pct=net_return_pct * 100.0,
                    max_drawdown_pct=max_drawdown_pct,
                    mtm_equity_curve=mtm_curve,
                    lifecycle_metrics=lifecycle_metrics,
                    risk_flags=risk_flags,
                    pricing_fallbacks=pricing_fallbacks,
                    mapping_metadata=mapping.metadata,
                    exit_override_reason=override_reason,
                    exit_override_details=override_details or {},
                    exit_override_timestamp=override_timestamp,
                    forced_close=forced_close,
                    forced_close_reason=forced_close_reason,
                    expiry_forced_close=expiry_forced_close,
                )
            )

            summary["processed"] += 1
            summary["realized_pnl"] += realized_pnl
            summary["net_realized_pnl"] += net_realized_pnl
            summary["total_costs"] += total_costs
            summary.setdefault("lifecycle_metrics", []).append(lifecycle_metrics)
            for fb in pricing_fallbacks:
                if fb:
                    summary["fallback_counts"][fb] += 1

            self._log_structured(
                "info",
                "trade_processed",
                trade_id=equity_trade.trade_id,
                ticker=ticker,
                realized_pnl=realized_pnl,
                net_realized_pnl=net_realized_pnl,
                transaction_costs=total_costs,
                pricing_mode_exit=exit_event.pricing_mode,
                fallbacks=len(pricing_fallbacks),
            )

            if risk_manager.kill_switch_triggered:
                self._log_structured(
                    "warning",
                    "kill_switch_engaged",
                    reason=risk_manager.kill_switch_reason,
                    trade_id=equity_trade.trade_id,
                    ticker=ticker,
                )
                break

        risk_manager.log_kill_switch_status()
        metrics = compute_replay_metrics(processed, self.config.risk.initial_portfolio_value)
        trades_df = self._emit_trade_dataframe(processed)
        positions_df = self._emit_positions_dataframe(processed)
        health_cards_df = self._emit_health_cards(processed)
        ticker_summary_payload: Dict[str, Dict[str, object]] = {}
        for ticker, data in ticker_summary.items():
            fallback_counts = data["fallback_counts"]
            lifecycle_metrics = data.get("lifecycle_metrics", [])
            delta_peaks = [metrics.get("delta_peak_abs", 0.0) for metrics in lifecycle_metrics]
            delta_drifts = [metrics.get("delta_drift", 0.0) for metrics in lifecycle_metrics]
            theta_cumulatives = [metrics.get("theta_cumulative", 0.0) for metrics in lifecycle_metrics]
            gamma_peaks = [metrics.get("gamma_peak_abs", 0.0) for metrics in lifecycle_metrics]
            assignment_counts = sum(1 for metrics in lifecycle_metrics if metrics.get("assignment_risk"))
            ticker_summary_payload[ticker] = {
                "processed": data["processed"],
                "skipped": data["skipped"],
                "realized_pnl": data["realized_pnl"],
                "net_realized_pnl": data["net_realized_pnl"],
                "total_costs": data["total_costs"],
                "average_delta_peak_abs": float(sum(delta_peaks) / len(delta_peaks)) if delta_peaks else 0.0,
                "average_delta_drift": float(sum(delta_drifts) / len(delta_drifts)) if delta_drifts else 0.0,
                "average_theta_cumulative": float(sum(theta_cumulatives) / len(theta_cumulatives)) if theta_cumulatives else 0.0,
                "average_gamma_peak_abs": float(sum(gamma_peaks) / len(gamma_peaks)) if gamma_peaks else 0.0,
                "max_unrealized_drawdown": float(min((metrics.get("max_unrealized_drawdown", 0.0) for metrics in lifecycle_metrics), default=0.0)),
                "assignment_risk_count": int(assignment_counts),
                "fallback_counts": dict(fallback_counts),
            }

        metrics_payload = {
            "summary": metrics.summary,
            "per_ticker": metrics.per_ticker,
            "ticker_summary": ticker_summary_payload,
            "diagnostics": metrics.diagnostics,
            "config_hash": self.config.config_hash,
            "run_id": self.run_id,
        }
        lifecycle_summary_df, lifecycle_summary_dict = _build_lifecycle_aggregates(health_cards_df)

        self._write_output(trades_df, "options_trades.csv")
        self._write_output(positions_df, "options_positions.csv")
        self._write_output(health_cards_df, "options_health_cards.csv")
        self._write_output(lifecycle_summary_df, "options_lifecycle_summary.csv")
        selection_summary = _build_selection_summary(trades_df)
        metrics_payload["selection_summary"] = selection_summary
        pricing_summary = _build_pricing_summary(trades_df)
        metrics_payload["pricing_summary"] = pricing_summary
        metrics_payload["lifecycle_summary"] = lifecycle_summary_dict
        self._write_json(metrics_payload, "options_metrics.json")

        risk_snapshot = risk_manager.summary()

        manifest = {
            "run_id": self.run_id,
            "config_hash": self.config.config_hash,
            "inputs": {
                "equity_trades_path": str(self.config.inputs.equity_trades_path),
                "equity_data_root": str(self.config.inputs.equity_data_root),
                "options_data_root": str(self.config.inputs.options_data_root),
                "date_ranges": list(date_ranges),
            },
            "tickers": {
                "selection_mode": ticker_selection_mode,
                "requested": resolved_tickers if resolved_tickers is not None else "auto",
                "processed": ticker_order,
            },
            "parallel": {
                "enabled": self.config.performance.parallel.enabled,
                "max_workers": self.config.performance.parallel.max_workers,
                "workers_used": workers_used,
            },
            "outputs": self.hash_records,
            "skipped_trades": skipped,
            "ticker_summary": ticker_summary_payload,
            "risk": risk_snapshot,
            "determinism": {
                "verify_hash": verify_hash,
            },
        }
        risk_event_payloads = [
            {
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "message": event.message,
                "details": event.details,
                "severity": event.severity,
            }
            for event in risk_events
        ]

        board_result: Optional[ControlBoardResult] = None
        if self.control_board:
            self.control_board.prepare_replay_payload(
                metrics_payload=metrics_payload,
                run_manifest=manifest,
                risk_snapshot=risk_snapshot,
                risk_events=risk_event_payloads,
                skipped_trades=skipped,
                hash_records=self.hash_records,
            )
            board_result = self.control_board.finalize(
                duration_seconds=(datetime.utcnow() - run_start).total_seconds(),
            )
            manifest["control_board"] = {
                "json_path": board_result.json_path,
                "markdown_path": board_result.markdown_path,
                "archive_path": board_result.archive_path,
                "json_sha256": board_result.json_sha256,
            }
            if board_result.json_path and board_result.json_sha256:
                self.hash_records[Path(board_result.json_path).name] = board_result.json_sha256

        self._write_json(manifest, "run_manifest.json")
        self._flush_structured_log()

        artefacts = ReplayRunArtifacts(
            run_id=self.run_id,
            config_hash=self.config.config_hash,
            trades=processed,
            risk_events=risk_events,
            skipped_trades=skipped,
            metadata={
                "metrics": metrics.summary,
                "metrics_per_ticker": metrics.per_ticker,
                "metrics_diagnostics": metrics.diagnostics,
                "hashes": self.hash_records,
                "output_dir": str(self.output_dir),
                "ticker_summary": ticker_summary_payload,
                "risk": risk_snapshot,
                "determinism": {"verify_hash": verify_hash},
            },
        )
        if board_result:
            artefacts.metadata["control_board"] = {
                "json_path": board_result.json_path,
                "markdown_path": board_result.markdown_path,
                "archive_path": board_result.archive_path,
                "json_sha256": board_result.json_sha256,
            }

        run_end = datetime.utcnow()
        if self.control_board:
            self.control_board.mark_run_end(run_end)
        if verify_hash:
            reference_path = self.config.output.output_dir / "previous_hash.json"
            if reference_path.exists():
                reference = json.loads(reference_path.read_text())
                if reference != self.hash_records:
                    raise ValueError("Replay output hashes do not match reference (determinism failure)")
            reference_path.write_text(json.dumps(self.hash_records, indent=2))

        self._log_structured(
            "info",
            "Replay run complete",
            total_trades=len(processed),
            skipped=len(skipped),
            tickers_processed=ticker_order,
            workers_used=workers_used,
            kill_switch=risk_manager.kill_switch_triggered,
            output_dir=str(self.output_dir),
        )
        return artefacts


def _build_selection_summary(trades_df: pd.DataFrame) -> Dict[str, object]:
    """Aggregate strike/expiry selection outcomes for reporting."""
    summary: Dict[str, object] = {}

    if trades_df.empty:
        return summary

    def _count(series: pd.Series) -> Dict[str, int]:
        counts = series.value_counts(dropna=False)
        return {str(k): int(v) for k, v in counts.items()}

    summary["strike_method_counts"] = _count(trades_df["strike_selection_method"])
    summary["strike_status_counts"] = _count(trades_df["strike_selection_status"])
    summary["expiry_method_counts"] = _count(trades_df["expiry_selection_method"])
    summary["expiry_type_counts"] = _count(trades_df["expiry_type"])

    diff_series = pd.to_numeric(trades_df["strike_delta_diff"], errors="coerce")
    if diff_series.notna().any():
        summary["strike_delta_diff_stats"] = {
            "mean": float(diff_series.mean()),
            "std": float(diff_series.std(ddof=0)),
            "max": float(diff_series.max()),
            "min": float(diff_series.min()),
        }

    fallback_series = trades_df["strike_fallback_reason"].dropna()
    if not fallback_series.empty:
        summary["strike_fallback_reasons"] = _count(fallback_series)

    premium_series = pd.to_numeric(trades_df["strike_premium_diff"], errors="coerce")
    if premium_series.notna().any():
        summary["strike_premium_diff_stats"] = {
            "mean": float(premium_series.mean()),
            "std": float(premium_series.std(ddof=0)),
            "max": float(premium_series.max()),
            "min": float(premium_series.min()),
        }

    expiry_reason_series = trades_df["expiry_selection_reason"].dropna()
    if not expiry_reason_series.empty:
        summary["expiry_fallback_reasons"] = _count(expiry_reason_series)

    return summary


def _build_pricing_summary(trades_df: pd.DataFrame) -> Dict[str, object]:
    """Aggregate pricing integrity diagnostics."""
    summary: Dict[str, object] = {}
    if trades_df.empty:
        return summary

    def _count(series: pd.Series) -> Dict[str, int]:
        counts = series.value_counts(dropna=False)
        return {str(k): int(v) for k, v in counts.items()}

    summary["entry_pricing_modes"] = _count(trades_df["pricing_mode_entry"])
    summary["exit_pricing_modes"] = _count(trades_df["pricing_mode_exit"])

    entry_fallbacks = trades_df["entry_fallback_reason"].dropna()
    exit_fallbacks = trades_df["exit_fallback_reason"].dropna()
    combined_fallbacks: List[str] = []
    for series in (entry_fallbacks, exit_fallbacks):
        for value in series:
            if value:
                combined_fallbacks.extend(str(value).split(";"))
    pricing_fallbacks = trades_df["pricing_fallbacks"].dropna()
    for value in pricing_fallbacks:
        if value:
            combined_fallbacks.extend(str(value).split(";"))
    combined_fallbacks = [item.strip() for item in combined_fallbacks if item and item.strip()]
    if combined_fallbacks:
        summary["fallback_reasons"] = _count(pd.Series(combined_fallbacks))

    summary["average_transaction_cost_total"] = float(
        pd.to_numeric(trades_df["total_transaction_costs"], errors="coerce").mean()
    )
    summary["average_slippage_cost"] = float(
        pd.to_numeric(trades_df["total_slippage_costs"], errors="coerce").mean()
    )
    summary["gross_pnl_total"] = float(pd.to_numeric(trades_df["realized_pnl"], errors="coerce").sum())
    summary["net_pnl_total"] = float(pd.to_numeric(trades_df["net_realized_pnl"], errors="coerce").sum())
    summary["average_net_return_pct"] = float(
        pd.to_numeric(trades_df["net_return_pct"], errors="coerce").mean()
    )
    summary["average_calibration_error_entry"] = float(
        pd.to_numeric(trades_df["entry_calibration_error"], errors="coerce").mean()
    )
    summary["average_calibration_error_exit"] = float(
        pd.to_numeric(trades_df["exit_calibration_error"], errors="coerce").mean()
    )

    if "forced_close" in trades_df.columns:
        forced_series = trades_df["forced_close"].fillna(False).astype(bool)
        forced_count = int(forced_series.sum())
        total_rows = len(trades_df)
        summary["forced_close_count"] = forced_count
        summary["forced_close_share_pct"] = float((forced_count / total_rows) * 100.0) if total_rows else 0.0
        if "forced_close_before_expiry" in trades_df.columns:
            summary["forced_close_before_expiry_count"] = int(
                trades_df.loc[forced_series, "forced_close_before_expiry"].fillna(False).astype(bool).sum()
            )
        if forced_count and "forced_close_reason" in trades_df.columns:
            reason_series = trades_df.loc[forced_series, "forced_close_reason"].dropna()
            if not reason_series.empty:
                summary["forced_close_by_reason"] = _count(reason_series)

    if "exit_override_reason" in trades_df.columns:
        override_series = trades_df["exit_override_reason"].dropna()
        override_count = int(override_series.shape[0])
        total_rows = len(trades_df)
        summary["exit_override_count"] = override_count
        summary["exit_override_share_pct"] = float((override_count / total_rows) * 100.0) if total_rows else 0.0
        if override_count:
            summary["exit_override_reason_counts"] = _count(override_series)

    return summary


def _evaluate_assignment_risk(
    assignment_cfg,
    contract: OptionContractSpec,
    exit_event: PricingEvent,
    exit_snapshot: Optional[OptionPositionSnapshot],
) -> Tuple[bool, Optional[str], float, float]:
    if not assignment_cfg.enabled:
        return False, None, 0.0, float("inf")

    exit_notes = exit_event.notes or {}
    time_to_expiry_days = exit_notes.get("time_to_expiry_days")
    if time_to_expiry_days is None:
        return False, None, 0.0, float("inf")

    dte_hours = float(time_to_expiry_days) * 24.0
    if dte_hours > assignment_cfg.dte_hours_threshold:
        return False, None, 0.0, dte_hours

    underlying_price = exit_event.underlying_price
    if underlying_price is None and exit_snapshot is not None:
        underlying_price = exit_snapshot.underlying_price
    if underlying_price is None:
        return False, None, 0.0, dte_hours

    intrinsic = 0.0
    if contract.option_type.value.upper() == "CE":
        intrinsic = float(underlying_price) - float(contract.strike)
    else:
        intrinsic = float(contract.strike) - float(underlying_price)

    if intrinsic > assignment_cfg.intrinsic_buffer:
        reason = f"dte_{dte_hours:.2f}h_intrinsic_{intrinsic:.2f}"
        return True, reason, intrinsic, dte_hours

    return False, None, intrinsic, dte_hours


def _build_lifecycle_aggregates(health_cards_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if health_cards_df.empty:
        return pd.DataFrame(), {}

    summary: Dict[str, object] = {}
    total_rows = len(health_cards_df)
    health_cards_df = health_cards_df.copy()

    def _stats(series: pd.Series) -> Dict[str, float]:
        series = pd.to_numeric(series, errors="coerce").dropna()
        if series.empty:
            return {}
        return {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "max": float(series.max()),
            "min": float(series.min()),
        }

    # Overall stats
    summary["duration_hours"] = _stats(health_cards_df["duration_hours"])
    summary["delta_peak_abs"] = _stats(health_cards_df["delta_peak_abs"])
    summary["theta_cumulative"] = _stats(health_cards_df["theta_cumulative"])
    summary["gamma_peak_abs"] = _stats(health_cards_df["gamma_peak_abs"])
    summary["max_unrealized_drawdown"] = _stats(health_cards_df["max_unrealized_drawdown"])

    bins = [0, 2, 4, 8, 24, 48, float("inf")]
    labels = ["<=2h", "2-4h", "4-8h", "8-24h", "24-48h", ">48h"]
    duration_hist = pd.cut(health_cards_df["duration_hours"], bins=bins, labels=labels, right=False)
    summary["duration_histogram"] = {label: int((duration_hist == label).sum()) for label in labels}

    forced_series = health_cards_df.get("forced_close")
    if forced_series is not None:
        forced_mask = forced_series.fillna(False).astype(bool)
        forced_count = int(forced_mask.sum())
        summary["forced_close_count"] = forced_count
        summary["forced_close_share_pct"] = float((forced_count / total_rows) * 100.0) if total_rows else 0.0
        if "forced_close_before_expiry" in health_cards_df.columns:
            summary["forced_close_before_expiry_count"] = int(
                health_cards_df.loc[forced_mask, "forced_close_before_expiry"].fillna(False).astype(bool).sum()
            )
        if forced_count and "forced_close_reason" in health_cards_df.columns:
            reasons = health_cards_df.loc[forced_mask, "forced_close_reason"].dropna().astype(str)
            if not reasons.empty:
                summary["forced_close_by_reason"] = dict(Counter(reasons))
        health_cards_df["forced_close_int"] = forced_mask.astype(int)
    else:
        health_cards_df["forced_close_int"] = 0

    override_series = health_cards_df.get("exit_override_reason")
    if override_series is not None:
        override_mask = override_series.fillna("").astype(str).str.len() > 0
        override_count = int(override_mask.sum())
        summary["exit_override_count"] = override_count
        summary["exit_override_share_pct"] = float((override_count / total_rows) * 100.0) if total_rows else 0.0
        if override_count:
            reasons = health_cards_df.loc[override_mask, "exit_override_reason"].dropna().astype(str)
            if not reasons.empty:
                summary["exit_override_reason_counts"] = dict(Counter(reasons))
        health_cards_df["exit_override_int"] = override_mask.astype(int)
    else:
        health_cards_df["exit_override_int"] = 0

    assignment_series = health_cards_df.get("assignment_risk")
    if assignment_series is not None:
        assignment_int = assignment_series.fillna(False).astype(int)
        summary["assignment_risk_count"] = int(assignment_int.sum())
        summary["assignment_risk_by_ticker"] = (
            assignment_int.groupby(health_cards_df["ticker"]).sum().to_dict()
        )
        health_cards_df["assignment_risk_int"] = assignment_int
    else:
        health_cards_df["assignment_risk_int"] = 0

    # Per ticker aggregation
    agg_columns = {
        "duration_hours": ["mean", "max"],
        "delta_peak_abs": ["mean", "max"],
        "delta_drift": ["mean", "max"],
        "theta_cumulative": ["mean", "sum"],
        "gamma_peak_abs": ["mean", "max"],
        "max_unrealized_drawdown": ["min"],
        "realized_pnl": ["sum"],
        "net_realized_pnl": ["sum"],
        "assignment_risk_int": ["sum"],
        "forced_close_int": ["sum"],
        "exit_override_int": ["sum"],
    }
    grouped = health_cards_df.groupby("ticker").agg(agg_columns)
    grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
    summary_df = grouped.reset_index()
    if "assignment_risk_int_sum" in summary_df.columns:
        summary_df = summary_df.rename(columns={"assignment_risk_int_sum": "assignment_risk_count"})
    if "forced_close_int_sum" in summary_df.columns:
        summary_df = summary_df.rename(columns={"forced_close_int_sum": "forced_close_count"})
    if "exit_override_int_sum" in summary_df.columns:
        summary_df = summary_df.rename(columns={"exit_override_int_sum": "exit_override_count"})

    # Risk flag counts
    flags = []
    for value in health_cards_df.get("risk_flags", []):
        if value:
            flags.extend(flag for flag in str(value).split(";") if flag)
    if flags:
        summary["risk_flag_counts"] = {flag: flags.count(flag) for flag in set(flags)}

    return summary_df, summary
