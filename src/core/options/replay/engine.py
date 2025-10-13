"""
High-level orchestration for the Phase 3 options replay engine.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.core.options.replay.config import OptionsReplayConfig
from src.core.options.replay.models import ReplayTradeResult, ReplayRunArtifacts
from src.core.options.replay.data_loader import (
    OptionDataStore,
    find_option_data_dir,
    load_equity_trades,
    load_underlying_data,
    parse_date_range,
)
from src.core.options.replay.pricing import HybridPricingEngine
from src.core.options.replay.trade_mapper import map_trade_to_option
from src.core.options.replay.metrics import compute_replay_metrics
from src.core.options.replay.risk import RiskManager


@dataclass
class TickerContext:
    """Container for per-ticker replay artefacts."""

    ticker: str
    pricer: HybridPricingEngine
    underlying_cache: Dict[str, pd.DataFrame]
    option_store: OptionDataStore
    date_ranges: Sequence[str]


class OptionsReplayEngine:
    """
    End-to-end orchestrator that consumes equity trades, maps them to
    options, applies hybrid pricing, enforces risk, and emits artefacts.
    """

    def __init__(self, config: OptionsReplayConfig, logger: Optional[logging.Logger] = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger("OptionsReplayEngine")
        self.logger.setLevel(getattr(logging, config.logging.level.upper(), logging.INFO))
        self.run_id = self._build_run_id()
        self.output_dir = self._prepare_output_dir()
        self.hash_records: Dict[str, str] = {}
        self.structured_log_path = self.output_dir / "logs.jsonl"
        self._structured_log_buffer: List[Dict[str, object]] = []

        random.seed(config.inputs.seed)
        np.random.seed(config.inputs.seed)

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
                    contexts[ticker] = TickerContext(
                        ticker=ticker,
                        pricer=pricer,
                        underlying_cache=underlying_cache,
                        option_store=option_store,
                        date_ranges=date_ranges,
                    )
                    self._log_structured(
                        "info",
                        "ticker_context_ready",
                        ticker=ticker,
                        date_ranges=date_ranges,
                        parallel_mode="threaded",
                    )
        else:
            for ticker, date_ranges in tasks:
                pricer, underlying_cache, option_store = self._build_pricers(ticker, date_ranges)
                contexts[ticker] = TickerContext(
                    ticker=ticker,
                    pricer=pricer,
                    underlying_cache=underlying_cache,
                    option_store=option_store,
                    date_ranges=date_ranges,
                )
                self._log_structured(
                    "info",
                    "ticker_context_ready",
                    ticker=ticker,
                    date_ranges=date_ranges,
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

    def _option_symbol(self, contract) -> str:
        expiry = contract.expiry.tz_convert(self.config.inputs.timezone)
        expiry_code = expiry.strftime("%y%m%d")
        strike_code = f"{int(round(contract.strike)):05d}"
        return f"{contract.ticker}{expiry_code}{strike_code}{contract.option_type.value}"

    def _emit_trade_dataframe(self, trades: List[ReplayTradeResult]) -> pd.DataFrame:
        rows = []
        for trade in trades:
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
                    "realized_pnl": trade.realized_pnl,
                    "return_pct": trade.return_pct,
                    "max_drawdown_pct": trade.max_drawdown_pct,
                    "risk_flags": ";".join(trade.risk_flags),
                    "pricing_fallbacks": ";".join(set(trade.pricing_fallbacks)),
                }
            )
        return pd.DataFrame(rows)

    def _emit_positions_dataframe(self, trades: List[ReplayTradeResult]) -> pd.DataFrame:
        rows = []
        for trade in trades:
            symbol = self._option_symbol(trade.contract)
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
                        **{f"greek_{k}": v for k, v in snapshot.greeks.items()},
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
            lambda: {"processed": 0, "skipped": 0, "realized_pnl": 0.0, "fallback_counts": Counter()}
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
            force_cfg = self.config.position_management.exit.force_close_before_expiry
            if force_cfg.enabled:
                expiry_cutoff = contract.expiry - pd.Timedelta(hours=force_cfg.hours_before)
                if exit_time > expiry_cutoff:
                    exit_time = expiry_cutoff
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
            entry_event, entry_fallback = context.pricer.price(contract, equity_trade.entry_time, entry_close)
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
            exit_underlying_price = prices[-1]
            exit_event, exit_fallback = context.pricer.price(contract, exit_time, exit_underlying_price)
            realized_pnl = (exit_event.price - entry_event.price) * quantity
            return_pct = realized_pnl / entry_cost if entry_cost else 0.0
            position_values = [snap.option_price * quantity for snap in lifecycle]
            max_drawdown = 0.0
            if position_values:
                peak_value = position_values[0]
                for value in position_values:
                    peak_value = max(peak_value, value)
                    if peak_value > 0:
                        drawdown = (value - peak_value) / peak_value
                        max_drawdown = min(max_drawdown, drawdown)
            max_drawdown_pct = max_drawdown * 100.0
            risk_flags: List[str] = []
            if forced_close:
                risk_flags.append("force_closed_before_expiry")
            pricing_fallbacks = [fb for fb in fallbacks if fb] + ([entry_fallback] if entry_fallback else []) + (
                [exit_fallback] if exit_fallback else []
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
                    max_drawdown_pct=max_drawdown_pct,
                    risk_flags=risk_flags,
                    pricing_fallbacks=pricing_fallbacks,
                )
            )

            summary["processed"] += 1
            summary["realized_pnl"] += realized_pnl
            for fb in pricing_fallbacks:
                if fb:
                    summary["fallback_counts"][fb] += 1

            self._log_structured(
                "info",
                "trade_processed",
                trade_id=equity_trade.trade_id,
                ticker=ticker,
                realized_pnl=realized_pnl,
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

        metrics = compute_replay_metrics(processed, self.config.risk.initial_portfolio_value)
        trades_df = self._emit_trade_dataframe(processed)
        positions_df = self._emit_positions_dataframe(processed)
        ticker_summary_payload: Dict[str, Dict[str, object]] = {}
        for ticker, data in ticker_summary.items():
            fallback_counts = data["fallback_counts"]
            ticker_summary_payload[ticker] = {
                "processed": data["processed"],
                "skipped": data["skipped"],
                "realized_pnl": data["realized_pnl"],
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
        self._write_output(trades_df, "options_trades.csv")
        self._write_output(positions_df, "options_positions.csv")
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
