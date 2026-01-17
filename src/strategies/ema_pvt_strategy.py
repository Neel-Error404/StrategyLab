"""
EMA + Pivot Strategy
====================

Implements the EMA crossover with pivot-level confirmations described in the
shared spec. The strategy relies on a single timeframe (default 5m) and
leverages the declarative indicator stack so EMA/Pivot variations can be
configured entirely via YAML.

Signal Philosophy
-----------------
- Short bias (default): trigger when the fast EMA crosses below the slow EMA
  and the previous close is below S1. Target S3, stop at the entry candle high.
- Long bias (optional): mirror logic using R1 for confirmation and R3 for the
  target with the entry candle low serving as the stop.

All indicator references are routed through the indicator_columns map so
renaming indicators in YAML does not require code changes, and `.shift(1)`
guards protect against look-ahead bias.
"""

from __future__ import annotations

from datetime import time
from typing import Any, Dict, Optional, Union

import pandas as pd

from .support.strategy_base import StrategyBase


class EmaPivotStrategy(StrategyBase):
    """Single-timeframe EMA crossover + pivot strategy."""

    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None, config=None):
        params = parameters or {}
        super().__init__(name=name, parameters=params, config=config)

        self.entry_timeframe = params.get("entry_timeframe", "5m")
        self.required_timeframes = [self.entry_timeframe]

        self.ema_fast_period = int(params.get("ema_fast_period", 20))
        self.ema_slow_period = int(params.get("ema_slow_period", 30))
        self.volume_column = params.get("volume_column", "volume")
        self.min_volume = float(params.get("min_volume", 0))
        self.bias = str(params.get("bias", "both")).lower()

        warmup_default = int(params.get("warmup_bars", max(self.ema_slow_period + 5, 60)))
        self.warmup_periods = {self.entry_timeframe: warmup_default}

        cutoff = params.get("intraday_cutoff", "15:15")
        self.market_close = self._parse_time(cutoff)

        default_long = self.bias in {"long_only", "long", "both"}
        default_short = self.bias not in {"long_only", "long"}
        self.enable_long = bool(params.get("enable_long", default_long))
        self.enable_short = bool(params.get("enable_short", default_short))

        indicator_columns = params.get("indicator_columns") or {}
        prefix = f"{self.entry_timeframe}_"
        self.indicator_columns = {
            "ema_fast": indicator_columns.get("ema_fast", f"{prefix}ema_{self.ema_fast_period}"),
            "ema_slow": indicator_columns.get("ema_slow", f"{prefix}ema_{self.ema_slow_period}"),
            "pivot_point": indicator_columns.get("pivot_point", f"{prefix}pivot_point"),
            "pivot_s1": indicator_columns.get("pivot_s1", f"{prefix}pivot_s1"),
            "pivot_s3": indicator_columns.get("pivot_s3", f"{prefix}pivot_s3"),
            "pivot_r1": indicator_columns.get("pivot_r1", f"{prefix}pivot_r1"),
            "pivot_r3": indicator_columns.get("pivot_r3", f"{prefix}pivot_r3"),
        }

    # ------------------------------------------------------------------ #
    # Core pipeline
    # ------------------------------------------------------------------ #
    def prepare_data(
        self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], ticker: str, pull_date: str
    ) -> Dict[str, pd.DataFrame]:
        frames = self._ensure_timeframe_dict(data)
        frames = self._apply_configured_indicators(frames)
        self._sync_warmup_with_indicator_registry()

        prepared: Dict[str, pd.DataFrame] = {}
        for timeframe, frame in frames.items():
            if timeframe not in self.required_timeframes or frame is None or frame.empty:
                continue
            normalized = frame.copy()
            normalized["timestamp"] = pd.to_datetime(normalized["timestamp"])
            normalized.sort_values("timestamp", inplace=True)
            normalized.reset_index(drop=True, inplace=True)
            warmup = self.warmup_periods.get(timeframe, 0)
            if warmup > 0 and len(normalized) > warmup:
                normalized = normalized.iloc[warmup:].reset_index(drop=True)
            prepared[timeframe] = normalized
        return prepared

    def generate_signals(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        frames = self._ensure_timeframe_dict(data)
        df = frames.get(self.entry_timeframe)

        if df is None or df.empty:
            raise ValueError(f"Entry timeframe {self.entry_timeframe} missing for {self.name}")

        working = df.copy()
        self._ensure_indicator_columns(working)

        for column in (
            "entry_signal_buy",
            "entry_signal_sell",
            "exit_signal_buy",
            "exit_signal_sell",
        ):
            if column not in working.columns:
                working[column] = False

        for column in (
            "stop_price_buy",
            "stop_price_sell",
            "target_price_buy",
            "target_price_sell",
            "exit_reason_buy",
            "exit_reason_sell",
        ):
            if column not in working.columns:
                working[column] = pd.NA

        session_ok = self._session_mask(working)
        volume_ok = self._volume_mask(working)
        guard = session_ok & volume_ok

        ema_fast = working[self.indicator_columns["ema_fast"]]
        ema_slow = working[self.indicator_columns["ema_slow"]]
        close = working["close"]
        pivot_s1 = working[self.indicator_columns["pivot_s1"]]
        pivot_s3 = working[self.indicator_columns["pivot_s3"]]
        pivot_r1 = working[self.indicator_columns["pivot_r1"]]
        pivot_r3 = working[self.indicator_columns["pivot_r3"]]

        if self.enable_short:
            bearish_cross = (ema_fast.shift(1) > ema_slow.shift(1)) & (ema_fast <= ema_slow)
            # Use prior-bar pivot levels against current close so breaks below support can trigger
            price_below_s1 = close < pivot_s1.shift(1)
            valid_levels = pivot_s1.shift(1).notna() & pivot_s3.shift(1).notna()
            short_entries = guard & bearish_cross & price_below_s1 & valid_levels
            working.loc[short_entries.fillna(False), "entry_signal_sell"] = True

        if self.enable_long:
            bullish_cross = (ema_fast.shift(1) < ema_slow.shift(1)) & (ema_fast >= ema_slow)
            # Use prior-bar pivot levels against current close so breaks above resistance can trigger
            price_above_r1 = close > pivot_r1.shift(1)
            valid_levels = pivot_r1.shift(1).notna() & pivot_r3.shift(1).notna()
            long_entries = guard & bullish_cross & price_above_r1 & valid_levels
            working.loc[long_entries.fillna(False), "entry_signal_buy"] = True

        working = self._apply_exit_logic(working)
        return working

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _ensure_timeframe_dict(
        self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    ) -> Dict[str, pd.DataFrame]:
        if isinstance(data, dict):
            return data
        return {self.entry_timeframe: data}

    def _sync_warmup_with_indicator_registry(self) -> None:
        if not self.indicator_registry:
            return
        registry_warmup = max(self.indicator_registry.required_warmup, 0)
        if registry_warmup <= 0:
            return
        current = self.warmup_periods.get(self.entry_timeframe, 0)
        self.warmup_periods[self.entry_timeframe] = max(current, registry_warmup)

    def _session_mask(self, df: pd.DataFrame) -> pd.Series:
        if self.market_close is None:
            return pd.Series(True, index=df.index)
        return df["timestamp"].dt.time < self.market_close

    def _volume_mask(self, df: pd.DataFrame) -> pd.Series:
        if self.min_volume <= 0 or self.volume_column not in df.columns:
            return pd.Series(True, index=df.index)
        return df[self.volume_column] >= self.min_volume

    def _ensure_indicator_columns(self, df: pd.DataFrame) -> None:
        missing = [col for col in self.indicator_columns.values() if col not in df.columns]
        if missing:
            raise ValueError(f"EmaPivotStrategy missing required indicator columns: {missing}")

    def _apply_exit_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for column in ("exit_signal_buy", "exit_signal_sell"):
            if column not in df.columns:
                df[column] = False

        in_long = False
        in_short = False
        long_stop: Optional[float] = None
        long_target: Optional[float] = None
        short_stop: Optional[float] = None
        short_target: Optional[float] = None

        for idx in range(len(df)):
            timestamp = df.at[idx, "timestamp"]
            current_time = timestamp.time() if isinstance(timestamp, pd.Timestamp) else None

            if self.market_close and current_time and current_time >= self.market_close:
                if in_long:
                    df.at[idx, "exit_signal_buy"] = True
                    df.at[idx, "exit_reason_buy"] = "intraday_square_off"
                    in_long = False
                if in_short:
                    df.at[idx, "exit_signal_sell"] = True
                    df.at[idx, "exit_reason_sell"] = "intraday_square_off"
                    in_short = False
                continue

            if (
                self.enable_long
                and df.at[idx, "entry_signal_buy"]
                and not in_long
                and not in_short
            ):
                target = df.at[idx, self.indicator_columns["pivot_r3"]]
                if pd.isna(target):
                    continue
                in_long = True
                long_stop = df.at[idx, "low"]
                long_target = target
                df.at[idx, "stop_price_buy"] = long_stop
                df.at[idx, "target_price_buy"] = long_target
                continue

            if in_long:
                exit_reason = self._evaluate_long_exit(df.iloc[idx], long_stop, long_target)
                if exit_reason:
                    df.at[idx, "exit_signal_buy"] = True
                    df.at[idx, "exit_reason_buy"] = exit_reason
                    in_long = False
                    long_stop = None
                    long_target = None
                continue

            if (
                self.enable_short
                and df.at[idx, "entry_signal_sell"]
                and not in_long
                and not in_short
            ):
                target = df.at[idx, self.indicator_columns["pivot_s3"]]
                if pd.isna(target):
                    continue
                in_short = True
                short_stop = df.at[idx, "high"]
                short_target = target
                df.at[idx, "stop_price_sell"] = short_stop
                df.at[idx, "target_price_sell"] = short_target
                continue

            if in_short:
                exit_reason = self._evaluate_short_exit(df.iloc[idx], short_stop, short_target)
                if exit_reason:
                    df.at[idx, "exit_signal_sell"] = True
                    df.at[idx, "exit_reason_sell"] = exit_reason
                    in_short = False
                    short_stop = None
                    short_target = None

        return df

    @staticmethod
    def _evaluate_long_exit(row: pd.Series, stop_price: Optional[float], target_price: Optional[float]) -> Optional[str]:
        if stop_price is not None and pd.notna(stop_price) and row["low"] <= stop_price:
            return "stop_loss"
        if target_price is not None and pd.notna(target_price) and row["high"] >= target_price:
            return "target_hit"
        return None

    @staticmethod
    def _evaluate_short_exit(row: pd.Series, stop_price: Optional[float], target_price: Optional[float]) -> Optional[str]:
        if stop_price is not None and pd.notna(stop_price) and row["high"] >= stop_price:
            return "stop_loss"
        if target_price is not None and pd.notna(target_price) and row["low"] <= target_price:
            return "target_hit"
        return None

    @staticmethod
    def _parse_time(value: Optional[str]) -> Optional[time]:
        if not value:
            return None
        try:
            hour, minute = value.split(":")
            return time(int(hour), int(minute))
        except Exception:
            return None

    def get_strategy_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": "Trend + Pivot Intraday",
            "description": (
                "Single-timeframe EMA crossover strategy that confirms entries with "
                "classical pivot levels (S1/R1) and targets extended levels (S3/R3)."
            ),
            "parameters": {
                "entry_timeframe": self.entry_timeframe,
                "ema_fast_period": self.ema_fast_period,
                "ema_slow_period": self.ema_slow_period,
                "bias": self.bias,
                "intraday_cutoff": self.market_close.isoformat(timespec="minutes") if self.market_close else "none",
            },
            "timeframes": self.required_timeframes,
        }
