"""
Bollinger Band Squeeze Breakout Strategy
=======================================

Implements the Bollinger squeeze breakout play: wait for a low-volatility
window (bands tight), then enter long when price closes above the upper band,
exiting when price drops below the middle band. Default bias is long-only to
match the "during uptrend" recommendation, but the logic can be mirrored for
shorts via parameters.
"""

from __future__ import annotations

from datetime import time
from typing import Any, Dict, Optional, Union

import pandas as pd

from .support.strategy_base import StrategyBase


class BollingerSqueezeStrategy(StrategyBase):
    """Single-timeframe Bollinger band squeeze breakout strategy."""

    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None, config=None):
        params = parameters or {}
        super().__init__(name=name, parameters=params, config=config)

        self.entry_timeframe = params.get("entry_timeframe", "5m")
        self.required_timeframes = [self.entry_timeframe]
        self.warmup_periods = {self.entry_timeframe: int(params.get("warmup_bars", 60))}

        self.period = int(params.get("bollinger_period", 20))
        self.std_dev = float(params.get("bollinger_std_dev", 2.0))

        # Percentile-based squeeze detection (normalized width)
        self.squeeze_lookback = int(params.get("squeeze_lookback", 100))
        self.squeeze_percentile = float(params.get("squeeze_percentile", 0.20))

        cutoff = params.get("intraday_cutoff", "15:15")
        self.market_close = self._parse_time(cutoff)
        self.volume_column = params.get("volume_column", "volume")
        self.min_volume = float(params.get("min_volume", 0))

        self.enable_short = bool(params.get("enable_short", True))

        indicator_columns = params.get("indicator_columns") or {}
        prefix = f"{self.entry_timeframe}_"
        self.indicator_columns = {
            "upper": indicator_columns.get("upper_band", f"{prefix}bollinger_upper"),
            "middle": indicator_columns.get("middle_band", f"{prefix}bollinger_middle"),
            "lower": indicator_columns.get("lower_band", f"{prefix}bollinger_lower"),
            "width": indicator_columns.get("bandwidth", f"{prefix}bbw"),
        }

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

        for column in ("entry_signal_buy", "entry_signal_sell", "exit_signal_buy", "exit_signal_sell"):
            if column not in working.columns:
                working[column] = False

        session_ok = self._session_mask(working)
        volume_ok = self._volume_mask(working)
        guard = session_ok & volume_ok

        close = working["close"]
        upper = working[self.indicator_columns["upper"]]
        middle = working[self.indicator_columns["middle"]]
        lower = working[self.indicator_columns["lower"]]
        width = working[self.indicator_columns["width"]]

        # Normalize width by price and compare to rolling percentile of prior bars
        width_pct = (upper - lower) / middle
        rolling_q = (
            width_pct.shift(1)
            .rolling(window=self.squeeze_lookback, min_periods=self.squeeze_lookback)
            .quantile(self.squeeze_percentile)
        )
        squeeze_ready = width_pct <= rolling_q

        breakout_long = (close.shift(1) <= upper.shift(1)) & (close > upper)
        if self.enable_short:
            breakdown_short = (close.shift(1) >= lower.shift(1)) & (close < lower)
        else:
            breakdown_short = pd.Series(False, index=working.index)

        working.loc[(guard & squeeze_ready & breakout_long).fillna(False), "entry_signal_buy"] = True
        if self.enable_short:
            working.loc[(guard & squeeze_ready & breakdown_short).fillna(False), "entry_signal_sell"] = True

        working = self._apply_exit_logic(working, middle)
        return working

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
        self.warmup_periods[self.entry_timeframe] = max(current, registry_warmup, self.squeeze_lookback)

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
            raise ValueError(f"{self.__class__.__name__} missing indicator columns: {missing}")

    def _apply_exit_logic(self, df: pd.DataFrame, middle_band: pd.Series) -> pd.DataFrame:
        df = df.copy()
        in_long = False
        in_short = False

        for idx in range(len(df)):
            timestamp = df.at[idx, "timestamp"]
            current_time = timestamp.time() if isinstance(timestamp, pd.Timestamp) else None

            if self.market_close and current_time and current_time >= self.market_close:
                if in_long:
                    df.at[idx, "exit_signal_buy"] = True
                    in_long = False
                if in_short:
                    df.at[idx, "exit_signal_sell"] = True
                    in_short = False
                continue

            if df.at[idx, "entry_signal_buy"] and not in_long and not in_short:
                in_long = True
                continue

            if in_long:
                if df.at[idx, "close"] <= middle_band.at[idx]:
                    df.at[idx, "exit_signal_buy"] = True
                    in_long = False
                continue

            if self.enable_short and df.at[idx, "entry_signal_sell"] and not in_long and not in_short:
                in_short = True
                continue

            if in_short:
                if df.at[idx, "close"] >= middle_band.at[idx]:
                    df.at[idx, "exit_signal_sell"] = True
                    in_short = False

        return df

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
            "category": "Bollinger Breakout",
            "description": "Bollinger band squeeze breakout strategy with middle-band exits.",
            "parameters": {
                "entry_timeframe": self.entry_timeframe,
                "bollinger_period": self.period,
                "bollinger_std_dev": self.std_dev,
                "squeeze_threshold": self.squeeze_threshold,
                "squeeze_min_bars": self.squeeze_min_bars,
                "enable_short": self.enable_short,
            },
            "timeframes": self.required_timeframes,
        }
