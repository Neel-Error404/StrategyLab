"""
Open Source Baseline Strategy
=============================

A vendor-neutral reference strategy suitable for open source distribution.
The logic favours simplicity, transparency, and reproducibility:

- Combines a short/long moving-average trend filter with a momentum overlay.
- Applies a basic volume gate to avoid illiquid periods.
- Emits deterministic boolean entry/exit signals consumed by the runner.

The implementation intentionally avoids broker-specific or proprietary logic
while demonstrating best practices for building strategies on this framework.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from .strategy_base import StrategyBase


class OpenSourceBaselineStrategy(StrategyBase):
    """
    Reference implementation for community strategies.

    Signal philosophy:
    - Go long when short trend > long trend AND momentum positive.
    - Go short when short trend < long trend AND momentum negative.
    - Exit if conditions flip or momentum decays past neutral.

    Parameters (override via config/templates):
        short_window: length of fast SMA (default 10)
        long_window: length of slow SMA (default 30)
        momentum_window: lookback for percentage momentum (default 20)
        momentum_exit_buffer: tolerance before flipping exits (default 0.0)
        min_volume: minimum volume filter (default 1_000)
    """

    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None):
        params = parameters or {}
        super().__init__(name=name, parameters=params)

        self.short_window = int(params.get("short_window", 10))
        self.long_window = int(params.get("long_window", 30))
        self.momentum_window = int(params.get("momentum_window", 20))
        self.momentum_exit_buffer = float(params.get("momentum_exit_buffer", 0.0))
        self.min_volume = int(params.get("min_volume", 1_000))
        self.required_timeframes = ["5m"]

        if self.short_window <= 0 or self.long_window <= 0:
            raise ValueError("Moving-average windows must be positive")
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be strictly less than long_window")
        if self.momentum_window <= 0:
            raise ValueError("momentum_window must be positive")
        if self.min_volume < 0:
            raise ValueError("min_volume cannot be negative")

        self.logger = logging.getLogger(f"strategy.{self.__class__.__name__}")
        self.logger.info(
            "Open Source Baseline initialised | short=%s long=%s momentum=%s volume>=%s",
            self.short_window,
            self.long_window,
            self.momentum_window,
            self.min_volume,
        )

    def prepare_data(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        ticker: str,
        pull_date: str,
    ) -> pd.DataFrame:
        """
        Calculate indicators required for signal generation.

        Args:
            data: OHLCV dataframe indexed by timestamp.
        """
        if isinstance(data, dict):
            timeframe = self.required_timeframes[0] if self.required_timeframes else next(iter(data))
            df = data.get(timeframe)
            if df is None or df.empty:
                self.logger.warning(
                    "No data available for %s on %s using timeframe %s", ticker, pull_date, timeframe
                )
                return pd.DataFrame()
        else:
            df = data

        if df.empty:
            self.logger.warning("No data provided for %s on %s", ticker, pull_date)
            return df

        df = df.copy()

        df["short_sma"] = df["close"].rolling(
            window=self.short_window, min_periods=self.short_window
        ).mean()
        df["long_sma"] = df["close"].rolling(
            window=self.long_window, min_periods=self.long_window
        ).mean()
        df["momentum"] = df["close"].pct_change(self.momentum_window)
        df["volume_ok"] = df["volume"] >= self.min_volume

        # Guard rails for signalling
        df["trend_slope"] = df["short_sma"] - df["long_sma"]
        df["trend_slope_prev"] = df["trend_slope"].shift(1)
        df["momentum_prev"] = df["momentum"].shift(1)

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Produce deterministic entry/exit signals.

        Returns a dataframe mirroring input with boolean signal columns.
        """
        if data.empty:
            return data

        df = data.copy()

        for column in (
            "entry_signal_buy",
            "entry_signal_sell",
            "exit_signal_buy",
            "exit_signal_sell",
        ):
            df[column] = False

        valid = (
            df["short_sma"].notna()
            & df["long_sma"].notna()
            & df["momentum"].notna()
            & df["volume_ok"]
        )

        bullish_state = valid & (df["trend_slope"] > 0) & (df["momentum"] > 0)
        bearish_state = valid & (df["trend_slope"] < 0) & (df["momentum"] < 0)

        prev_bullish = bullish_state.shift(1)
        prev_bullish = prev_bullish.mask(prev_bullish.isna(), False).astype(bool)

        prev_bearish = bearish_state.shift(1)
        prev_bearish = prev_bearish.mask(prev_bearish.isna(), False).astype(bool)

        bullish_cross = bullish_state & ~prev_bullish
        bearish_cross = bearish_state & ~prev_bearish

        df.loc[bullish_cross, "entry_signal_buy"] = True
        df.loc[bearish_cross, "entry_signal_sell"] = True

        # Exit criteria when trend/momentum unwind.
        exit_buy = (
            bearish_cross
            | (valid & (df["momentum"] < self.momentum_exit_buffer))
            | (valid & (df["trend_slope"] <= 0))
        )
        exit_sell = (
            bullish_cross
            | (valid & (df["momentum"] > -self.momentum_exit_buffer))
            | (valid & (df["trend_slope"] >= 0))
        )

        df.loc[exit_buy, "exit_signal_buy"] = True
        df.loc[exit_sell, "exit_signal_sell"] = True

        # Metrics for downstream analysis.
        df["regime"] = np.select(
            [bullish_state, bearish_state], ["bullish", "bearish"], default="neutral"
        )

        return df

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Provide metadata used by CLI + reporting layers.
        """
        return {
            "name": self.name,
            "category": "Trend + Momentum Hybrid",
            "description": (
                "Reference open-source strategy blending moving-average trend "
                "filtering with momentum confirmation."
            ),
            "parameters": {
                "short_window": self.short_window,
                "long_window": self.long_window,
                "momentum_window": self.momentum_window,
                "momentum_exit_buffer": self.momentum_exit_buffer,
                "min_volume": self.min_volume,
            },
            "required_columns": self.required_columns,
            "timeframes": self.required_timeframes,
        }
