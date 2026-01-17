"""
RSI Divergence Strategy
========================

Economic Rationale:
------------------
Divergence between price and momentum indicators signals potential reversals.
- Bullish Divergence: Price makes lower low, RSI makes higher low -> buy signal
- Bearish Divergence: Price makes higher high, RSI makes lower high -> sell signal

This captures situations where price momentum is weakening despite continued
price movement, indicating exhaustion and potential reversal.

Look-Ahead Bias Prevention:
--------------------------
All comparisons use .shift(1) and .shift(2) to ensure signals use only
historical data available at the time of decision.
"""

from src.strategies.support.strategy_base import StrategyBase
from typing import Dict, Union
import pandas as pd
import numpy as np


class StrategyRSIDivergence(StrategyBase):
    """
    RSI Divergence Detection Strategy

    Identifies bullish and bearish divergences between price action and RSI
    to generate reversal signals.
    """

    def __init__(self, name: str, parameters: dict = None, config=None):
        """
        Initialize RSI divergence strategy.

        Args:
            name: Strategy name
            parameters: Dict containing rsi_period, lookback_bars, divergence_threshold
            config: Strategy configuration object
        """
        super().__init__(name, parameters or {}, config)

        # Extract parameters
        self.rsi_period = int(self.parameters.get("rsi_period", 14))
        self.lookback_bars = int(self.parameters.get("lookback_bars", 10))
        self.divergence_threshold = float(self.parameters.get("divergence_threshold", 3.0))

        # Define required timeframes
        self.required_timeframes = ["5m"]

        # Set warmup periods (RSI needs period + buffer + lookback)
        self.warmup_periods = {"5m": self.rsi_period + self.lookback_bars + 20}

    def prepare_data(
        self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], ticker: str, pull_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare data by applying RSI indicator and removing warmup periods.

        Args:
            data: DataFrame or Dict of DataFrames by timeframe
            ticker: Ticker symbol
            pull_date: Date of data pull

        Returns:
            Dictionary of prepared dataframes with indicators applied
        """
        # CRITICAL: Convert DataFrame to dict first
        frames = self._ensure_timeframe_dict(data)

        # Apply configured indicators (RSI defined in YAML)
        frames = self._apply_configured_indicators(frames)

        # Remove warmup periods
        for tf, df in frames.items():
            warmup = self.warmup_periods.get(tf, 0)
            frames[tf] = df.iloc[warmup:].reset_index(drop=True)

        return frames

    def _ensure_timeframe_dict(
        self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    ) -> Dict[str, pd.DataFrame]:
        """Convert DataFrame to dict format if needed."""
        if isinstance(data, dict):
            return data
        return {self.required_timeframes[0]: data}

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate buy/sell signals based on RSI divergence detection.

        Bullish Divergence:
        - Price makes lower low (LL)
        - RSI makes higher low (HL)
        - Entry signal: Buy at next bar open

        Bearish Divergence:
        - Price makes higher high (HH)
        - RSI makes lower high (LH)
        - Entry signal: Sell at next bar open

        Exit signals:
        - Exit long when bearish divergence detected
        - Exit short when bullish divergence detected

        Args:
            data: Dictionary of prepared dataframes

        Returns:
            DataFrame with signal columns added
        """
        df = data["5m"].copy()

        # Initialize signal columns
        df["entry_signal_buy"] = False
        df["entry_signal_sell"] = False
        df["exit_signal_buy"] = False
        df["exit_signal_sell"] = False

        # Define RSI column name
        rsi_col = f"5m_rsi_{self.rsi_period}"

        # Need at least lookback bars to detect divergence
        if len(df) < self.lookback_bars:
            return df

        # Detect divergences using shifted data (prevent look-ahead bias)
        for i in range(self.lookback_bars, len(df)):
            # Use data up to previous bar only (shift(1) effect)
            window_start = i - self.lookback_bars
            window_end = i  # Exclusive, so this is i-1 inclusive

            # Get price and RSI windows (previous bars only)
            price_window = df.loc[window_start:window_end-1, "close"].values
            rsi_window = df.loc[window_start:window_end-1, rsi_col].values

            # Skip if any NaN values
            if np.any(pd.isna(price_window)) or np.any(pd.isna(rsi_window)):
                continue

            # BULLISH DIVERGENCE: Price LL, RSI HL
            # Find lowest price point in window
            price_min_idx = np.argmin(price_window)
            price_min = price_window[price_min_idx]

            # Check if current price is lower than previous low
            if price_min_idx < len(price_window) - 1:  # Not the most recent bar
                recent_price_low = np.min(price_window[price_min_idx+1:])
                price_makes_ll = recent_price_low < price_min

                if price_makes_ll:
                    # Check if RSI makes higher low
                    rsi_at_first_low = rsi_window[price_min_idx]
                    recent_rsi_low_idx = np.argmin(price_window[price_min_idx+1:]) + price_min_idx + 1
                    rsi_at_second_low = rsi_window[recent_rsi_low_idx]

                    # RSI makes HL if second low RSI > first low RSI
                    rsi_makes_hl = rsi_at_second_low > rsi_at_first_low

                    # Require minimum threshold difference
                    rsi_diff = abs(rsi_at_second_low - rsi_at_first_low)

                    if rsi_makes_hl and rsi_diff >= self.divergence_threshold:
                        df.at[i, "entry_signal_buy"] = True
                        df.at[i, "exit_signal_sell"] = True  # Exit shorts on bullish div

            # BEARISH DIVERGENCE: Price HH, RSI LH
            # Find highest price point in window
            price_max_idx = np.argmax(price_window)
            price_max = price_window[price_max_idx]

            # Check if current price is higher than previous high
            if price_max_idx < len(price_window) - 1:  # Not the most recent bar
                recent_price_high = np.max(price_window[price_max_idx+1:])
                price_makes_hh = recent_price_high > price_max

                if price_makes_hh:
                    # Check if RSI makes lower high
                    rsi_at_first_high = rsi_window[price_max_idx]
                    recent_rsi_high_idx = np.argmax(price_window[price_max_idx+1:]) + price_max_idx + 1
                    rsi_at_second_high = rsi_window[recent_rsi_high_idx]

                    # RSI makes LH if second high RSI < first high RSI
                    rsi_makes_lh = rsi_at_second_high < rsi_at_first_high

                    # Require minimum threshold difference
                    rsi_diff = abs(rsi_at_second_high - rsi_at_first_high)

                    if rsi_makes_lh and rsi_diff >= self.divergence_threshold:
                        df.at[i, "entry_signal_sell"] = True
                        df.at[i, "exit_signal_buy"] = True  # Exit longs on bearish div

        return df
