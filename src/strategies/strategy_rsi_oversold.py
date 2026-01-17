"""
RSI Oversold/Overbought Mean Reversion Strategy

Buys when RSI < 30 (oversold), sells when RSI > 70 (overbought)
Exits when price returns to 50 RSI (neutral zone)
"""

from src.strategies.support.strategy_base import StrategyBase
from typing import Dict, Union
import pandas as pd


class StrategyRSIOversold(StrategyBase):
    """
    RSI Mean Reversion Strategy Implementation

    Entry: RSI crosses into oversold (<30) or overbought (>70) zones
    Exit: RSI returns to neutral zone (crosses 50)
    """

    def __init__(self, name: str, parameters: dict = None, config=None):
        """
        Initialize the RSI strategy.

        Args:
            name: Strategy name
            parameters: Dictionary containing rsi_period, oversold, overbought
            config: Strategy configuration object
        """
        super().__init__(name, parameters or {}, config)

        # Extract strategy parameters
        self.rsi_period = self.parameters.get("rsi_period", 14)
        self.oversold = self.parameters.get("oversold", 30)
        self.overbought = self.parameters.get("overbought", 70)

        # Define required timeframes
        self.required_timeframes = ["5m"]

        # Set warmup periods
        self.warmup_periods = {"5m": self.rsi_period + 10}

    def prepare_data(
        self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], ticker: str, pull_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare data by applying indicators and removing warmup periods.

        Args:
            data: DataFrame or Dict of DataFrames by timeframe
            ticker: Ticker symbol
            pull_date: Date of data pull

        Returns:
            Dictionary of prepared dataframes with indicators applied
        """
        # Convert DataFrame to dict first
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
        Generate buy/sell signals based on RSI levels.

        Uses shift(1) and shift(2) to prevent look-ahead bias by comparing
        previous bar states to detect zone crossings.

        Args:
            data: Dictionary of prepared dataframes

        Returns:
            DataFrame with signal columns added
        """
        df = data["5m"].copy()

        # Define column name for RSI
        rsi_col = f"5m_rsi_{self.rsi_period}"

        # CRITICAL: Use .shift(1) and .shift(2) to prevent look-ahead bias

        # Entry Long: RSI crosses below oversold threshold (30)
        # Previous bar RSI < oversold AND bar before that RSI >= oversold
        df["entry_signal_buy"] = (
            (df[rsi_col].shift(1) < self.oversold) &
            (df[rsi_col].shift(2) >= self.oversold)
        )

        # Entry Short: RSI crosses above overbought threshold (70)
        # Previous bar RSI > overbought AND bar before that RSI <= overbought
        df["entry_signal_sell"] = (
            (df[rsi_col].shift(1) > self.overbought) &
            (df[rsi_col].shift(2) <= self.overbought)
        )

        # Exit Long: RSI crosses above 50 (neutral zone)
        df["exit_signal_buy"] = (
            (df[rsi_col].shift(1) > 50) &
            (df[rsi_col].shift(2) <= 50)
        )

        # Exit Short: RSI crosses below 50 (neutral zone)
        df["exit_signal_sell"] = (
            (df[rsi_col].shift(1) < 50) &
            (df[rsi_col].shift(2) >= 50)
        )

        return df
