"""
Volatility model implementations used by the synthetic options pricing stack.

The models here operate on already-sampled daily underlying data to keep the
pricing pipeline deterministic and auditable. Each model returns a
`VolatilityModelOutput` containing the per-session volatility series along with
any auxiliary metadata (e.g. ATM implied vols used during calibration).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


IST = "Asia/Kolkata"


@dataclass
class VolatilityModelOutput:
    """Container for volatility series and supporting metadata."""

    series: pd.Series
    metadata: Dict[str, Any]


class BaseVolatilityModel(ABC):
    """
    Base class for volatility estimators.

    Subclasses should only implement the `_compute_impl` method; the base class
    handles timestamp normalisation, sorting, and metadata wiring.
    """

    def __init__(self, annualization_factor: float = 252.0):
        self.annualization_factor = float(annualization_factor)

    def compute(
        self,
        underlying: pd.DataFrame,
        option_data: Optional[pd.DataFrame] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> VolatilityModelOutput:
        """
        Run the volatility model on the provided underlying data.

        Args:
            underlying: DataFrame with at least `timestamp` and `close` columns.
            option_data: Optional options DataFrame (used by calibrated models).
            context: Optional context dictionary (e.g. expiry date).
        """
        prepared = self._prepare_underlying(underlying)
        return self._compute_impl(prepared, option_data=option_data, context=context)

    @abstractmethod
    def _compute_impl(
        self,
        underlying: pd.DataFrame,
        option_data: Optional[pd.DataFrame] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> VolatilityModelOutput:
        """Subclasses implement the actual volatility calculation."""

    @staticmethod
    def _ensure_timezone(series: pd.Series) -> pd.Series:
        """
        Convert timestamps to Asia/Kolkata timezone for deterministic alignment.
        """
        ts = pd.to_datetime(series)
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(IST)
        else:
            ts = ts.dt.tz_convert(IST)
        return ts

    def _prepare_underlying(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise the underlying data into a daily frame sorted by session date.
        """
        if "timestamp" not in df.columns:
            raise ValueError("Underlying data must contain a 'timestamp' column")
        if "close" not in df.columns:
            raise ValueError("Underlying data must contain a 'close' column")

        working = df.copy()
        working["timestamp"] = self._ensure_timezone(working["timestamp"])
        working = working.sort_values("timestamp").drop_duplicates("timestamp")
        working["trade_date"] = working["timestamp"].dt.normalize()

        # If multiple bars per day exist (intraday data), aggregate to daily OHLC.
        if working["trade_date"].duplicated().any():
            agg_map = {"close": "last"}
            if "open" in working.columns:
                agg_map["open"] = "first"
            if "high" in working.columns:
                agg_map["high"] = "max"
            if "low" in working.columns:
                agg_map["low"] = "min"

            aggregated = working.set_index("timestamp").resample("1D", offset="0H", origin="start").agg(agg_map)
            aggregated = aggregated.dropna(subset=["close"]).reset_index()
            aggregated["trade_date"] = self._ensure_timezone(aggregated["timestamp"]).dt.normalize()
            return aggregated

        return working


class HistoricalVolatilityModel(BaseVolatilityModel):
    """Realised volatility from rolling log returns of the close price."""

    def __init__(self, window: int, annualization_factor: float = 252.0, min_periods: Optional[int] = None):
        super().__init__(annualization_factor=annualization_factor)
        if window <= 1:
            raise ValueError("Historical volatility window must be > 1")
        self.window = int(window)
        self.min_periods = min_periods if min_periods is not None else max(2, self.window // 2)

    def _compute_impl(
        self,
        underlying: pd.DataFrame,
        option_data: Optional[pd.DataFrame] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> VolatilityModelOutput:
        prices = underlying.set_index("trade_date")["close"].astype(float)
        log_returns = np.log(prices / prices.shift(1))
        rolling_std = log_returns.rolling(self.window, min_periods=self.min_periods).std()
        volatility = (rolling_std * np.sqrt(self.annualization_factor)).dropna()

        return VolatilityModelOutput(
            series=volatility,
            metadata={
                "model": "historical",
                "window": self.window,
                "annualization_factor": self.annualization_factor,
            },
        )


class ParkinsonVolatilityModel(BaseVolatilityModel):
    """
    Parkinson volatility estimator using the high-low price range.

    Reference:
    Parkinson, M. (1980). The extreme value method for estimating the variance
    of the stock price. Journal of Business, 53(1), 61-65.
    """

    def __init__(self, window: int, annualization_factor: float = 252.0, min_periods: Optional[int] = None):
        super().__init__(annualization_factor=annualization_factor)
        if window <= 1:
            raise ValueError("Parkinson volatility window must be > 1")
        self.window = int(window)
        self.min_periods = min_periods if min_periods is not None else max(2, self.window // 2)

    def _compute_impl(
        self,
        underlying: pd.DataFrame,
        option_data: Optional[pd.DataFrame] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> VolatilityModelOutput:
        if not {"high", "low"}.issubset(underlying.columns):
            raise ValueError("Parkinson model requires 'high' and 'low' columns in underlying data")

        daily = underlying.set_index("trade_date")[["high", "low"]].astype(float)
        hl_term = np.log(daily["high"] / daily["low"]) ** 2
        coefficient = 1.0 / (4.0 * np.log(2.0))
        rolling_mean = hl_term.rolling(self.window, min_periods=self.min_periods).mean()
        volatility = np.sqrt(coefficient * rolling_mean) * np.sqrt(self.annualization_factor)
        volatility = volatility.dropna()

        return VolatilityModelOutput(
            series=volatility,
            metadata={
                "model": "parkinson",
                "window": self.window,
                "annualization_factor": self.annualization_factor,
            },
        )


class CalibratedIVModel(BaseVolatilityModel):
    """
    Calibrate implied volatility to actual ATM option prices and propagate
    adjustments to other strikes using a configurable skew model.

    The implementation defaults to historical volatility as a baseline and
    replaces the ATM node with the observed implied volatility wherever
    possible. Missing calibrations fall back to the baseline series.
    """

    def __init__(
        self,
        atm_method: str = "closest_strike",
        skew_model: str = "linear",
        skew_coefficient: float = 0.0,
        base_model: Optional[BaseVolatilityModel] = None,
        annualization_factor: float = 252.0,
    ):
        super().__init__(annualization_factor=annualization_factor)
        self.atm_method = atm_method
        self.skew_model = skew_model
        self.skew_coefficient = float(skew_coefficient)
        self.base_model = base_model or HistoricalVolatilityModel(window=20, annualization_factor=annualization_factor)

    def _compute_impl(
        self,
        underlying: pd.DataFrame,
        option_data: Optional[pd.DataFrame] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> VolatilityModelOutput:
        if option_data is None:
            raise ValueError("Calibrated IV model requires actual option data for calibration")

        from src.core.options.options_engine import BlackScholesEngine  # Lazy import to avoid cycles

        underlying = underlying.copy()
        underlying["trade_date"] = underlying["timestamp"].dt.tz_convert(IST).dt.normalize()
        prices = underlying.set_index("trade_date")["close"].astype(float)

        risk_free_rate = 0.0
        if context:
            risk_free_rate = context.get("risk_free_rate", 0.0) or 0.0

        base_output = self.base_model.compute(underlying, option_data=option_data, context=context)
        base_series = base_output.series

        options_working = option_data.copy()
        options_working["timestamp"] = self._ensure_timezone(options_working["timestamp"])
        options_working["trade_date"] = options_working["timestamp"].dt.normalize()

        if "expiry" not in options_working.columns:
            raise ValueError("Option data must include an 'expiry' column for calibration")

        expiry_value = options_working["expiry"].iloc[0]
        expiry_date = pd.to_datetime(expiry_value).date()

        bs_engine = BlackScholesEngine()
        atm_iv_series: Dict[pd.Timestamp, float] = {}

        for trade_date, daily_slice in options_working.groupby("trade_date"):
            if trade_date not in prices.index or trade_date not in base_series.index:
                continue

            spot = prices.loc[trade_date]
            if not np.isfinite(spot) or spot <= 0:
                continue

            dte_days = (expiry_date - trade_date.date()).days
            if dte_days <= 0:
                # Use a small positive tenor to avoid division by zero
                time_to_expiry = 1.0 / 365.0
            else:
                time_to_expiry = dte_days / 365.0

            atm_contract = self._select_atm_contract(daily_slice, spot)
            if atm_contract is None:
                continue

            market_price = float(atm_contract["close"])
            strike = float(atm_contract["strike"])
            option_type = "call" if atm_contract["option_type"] == "CE" else "put"

            implied_vol = bs_engine.calculate_implied_volatility(
                option_price=market_price,
                S=spot,
                K=strike,
                T=time_to_expiry,
                r=risk_free_rate,
                option_type=option_type,
            )

            if np.isfinite(implied_vol) and implied_vol > 0:
                atm_iv_series[trade_date] = implied_vol

        atm_series = pd.Series(atm_iv_series, name="atm_iv").sort_index()
        combined = base_series.copy()
        combined.update(atm_series)
        combined = combined.sort_index()
        combined = combined.ffill()

        metadata = {
            "model": "calibrated_iv",
            "base_model": base_output.metadata,
            "atm_method": self.atm_method,
            "skew_model": self.skew_model,
            "skew_coefficient": self.skew_coefficient,
            "atm_series": atm_series,
            "risk_free_rate": risk_free_rate,
        }

        return VolatilityModelOutput(series=combined, metadata=metadata)

    def _select_atm_contract(self, slice_df: pd.DataFrame, spot: float) -> Optional[pd.Series]:
        """
        Choose the ATM contract from the available strikes for the calibration date.
        """
        if self.atm_method != "closest_strike":
            raise NotImplementedError(f"ATM method '{self.atm_method}' is not implemented")

        ranked = slice_df.assign(strike_diff=np.abs(slice_df["strike"].astype(float) - spot))
        ranked = ranked.sort_values(["strike_diff", "option_type"])
        if ranked.empty:
            return None
        return ranked.iloc[0]
