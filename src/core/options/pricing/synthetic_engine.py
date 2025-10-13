"""
Synthetic pricing engine built on top of Black-Scholes and configurable
volatility models.

This module provides a deterministic, auditable interface for generating
synthetic option prices and Greeks used during pricing validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
from datetime import timedelta

from src.core.options.options_engine import BlackScholesEngine
from src.core.options.pricing.volatility_models import (
    BaseVolatilityModel,
    HistoricalVolatilityModel,
    ParkinsonVolatilityModel,
    CalibratedIVModel,
    VolatilityModelOutput,
    IST,
)


def _ensure_ist(series: pd.Series) -> pd.Series:
    """Normalise timestamps to Asia/Kolkata timezone."""
    ts = pd.to_datetime(series)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(IST)
    else:
        ts = ts.dt.tz_convert(IST)
    return ts


def build_volatility_model(volatility_cfg: Dict[str, Any]) -> BaseVolatilityModel:
    """
    Construct a volatility model instance from configuration.
    """
    vol_type = volatility_cfg.get("type")
    annualization = volatility_cfg.get("annualization_factor", 252)

    if vol_type.startswith("historical"):
        window = volatility_cfg.get("window")
        if window is None and "_" in vol_type:
            try:
                suffix = vol_type.split("_", 1)[1]
                window = int("".join(ch for ch in suffix if ch.isdigit())) or 20
            except ValueError:
                window = 20
        window = window or 20
        return HistoricalVolatilityModel(window=window, annualization_factor=annualization)
    if vol_type.startswith("parkinson"):
        window = volatility_cfg.get("window", 20)
        return ParkinsonVolatilityModel(window=window, annualization_factor=annualization)
    if vol_type == "calibrated_iv":
        calibration = volatility_cfg.get("calibration", {})
        base_cfg = calibration.get("base_model", {"type": "historical", "window": 20})
        base_model = build_volatility_model({**base_cfg, "annualization_factor": annualization})
        return CalibratedIVModel(
            atm_method=calibration.get("atm_method", "closest_strike"),
            skew_model=calibration.get("skew_model", "linear"),
            skew_coefficient=calibration.get("skew_coefficient", 0.0),
            base_model=base_model,
            annualization_factor=annualization,
        )
    raise ValueError(f"Unsupported volatility model type: {vol_type}")


@dataclass
class SyntheticPricingResult:
    """Synthetic pricing output at the row level."""

    dataframe: pd.DataFrame
    volatility_metadata: Dict[str, Any]


class SyntheticPricingEngine:
    """
    Orchestrates volatility estimation and Black-Scholes pricing for option rows.
    """

    def __init__(
        self,
        volatility_model: BaseVolatilityModel,
        risk_free_rate: float = 0.06,
        dividend_yield: float = 0.0,
        vol_floor: float = 0.10,
        vol_cap: float = 1.50,
    ):
        self.volatility_model = volatility_model
        self.risk_free_rate = float(risk_free_rate)
        self.dividend_yield = float(dividend_yield)
        self.vol_floor = float(vol_floor)
        self.vol_cap = float(vol_cap)
        self.bs_engine = BlackScholesEngine(risk_free_rate=self.risk_free_rate)

        self._underlying_daily: Optional[pd.DataFrame] = None
        self._vol_series: Optional[pd.Series] = None
        self._vol_metadata: Dict[str, Any] = {}

    def set_underlying_data(self, underlying_df: pd.DataFrame) -> None:
        """
        Register the underlying equity data (daily resolution) used for pricing.
        """
        if underlying_df.empty:
            raise ValueError("Underlying data is empty; cannot initialise synthetic engine")

        data = underlying_df.copy()
        data["timestamp"] = _ensure_ist(data["timestamp"])
        data = data.sort_values("timestamp")
        data["trade_date"] = data["timestamp"].dt.normalize()

        # If intraday data is provided, aggregate to daily closes deterministically.
        if data["trade_date"].duplicated().any():
            agg_map = {"close": "last"}
            if "open" in data.columns:
                agg_map["open"] = "first"
            if "high" in data.columns:
                agg_map["high"] = "max"
            if "low" in data.columns:
                agg_map["low"] = "min"

            daily = data.set_index("timestamp").resample("1D").agg(agg_map).dropna(subset=["close"]).reset_index()
            daily["trade_date"] = _ensure_ist(daily["timestamp"]).dt.normalize()
        else:
            daily = data

        self._underlying_daily = daily
        self._vol_series = None  # Invalidate cached volatility when underlying changes
        self._vol_metadata = {}

    def fit_volatility_surface(
        self,
        option_data: Optional[pd.DataFrame] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> pd.Series:
        """
        Estimate the volatility surface using the configured volatility model.
        """
        if self._underlying_daily is None:
            raise RuntimeError("Underlying data not set. Call `set_underlying_data` first.")

        ctx = dict(context or {})
        ctx.setdefault("risk_free_rate", self.risk_free_rate)

        vol_output: VolatilityModelOutput = self.volatility_model.compute(
            underlying=self._underlying_daily,
            option_data=option_data,
            context=ctx,
        )

        series = (
            vol_output.series.sort_index()
            .clip(lower=self.vol_floor, upper=self.vol_cap)
            .ffill()
        )

        self._vol_series = series
        self._vol_metadata = vol_output.metadata
        self._vol_metadata["vol_floor"] = self.vol_floor
        self._vol_metadata["vol_cap"] = self.vol_cap
        return series

    def price_options(
        self,
        options_df: pd.DataFrame,
        option_metadata: Optional[Dict[str, Any]] = None,
    ) -> SyntheticPricingResult:
        """
        Price options using the previously-fitted volatility surface.

        Args:
            options_df: DataFrame with option OHLC data (single expiry preferred).
            option_metadata: Optional dict with context such as expiry date.
        """
        if self._underlying_daily is None:
            raise RuntimeError("Underlying data not set. Call `set_underlying_data` first.")
        if self._vol_series is None:
            raise RuntimeError("Volatility surface not available. Call `fit_volatility_surface` first.")

        if options_df.empty:
            raise ValueError("Options DataFrame is empty; nothing to price.")

        working = options_df.copy()
        working["timestamp"] = _ensure_ist(working["timestamp"])
        working = working.sort_values(["timestamp", "strike", "option_type"])
        working["trade_date"] = working["timestamp"].dt.normalize()
        working["option_type_bsm"] = working["option_type"].map({"CE": "call", "PE": "put"})

        if working["option_type_bsm"].isna().any():
            raise ValueError("Option types must be 'CE' or 'PE'")

        # Merge underlying closes
        underlying = self._underlying_daily[["trade_date", "close"]].drop_duplicates("trade_date")
        working = working.merge(underlying, on="trade_date", how="left", suffixes=("", "_underlying"))
        working = working.rename(columns={"close_underlying": "underlying_close"})

        if "underlying_close" not in working.columns or working["underlying_close"].isna().any():
            raise ValueError("Missing underlying close for some option rows after merge")

        # Merge volatility series
        vol_series = self._vol_series.rename("model_vol")
        working = working.merge(vol_series, left_on="trade_date", right_index=True, how="left")
        if working["model_vol"].isna().any():
            # Forward fill within each expiry to avoid dropping rows; still deterministic
            working["model_vol"] = working.groupby("expiry")["model_vol"].ffill().bfill()
        if working["model_vol"].isna().any():
            raise ValueError("Volatility series missing for certain trade dates even after forward fill")

        skew_model = self._vol_metadata.get("skew_model")
        skew_coeff = self._vol_metadata.get("skew_coefficient", 0.0)
        if skew_model is None:
            skew_coeff = 0.0

        expiry_series = working["expiry"]
        expiry_value = expiry_series.iloc[0] if expiry_series.nunique() == 1 else None

        def _compute_time_to_expiry(row) -> float:
            expiry_obj = row["expiry"]
            if not pd.notna(expiry_obj):
                raise ValueError("Expiry missing in options data")
            expiry_date = pd.to_datetime(expiry_obj).date()
            trade_ts = row["timestamp"]
            expiry_ts = pd.Timestamp(expiry_date, tz=IST) + timedelta(hours=15, minutes=30)
            tenor_days = (expiry_ts - trade_ts).total_seconds() / 86400.0
            return max(tenor_days / 365.0, 0.0)

        working["time_to_expiry"] = working.apply(_compute_time_to_expiry, axis=1)

        def _apply_skew(row, base_vol: float) -> float:
            if skew_coeff == 0.0:
                return base_vol

            option_type = row["option_type_bsm"]
            spot = row["underlying_close"]
            strike = float(row["strike"])

            if option_type == "call":
                moneyness = spot / strike
            else:
                moneyness = strike / spot

            adjustment = 1.0 + skew_coeff * (moneyness - 1.0)
            adjusted = base_vol * max(adjustment, 0.01)
            return float(np.clip(adjusted, self.vol_floor, self.vol_cap))

        working["implied_volatility"] = working.apply(lambda r: _apply_skew(r, float(r["model_vol"])), axis=1)

        # Price options row by row for clarity/auditability
        prices = []
        deltas = []
        gammas = []
        thetas = []
        vegas = []
        rhos = []

        for _, row in working.iterrows():
            S = float(row["underlying_close"])
            K = float(row["strike"])
            T = float(row["time_to_expiry"])
            sigma = float(row["implied_volatility"])
            option_type = row["option_type_bsm"]

            price = self.bs_engine.calculate_option_price(
                S=S,
                K=K,
                T=T,
                r=self.risk_free_rate,
                sigma=sigma,
                option_type=option_type,
            )
            greeks = self.bs_engine.calculate_greeks(
                S=S,
                K=K,
                T=T,
                r=self.risk_free_rate,
                sigma=sigma,
                option_type=option_type,
            )

            prices.append(price)
            deltas.append(greeks["delta"])
            gammas.append(greeks["gamma"])
            thetas.append(greeks["theta"])
            vegas.append(greeks["vega"])
            rhos.append(greeks["rho"])

        working["synthetic_price"] = prices
        working["delta"] = deltas
        working["gamma"] = gammas
        working["theta"] = thetas
        working["vega"] = vegas
        working["rho"] = rhos

        metadata = {
            "risk_free_rate": self.risk_free_rate,
            "dividend_yield": self.dividend_yield,
            "volatility_metadata": self._vol_metadata,
        }
        if expiry_value:
            metadata["expiry"] = pd.to_datetime(expiry_value).date().isoformat()
        if option_metadata:
            metadata.update(option_metadata)

        return SyntheticPricingResult(dataframe=working, volatility_metadata=metadata)
