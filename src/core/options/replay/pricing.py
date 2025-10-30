"""
Hybrid pricing utilities for the options replay engine.

Combines cached option bars with the synthetic pricing stack to deliver
deterministic entry/exit valuations and mark-to-market series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.options.options_engine import BlackScholesEngine
from src.core.options.pricing.synthetic_engine import (
    SyntheticPricingEngine,
    build_volatility_model,
)
from src.core.options.data.schemas import OptionType
from src.core.options.replay.config import OptionsReplayConfig
from src.core.options.pricing.costs import LegCostBreakdown, OptionTransactionCostModel
from .models import OptionContractSpec, PricingEvent, OptionPositionSnapshot
from .data_loader import OptionDataStore


class LiquidityFilterError(Exception):
    """Raised when liquidity filters request skipping a trade."""

    def __init__(self, reasons: List[str], metadata: Dict[str, object]) -> None:
        super().__init__(";".join(reasons))
        self.reasons = reasons
        self.metadata = metadata


def _time_to_expiry(expiry: pd.Timestamp, timestamp: pd.Timestamp) -> float:
    """Return time to expiry in years (non-negative)."""
    delta_seconds = max((expiry - timestamp).total_seconds(), 0.0)
    return delta_seconds / (365.0 * 24.0 * 3600.0)


def _option_type_to_label(option_type: OptionType) -> str:
    return "call" if option_type == OptionType.CALL else "put"


def _choose_actual_price(row: pd.Series, assumption: str) -> Optional[float]:
    """Determine fill price from actual option bar."""
    assumption = assumption.lower()
    if assumption == "open":
        return float(row["open"])
    if assumption == "close":
        return float(row["close"])
    if assumption == "mid":
        if {"bid", "ask"}.issubset(row.index) and not row[["bid", "ask"]].isna().any():
            return float((row["bid"] + row["ask"]) / 2.0)
        return float((row["high"] + row["low"]) / 2.0)
    if assumption == "vwap" and "vwap" in row.index and not pd.isna(row["vwap"]):
        return float(row["vwap"])
    # Fallback to close if assumption unsupported
    return float(row["close"])


@dataclass
class PricingContext:
    """Cached pricing artefacts per expiry/strike."""

    vol_series: pd.Series
    option_chain: pd.DataFrame
    source_timeframe: str


class HybridPricingEngine:
    """
    Hybrid pricing engine that attempts to use cached option data and falls back to synthetic pricing.
    """

    def __init__(
        self,
        config: OptionsReplayConfig,
        underlying_data: pd.DataFrame,
        option_store: OptionDataStore,
    ) -> None:
        self.config = config
        self.option_store = option_store
        synthetic_cfg = config.pricing.synthetic
        vol_model = build_volatility_model(
            {
                "type": synthetic_cfg.volatility_model,
                "window": 20,  # window handled inside specific models
                "annualization_factor": 252,
            }
        )
        self.synthetic_engine = SyntheticPricingEngine(
            volatility_model=vol_model,
            risk_free_rate=synthetic_cfg.risk_free_rate,
            dividend_yield=synthetic_cfg.dividend_yield,
            vol_floor=synthetic_cfg.vol_floor,
            vol_cap=synthetic_cfg.vol_cap,
        )
        self.synthetic_engine.set_underlying_data(underlying_data)
        self.black_scholes = BlackScholesEngine(risk_free_rate=synthetic_cfg.risk_free_rate)
        self._pricing_cache: Dict[str, PricingContext] = {}
        self._liquidity_cfg = config.liquidity
        self._cost_model = OptionTransactionCostModel(config.pricing.costs, config.pricing.slippage)

    def _ensure_context(self, expiry: pd.Timestamp) -> PricingContext:
        key = expiry.normalize().strftime("%Y-%m-%d")
        if key in self._pricing_cache:
            return self._pricing_cache[key]
        chain, timeframe = self.option_store.get_chain_for_vol(expiry)
        vol_series = self.synthetic_engine.fit_volatility_surface(
            option_data=chain,
            context={"expiry": expiry.strftime("%Y-%m-%d")},
        )
        vol_series = vol_series.sort_index().ffill()
        ctx = PricingContext(vol_series=vol_series, option_chain=chain, source_timeframe=timeframe)
        self._pricing_cache[key] = ctx
        return ctx

    def _check_liquidity(self, row: pd.Series) -> List[str]:
        failures: List[str] = []
        cfg = self._liquidity_cfg

        oi = row.get("open_interest")
        if pd.isna(oi):
            failures.append("open_interest_missing")
        elif oi < cfg.min_open_interest:
            failures.append("open_interest_below_min")

        volume = row.get("volume")
        if pd.isna(volume):
            failures.append("volume_missing")
        elif volume < cfg.min_volume:
            failures.append("volume_below_min")

        max_spread = cfg.max_spread_pct
        if max_spread is not None and max_spread >= 0:
            bid = row.get("bid")
            ask = row.get("ask")
            spread_ratio: Optional[float] = None
            if not pd.isna(bid) and not pd.isna(ask) and bid > 0 and ask > 0:
                mid = (bid + ask) / 2.0
                if mid > 0:
                    spread_ratio = (ask - bid) / mid
            if spread_ratio is None:
                failures.append("bid_ask_missing")
            elif spread_ratio > max_spread:
                failures.append("spread_above_max")

        return failures

    def _extract_actual_price(
        self,
        context: PricingContext,
        contract: OptionContractSpec,
        timestamp: pd.Timestamp,
        allow_skip: bool,
    ) -> Tuple[Optional[float], Dict[str, object]]:
        row, metadata = self.option_store.find_price_bar(
            expiry=contract.expiry,
            option_type=contract.option_type,
            strike=contract.strike,
            timestamp=timestamp,
        )
        if row is None:
            fallback_meta = {"reason": metadata.get("reason", "no_actual_bar")}
            if "attempts" in metadata:
                fallback_meta["attempts"] = metadata["attempts"]
            return None, fallback_meta
        price = _choose_actual_price(row, self.config.pricing.actual.fill_assumption)
        meta = {
            "source": "actual_cache",
            "timeframe": metadata.get("timeframe"),
            "alignment": metadata.get("alignment"),
            "bar_timestamp": metadata.get("bar_timestamp"),
        }
        if "attempts" in metadata:
            meta["attempts"] = metadata["attempts"]

        liquidity_failures = self._check_liquidity(row)
        if liquidity_failures:
            meta["liquidity_failures"] = liquidity_failures
            behavior = self._liquidity_cfg.on_filter_fail
            if behavior == "ignore_filters":
                pass
            elif behavior == "use_synthetic":
                meta["reason"] = "liquidity_violation"
                return None, meta
            elif behavior == "skip_trade" and allow_skip:
                meta["reason"] = "liquidity_violation_skip"
                raise LiquidityFilterError(liquidity_failures, meta)
            else:
                # During lifecycle pricing we fall back to synthetic rather than skipping the trade.
                meta["reason"] = "liquidity_violation"
                return None, meta
        else:
            meta["liquidity_failures"] = []

        return price, meta

    def _compute_synthetic(
        self,
        context: PricingContext,
        contract: OptionContractSpec,
        timestamp: pd.Timestamp,
        underlying_price: float,
    ) -> Tuple[float, float]:
        trade_date = timestamp.tz_convert("Asia/Kolkata").normalize()
        vol_series = context.vol_series
        if trade_date not in vol_series.index:
            vol_value = float(vol_series.ffill().iloc[-1])
        else:
            vol_value = float(vol_series.loc[trade_date])
        vol_value = float(np.clip(vol_value, self.config.pricing.synthetic.vol_floor, self.config.pricing.synthetic.vol_cap))
        time_to_expiry = max(_time_to_expiry(contract.expiry, timestamp), 1e-6)
        bs_type = _option_type_to_label(contract.option_type)
        synthetic_price = self.black_scholes.calculate_option_price(
            S=float(underlying_price),
            K=float(contract.strike),
            T=time_to_expiry,
            r=self.config.pricing.synthetic.risk_free_rate,
            sigma=vol_value,
            option_type=bs_type,
        )
        return synthetic_price, vol_value

    def price(
        self,
        contract: OptionContractSpec,
        timestamp: pd.Timestamp,
        underlying_price: float,
        allow_skip: bool = True,
    ) -> Tuple[PricingEvent, Optional[str]]:
        """
        Compute price at a given timestamp for the provided contract.

        Returns:
            PricingEvent, fallback_reason (if any)
        """
        ctx = self._ensure_context(contract.expiry)
        synthetic_price, sigma = self._compute_synthetic(ctx, contract, timestamp, underlying_price)
        price_used = float(synthetic_price)
        implied_vol = float(sigma)
        pricing_mode = "synthetic"
        fallback_reason: Optional[str] = None
        actual_meta: Dict[str, object] = {}
        actual_price: Optional[float] = None
        time_to_expiry = max(_time_to_expiry(contract.expiry, timestamp), 1e-6)
        dte_days = time_to_expiry * 365.0

        if self.config.pricing.mode in {"actual", "hybrid"}:
            try:
                actual_price, actual_meta = self._extract_actual_price(ctx, contract, timestamp, allow_skip)
            except LiquidityFilterError:
                raise
            if actual_price is not None:
                pricing_mode = "actual"
            else:
                fallback_reason = actual_meta.get("reason") or "actual_price_missing"
                if self.config.pricing.mode == "actual" and not fallback_reason:
                    fallback_reason = "actual_price_missing"
                attempts = actual_meta.get("attempts")
                if attempts and fallback_reason:
                    missing_tfs = [attempt.get("timeframe") for attempt in attempts if attempt.get("reason")]
                    missing_tfs = [tf for tf in missing_tfs if tf]
                    if missing_tfs:
                        fallback_reason = f"{fallback_reason}({'/'.join(missing_tfs)})"

        notes: Dict[str, object] = {
            "synthetic_price": float(synthetic_price),
            "synthetic_vol": float(sigma),
            "vol_source_timeframe": ctx.source_timeframe,
            "actual_details": actual_meta,
            "fallback_reason": fallback_reason,
            "time_to_expiry_days": float(dte_days),
        }

        if actual_price is not None:
            price_used = float(actual_price)
            # Attempt implied volatility inversion for auditability.
            bs_type = _option_type_to_label(contract.option_type)
            try:
                implied_vol = self.black_scholes.calculate_implied_volatility(
                    option_price=price_used,
                    S=float(underlying_price),
                    K=float(contract.strike),
                    T=time_to_expiry,
                    r=self.config.pricing.synthetic.risk_free_rate,
                    option_type=bs_type,
                )
            except Exception:
                implied_vol = float(sigma)
            if np.isnan(implied_vol):
                implied_vol = float(sigma)
            notes["actual_price"] = price_used
            notes["implied_vol_observed"] = float(implied_vol)
            notes["calibration_error"] = float(price_used - synthetic_price)
        event = PricingEvent(
            timestamp=timestamp,
            price=float(price_used),
            pricing_mode=pricing_mode,
            implied_vol=None if np.isnan(implied_vol) else float(implied_vol),
            underlying_price=float(underlying_price),
            notes=notes,
        )
        return event, fallback_reason

    def compute_leg_costs(
        self,
        *,
        price: float,
        quantity: int,
        side: str,
        exchange: Optional[str] = None,
        intrinsic_value: Optional[float] = None,
        exercised: bool = False,
    ) -> LegCostBreakdown:
        """
        Calculate transaction costs for a single execution leg.
        """
        premium_turnover = max(float(price) * float(quantity), 0.0)
        return self._cost_model.calculate_leg_costs(
            premium_turnover=premium_turnover,
            side=side,
            exchange=exchange,
            intrinsic_value=intrinsic_value,
            exercised=exercised,
        )

    def price_path(
        self,
        contract: OptionContractSpec,
        timestamps: Sequence[pd.Timestamp],
        underlying_prices: Sequence[float],
        include_greeks: bool = True,
    ) -> Tuple[List[OptionPositionSnapshot], List[str]]:
        """
        Price the entire lifecycle of a trade.

        Returns a list of snapshots and list of fallback reasons encountered.
        """
        snapshots: List[OptionPositionSnapshot] = []
        fallbacks: List[str] = []
        greeks_to_compute = [metric.lower() for metric in self.config.greeks.metrics]

        for ts, spot in zip(timestamps, underlying_prices):
            try:
                event, fallback = self.price(contract, ts, spot, allow_skip=False)
            except LiquidityFilterError as exc:
                ctx = self._ensure_context(contract.expiry)
                synthetic_price, sigma = self._compute_synthetic(ctx, contract, ts, spot)
                event = PricingEvent(
                    timestamp=ts,
                    price=float(synthetic_price),
                    pricing_mode="synthetic",
                    implied_vol=float(sigma),
                    underlying_price=float(spot),
                    notes={
                        "synthetic_price": float(synthetic_price),
                        "synthetic_vol": float(sigma),
                        "vol_source_timeframe": ctx.source_timeframe,
                        "actual_details": {
                            "liquidity_failures": exc.reasons,
                            "source": "actual_cache",
                            "reason": "liquidity_violation",
                        },
                        "fallback_reason": "liquidity_violation",
                    },
                )
                fallback = "liquidity_violation"
            if fallback:
                fallbacks.append(fallback)
            greeks: Dict[str, float] = {}
            if include_greeks:
                time_to_expiry = max(_time_to_expiry(contract.expiry, ts), 1e-6)
                bs_type = _option_type_to_label(contract.option_type)
                sigma_source = event.implied_vol
                if sigma_source is None or np.isnan(sigma_source):
                    sigma_source = float(event.notes.get("synthetic_vol", self.config.pricing.synthetic.vol_floor))
                greeks_all = self.black_scholes.calculate_greeks(
                    S=float(spot),
                    K=float(contract.strike),
                    T=time_to_expiry,
                    r=self.config.pricing.synthetic.risk_free_rate,
                    sigma=float(sigma_source),
                    option_type=bs_type,
                )
                for metric in greeks_to_compute:
                    key = metric.lower()
                    if key in greeks_all:
                        greeks[key] = float(greeks_all[key])
            snapshots.append(
                OptionPositionSnapshot(
                    timestamp=ts,
                    option_price=float(event.price),
                    underlying_price=float(spot),
                    dte=_time_to_expiry(contract.expiry, ts),
                    greeks=greeks,
                    pricing_mode=event.pricing_mode,
                )
            )
        return snapshots, fallbacks
