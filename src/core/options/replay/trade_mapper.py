"""
Trade mapping utilities for the options replay engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import pandas as pd

from src.core.options.data.schemas import OptionType
from src.core.options.options_engine import BlackScholesEngine
from src.core.options.replay.config import OptionsReplayConfig
from .models import EquityTrade, OptionContractSpec
from .data_loader import OptionDataStore


@dataclass
class MappingResult:
    contract: OptionContractSpec
    metadata: Dict[str, object]


def _pick_option_type(config: OptionsReplayConfig, trade: EquityTrade) -> OptionType:
    signal = trade.side.upper()
    if signal == "LONG":
        return OptionType(config.option_type.long_signal)
    return OptionType(config.option_type.short_signal)


def _choose_expiry(
    config: OptionsReplayConfig,
    option_store: OptionDataStore,
    trade: EquityTrade,
) -> Tuple[pd.Timestamp, Dict[str, object]]:
    expiries = option_store.list_expiries(require_data=True)
    if not expiries:
        raise ValueError("No expiries available in option store")

    timezone = config.inputs.timezone
    entry_day = trade.entry_time.tz_convert(timezone).normalize()
    entry_cfg = config.position_management.entry
    min_dte = entry_cfg.min_dte_to_enter
    max_dte = entry_cfg.max_dte_to_enter

    info = []
    for expiry in expiries:
        metadata = option_store.get_metadata(expiry)
        expiry_day = expiry.tz_convert(timezone).normalize()
        dte_days = (expiry_day - entry_day).days
        info.append(
            {
                "expiry": expiry,
                "metadata": metadata,
                "expiry_day": expiry_day,
                "dte_days": dte_days,
            }
        )

    def within_bounds(items, min_days, max_days):
        return [
            item for item in items if min_days <= item["dte_days"] <= max_days and item["dte_days"] >= 0
        ]

    candidates = within_bounds(info, min_dte, max_dte)
    method = config.expiry_selection.method
    selection_details: Dict[str, object] = {
        "method": method,
        "available_expiry_count": len(info),
        "available_expiry_sample": [item["expiry"].isoformat() for item in info[:5]],
        "candidates_considered": len(candidates),
        "min_dte": min_dte,
        "max_dte": max_dte,
    }

    selected_item: Optional[Dict[str, object]] = None

    if method in {"nearest_weekly", "nearest_monthly"}:
        expiry_type = "weekly" if method == "nearest_weekly" else "monthly"
        typed = [item for item in candidates if item["metadata"].expiry_type.lower() == expiry_type]
        selection_details["filtered_by_type"] = len(typed)
        if typed:
            selected_item = min(typed, key=lambda item: (item["dte_days"], item["expiry"]))
        elif candidates:
            selection_details["fallback_reason"] = f"no_{expiry_type}_expiry_found"
            selected_item = min(candidates, key=lambda item: (item["dte_days"], item["expiry"]))

    elif method == "fixed_dte":
        target = config.expiry_selection.fixed_dte_target
        tolerance = config.expiry_selection.fixed_dte_tolerance or 0
        if target is not None:
            eligible = [
                item for item in candidates if abs(item["dte_days"] - target) <= tolerance
            ]
            selection_details["filtered_by_type"] = len(eligible)
            if eligible:
                selected_item = min(
                    eligible,
                    key=lambda item: (abs(item["dte_days"] - target), item["expiry"]),
                )
            elif candidates:
                selection_details["fallback_reason"] = "no_expiry_within_fixed_dte_range"
                selected_item = min(candidates, key=lambda item: (abs(item["dte_days"] - target), item["expiry"]))

    elif method == "next_expiry":
        next_cfg = config.expiry_selection.next_expiry
        min_next = next_cfg.min_dte or min_dte
        max_next = next_cfg.max_dte or max_dte
        eligible = within_bounds(info, min_next, max_next)
        selection_details["filtered_by_type"] = len(eligible)
        if not eligible:
            selection_details["fallback_reason"] = "no_expiry_within_next_expiry_bounds"
            eligible = candidates
        if eligible:
            selected_item = min(eligible, key=lambda item: (item["expiry_day"], item["expiry"]))

    if selected_item is None and candidates:
        selection_details.setdefault("fallback_reason", "default_min_dte_match")
        selected_item = min(candidates, key=lambda item: (item["expiry_day"], item["expiry"]))

    if selected_item is None:
        # No expiry satisfied constraints; pick earliest available to avoid hard failure.
        selection_details["fallback_reason"] = "no_expiry_meets_constraints"
        selected_item = min(info, key=lambda item: (item["expiry_day"], item["expiry"]))

    selection_details["selected_dte_days"] = selected_item["dte_days"]
    selection_details["dte_days"] = selected_item["dte_days"]
    selection_details["selected_expiry"] = selected_item["expiry"].isoformat()
    selection_details["total_expiries_available"] = len(info)
    selected_token = selected_item["expiry"].tz_convert("Asia/Kolkata").normalize().strftime("%Y-%m-%d")
    data_path = option_store.base_dir / option_store.timeframe / f"expiry_{selected_token}.parquet"
    selection_details["data_file_exists"] = data_path.exists()

    expiry_metadata = option_store.get_metadata(selected_item["expiry"])
    selection_details["expiry_type"] = expiry_metadata.expiry_type

    expiry_ts = selected_item["expiry"].tz_convert(timezone).normalize().replace(hour=15, minute=30)
    return expiry_ts, selection_details


def map_trade_to_option(
    config: OptionsReplayConfig,
    option_store: OptionDataStore,
    trade: EquityTrade,
    underlying_entry_price: float,
) -> MappingResult:
    """
    Map an equity trade to an option contract according to config.
    """
    option_type = _pick_option_type(config, trade)
    expiry_ts, expiry_meta = _choose_expiry(config, option_store, trade)

    strike, strike_meta = _choose_strike(
        config=config,
        option_store=option_store,
        option_type=option_type,
        expiry_ts=expiry_ts,
        underlying_price=underlying_entry_price,
        trade_entry_time=trade.entry_time,
    )
    expiry_metadata = option_store.get_metadata(expiry_ts)
    lot_size = expiry_metadata.lot_size
    contract = OptionContractSpec(
        ticker=trade.ticker,
        expiry=expiry_ts,
        strike=float(strike),
        option_type=option_type,
        lot_size=int(lot_size),
    )
    metadata = {
        "option_type": option_type.value,
        "expiry_info": expiry_meta,
        "strike_info": strike_meta,
        "strike_reference_price": underlying_entry_price,
    }
    return MappingResult(contract=contract, metadata=metadata)


def _choose_strike(
    config: OptionsReplayConfig,
    option_store: OptionDataStore,
    option_type: OptionType,
    expiry_ts: pd.Timestamp,
    underlying_price: float,
    trade_entry_time: pd.Timestamp,
) -> Tuple[float, Dict[str, object]]:
    method = config.strike_selection.method
    if method == "atm":
        return _select_strike_atm(option_store, option_type, expiry_ts, underlying_price)
    if method == "delta":
        return _select_strike_delta(
            config=config,
            option_store=option_store,
            option_type=option_type,
            expiry_ts=expiry_ts,
            underlying_price=underlying_price,
            trade_entry_time=trade_entry_time,
        )
    if method == "premium_pct":
        return _select_strike_premium_pct(
            config=config,
            option_store=option_store,
            option_type=option_type,
            expiry_ts=expiry_ts,
            underlying_price=underlying_price,
            trade_entry_time=trade_entry_time,
        )
    raise NotImplementedError(f"Strike selection method '{method}' is not implemented yet")


def _select_strike_atm(
    option_store: OptionDataStore,
    option_type: OptionType,
    expiry_ts: pd.Timestamp,
    underlying_price: float,
) -> Tuple[float, Dict[str, object]]:
    strike = option_store.get_nearest_strike(expiry_ts, option_type, underlying_price)
    return float(strike), {
        "method": "atm",
        "reason": "nearest_strike_to_underlying",
        "strike": float(strike),
    }


def _select_strike_delta(
    config: OptionsReplayConfig,
    option_store: OptionDataStore,
    option_type: OptionType,
    expiry_ts: pd.Timestamp,
    underlying_price: float,
    trade_entry_time: pd.Timestamp,
) -> Tuple[float, Dict[str, object]]:
    delta_cfg = config.strike_selection.delta
    if delta_cfg is None:
        raise ValueError("strike_selection.delta configuration missing for delta method")

    expiry_metadata = option_store.get_metadata(expiry_ts)
    if option_type == OptionType.CALL:
        strikes = expiry_metadata.call_strikes
    else:
        strikes = expiry_metadata.put_strikes

    # Fallback to chain scan if metadata doesn't include strikes
    if not strikes:
        chain = option_store.load_chain(expiry_ts)
        subset = chain[chain["option_type"] == option_type.value]
        strikes = subset["strike"].unique().tolist()

    if not strikes:
        raise ValueError(f"No strikes available for {option_type.value} at expiry {expiry_ts.date()}")

    strikes = sorted(float(value) for value in strikes)
    time_to_expiry = max(
        (expiry_ts - trade_entry_time).total_seconds() / (365.0 * 24.0 * 3600.0),
        1e-6,
    )

    bs_engine = BlackScholesEngine(risk_free_rate=config.pricing.synthetic.risk_free_rate)
    sigma = float(delta_cfg.volatility)
    sigma = min(max(sigma, config.pricing.synthetic.vol_floor), config.pricing.synthetic.vol_cap)
    target_delta = float(delta_cfg.target)
    tolerance = float(delta_cfg.tolerance)

    best: Optional[Tuple[float, float, float]] = None  # (strike, effective_delta, diff)
    candidates: List[Dict[str, float]] = []
    for strike in strikes:
        greeks = bs_engine.calculate_greeks(
            S=underlying_price,
            K=strike,
            T=time_to_expiry,
            r=bs_engine.risk_free_rate,
            sigma=sigma,
            option_type="call" if option_type == OptionType.CALL else "put",
        )
        raw_delta = float(greeks["delta"])
        effective_delta = raw_delta if option_type == OptionType.CALL else abs(raw_delta)
        diff = abs(effective_delta - target_delta)
        candidates.append(
            {
                "strike": float(strike),
                "delta": effective_delta,
                "delta_diff": diff,
            }
        )
        if best is None or diff < best[2]:
            best = (float(strike), effective_delta, diff)

    if best is None:
        raise ValueError("Unable to evaluate strikes for delta selection")

    strike, selected_delta, diff = best
    within_tolerance = diff <= tolerance

    if within_tolerance:
        return strike, {
            "method": "delta",
        "selection_status": "within_tolerance",
        "target_delta": target_delta,
        "selected_delta": selected_delta,
        "delta_diff": diff,
        "volatility_used": sigma,
        "time_to_expiry_years": time_to_expiry,
        "candidates_considered": candidates,
    }

    atm_strike, atm_meta = _select_strike_atm(option_store, option_type, expiry_ts, underlying_price)
    return atm_strike, {
        "method": "delta",
        "selection_status": "fallback_atm",
        "target_delta": target_delta,
        "selected_delta": selected_delta,
        "delta_diff": diff,
        "volatility_used": sigma,
        "time_to_expiry_years": time_to_expiry,
        "fallback_reason": f"delta_diff_exceeds_tolerance_{tolerance}",
        "fallback_strike": atm_strike,
        "fallback_details": atm_meta,
        "candidates_considered": candidates,
    }


def _select_strike_premium_pct(
    config: OptionsReplayConfig,
    option_store: OptionDataStore,
    option_type: OptionType,
    expiry_ts: pd.Timestamp,
    underlying_price: float,
    trade_entry_time: pd.Timestamp,
) -> Tuple[float, Dict[str, object]]:
    premium_cfg = config.strike_selection.premium_pct
    if hasattr(premium_cfg, "get"):
        target_pct = float(premium_cfg.get("target", 0.0))  # type: ignore[attr-defined]
        tolerance_pct = float(premium_cfg.get("tolerance", 0.0))  # type: ignore[attr-defined]
    else:
        target_pct = float(getattr(premium_cfg, "target", 0.0))
        tolerance_pct = float(getattr(premium_cfg, "tolerance", 0.0))
    target_price = underlying_price * target_pct
    tolerance_abs = underlying_price * tolerance_pct

    chain = option_store.load_chain(expiry_ts)
    subset = chain[chain["option_type"] == option_type.value].copy()
    if subset.empty:
        raise ValueError(f"No option data available for {option_type.value} at expiry {expiry_ts.date()}")

    entry_ts = trade_entry_time.tz_convert(config.inputs.timezone)
    subset["timestamp"] = subset["timestamp"].dt.tz_convert(config.inputs.timezone)
    subset = subset.sort_values(["strike", "timestamp"])

    latest_rows = subset[subset["timestamp"] <= entry_ts]
    if latest_rows.empty:
        latest_rows = subset
    latest_by_strike = latest_rows.groupby("strike").tail(1)

    latest_by_strike = latest_by_strike.copy()
    latest_by_strike.loc[:, "premium_diff"] = (latest_by_strike["close"] - target_price).abs()
    best_row = latest_by_strike.iloc[0]
    for _, row in latest_by_strike.iterrows():
        if row["premium_diff"] < best_row["premium_diff"]:
            best_row = row

    diff_abs = float(best_row["premium_diff"])
    within_tolerance = diff_abs <= tolerance_abs if tolerance_abs > 0 else diff_abs == 0.0

    metadata = {
        "method": "premium_pct",
        "target_premium": target_price,
        "selected_premium": float(best_row["close"]),
        "premium_diff": diff_abs,
        "candidate_count": int(len(latest_by_strike)),
    }

    if within_tolerance:
        metadata["selection_status"] = "within_tolerance"
        metadata["fallback_reason"] = None
        return float(best_row["strike"]), metadata

    atm_strike, atm_meta = _select_strike_atm(option_store, option_type, expiry_ts, underlying_price)
    metadata["selection_status"] = "fallback_atm"
    metadata["fallback_reason"] = "premium_diff_exceeds_tolerance"
    metadata["fallback_strike"] = atm_strike
    metadata["fallback_details"] = atm_meta
    return atm_strike, metadata
