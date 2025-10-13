"""
Trade mapping utilities for the options replay engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from src.core.options.data.schemas import OptionType
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
    expiries = option_store.list_expiries()
    if not expiries:
        raise ValueError("No expiries available in option store")
    entry_cfg = config.position_management.entry
    min_dte = entry_cfg.min_dte_to_enter
    max_dte = entry_cfg.max_dte_to_enter
    selected: Optional[pd.Timestamp] = None
    entry_day = trade.entry_time.tz_convert(config.inputs.timezone).normalize()
    for expiry in expiries:
        expiry_day = expiry.tz_convert(config.inputs.timezone).normalize()
        dte_days = (expiry_day - entry_day).days
        if dte_days < min_dte:
            continue
        if dte_days > max_dte:
            continue
        selected = expiry
        break
    if selected is None:
        raise ValueError(
            f"No expiry matched DTE constraints [{min_dte}, {max_dte}] for trade {trade.trade_id}"
        )
    expiry_metadata = option_store.get_metadata(selected)
    expiry_ts = selected.tz_convert(config.inputs.timezone).normalize().replace(hour=15, minute=30)
    return expiry_ts, {
        "expiry_type": expiry_metadata.expiry_type,
        "dte_days": (expiry_ts.normalize() - entry_day).days,
    }


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
    if config.strike_selection.method != "atm":
        raise NotImplementedError(f"Strike selection method '{config.strike_selection.method}' not implemented in MVP")
    strike = option_store.get_nearest_strike(expiry_ts, option_type, underlying_entry_price)
    lot_size = option_store.get_metadata(expiry_ts).lot_size
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
        "strike_method": config.strike_selection.method,
        "strike_reference_price": underlying_entry_price,
    }
    return MappingResult(contract=contract, metadata=metadata)
