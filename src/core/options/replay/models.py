"""
Core data models for the options replay engine.

These lightweight ``dataclass`` definitions enforce schema contracts for
equity trades, option mapping artefacts, pricing decisions, and replay
outputs.  They enable type checking throughout the pipeline and provide a
consistent interface for serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from src.core.options.data.schemas import OptionType


def ensure_timestamp(value: pd.Timestamp | datetime | str, tz: str = "Asia/Kolkata") -> pd.Timestamp:
    """
    Convert user supplied timestamps into timezone-aware pandas Timestamps.
    """
    if isinstance(value, pd.Timestamp):
        ts = value
    else:
        ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(tz)
    else:
        ts = ts.tz_convert(tz)
    return ts


@dataclass(frozen=True)
class EquityTrade:
    """Single equity trade to be replayed using options contracts."""

    trade_id: str
    ticker: str
    side: str  # LONG | SHORT
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float

    @staticmethod
    def from_row(row: Dict[str, object], tz: str = "Asia/Kolkata") -> "EquityTrade":
        """Create an ``EquityTrade`` from a pandas row/dict."""
        required = [
            "trade_id",
            "ticker",
            "side",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "quantity",
            "pnl",
        ]
        missing = [col for col in required if col not in row or pd.isna(row[col])]
        if missing:
            raise ValueError(f"Cannot create EquityTrade; missing fields: {missing}")

        side = str(row["side"]).upper()
        if side not in {"LONG", "SHORT", "BUY", "SELL"}:
            raise ValueError(f"Unsupported trade side: {side}")
        side = "LONG" if side in {"LONG", "BUY"} else "SHORT"

        return EquityTrade(
            trade_id=str(row["trade_id"]),
            ticker=str(row["ticker"]).upper(),
            side=side,
            entry_time=ensure_timestamp(row["entry_time"], tz),
            exit_time=ensure_timestamp(row["exit_time"], tz),
            entry_price=float(row["entry_price"]),
            exit_price=float(row["exit_price"]),
            quantity=float(row["quantity"]),
            pnl=float(row["pnl"]),
        )


@dataclass(frozen=True)
class OptionContractSpec:
    """Mapped option contract for a replayed trade."""

    ticker: str
    expiry: pd.Timestamp
    strike: float
    option_type: OptionType
    lot_size: int
    symbol: Optional[str] = None
    instrument_key: Optional[str] = None


@dataclass(frozen=True)
class PricingEvent:
    """Single pricing decision (entry/exit or mark-to-market)."""

    timestamp: pd.Timestamp
    price: float
    pricing_mode: str  # synthetic | actual
    implied_vol: Optional[float] = None
    underlying_price: Optional[float] = None
    notes: Dict[str, object] = field(default_factory=dict)


@dataclass
class OptionPositionSnapshot:
    """Mark-to-market snapshot for an open option position."""

    timestamp: pd.Timestamp
    option_price: float
    underlying_price: float
    dte: float
    greeks: Dict[str, float]
    pricing_mode: str


@dataclass
class ReplayTradeResult:
    """Final result for a replayed trade."""

    equity_trade: EquityTrade
    contract: OptionContractSpec
    lots: int
    quantity: int
    entry: PricingEvent
    exit: PricingEvent
    lifecycle: List[OptionPositionSnapshot]
    realized_pnl: float
    return_pct: float
    max_drawdown_pct: Optional[float]
    risk_flags: List[str] = field(default_factory=list)
    pricing_fallbacks: List[str] = field(default_factory=list)


@dataclass
class RiskEvent:
    """Represents an enforcement or breach of a configured risk guard."""

    timestamp: pd.Timestamp
    event_type: str
    message: str
    details: Dict[str, object] = field(default_factory=dict)


@dataclass
class ReplayRunArtifacts:
    """
    Aggregated artefacts produced by a replay run.
    """

    run_id: str
    config_hash: str
    trades: List[ReplayTradeResult]
    risk_events: List[RiskEvent]
    skipped_trades: List[Dict[str, object]]
    metadata: Dict[str, object]

