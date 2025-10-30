"""
Typed configuration loader for the Phase 3 options replay engine.

This module converts the YAML configuration defined in
``src/core/options/config/options_config.yaml``into strongly typed
dataclasses with light-weight validation.  The resulting objects are used
throughout the replay workflow to guarantee schema contracts and provide
clear typing for downstream components.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

CONFIG_TIMEZONE_DEFAULT = "Asia/Kolkata"


def _ensure_path(value: str, base_dir: Optional[Path] = None) -> Path:
    """Resolve a relative path against the repository root."""
    path = Path(value)
    if not path.is_absolute() and base_dir is not None:
        path = (base_dir / path).resolve()
    return path


def _parse_timeframes(raw_timeframes: Any) -> Tuple[str, ...]:
    """Normalise timeframe configuration into a tuple ordered by resolution."""
    if isinstance(raw_timeframes, str):
        candidates = [part.strip() for part in raw_timeframes.replace(",", " ").split() if part.strip()]
    elif isinstance(raw_timeframes, (list, tuple)):
        candidates = [str(part).strip() for part in raw_timeframes if str(part).strip()]
    else:
        raise TypeError("inputs.options_timeframe must be a string or list of strings")
    if not candidates:
        raise ValueError("inputs.options_timeframe must include at least one timeframe")

    def _resolution_seconds(label: str) -> int:
        label = label.lower()
        if label.endswith("minute"):
            return int(label.replace("minute", "")) * 60
        if label.endswith("min"):
            return int(label.replace("min", "")) * 60
        if label.endswith("hour"):
            return int(label.replace("hour", "")) * 3600
        if label.endswith("day") or label.endswith("d"):
            return int(label.split("day")[0] or "1") * 86400
        if label.endswith("week") or label.endswith("w"):
            return int(label.split("week")[0] or "1") * 604800
        # Fallback: treat as daily
        return 86400

    seen: set[str] = set()
    normalised: List[str] = []
    for candidate in candidates:
        lower = candidate.lower()
        if lower not in seen:
            seen.add(lower)
            normalised.append(lower)
    normalised.sort(key=_resolution_seconds)
    return tuple(normalised)


@dataclass(frozen=True)
class InputsConfig:
    """Configuration describing input artefacts consumed by the replay."""

    equity_trades_path: Path
    equity_data_root: Path
    options_data_root: Path
    underlying_timeframe: str
    options_timeframe: str
    options_timeframes: Tuple[str, ...] = ("1day",)
    timezone: str = CONFIG_TIMEZONE_DEFAULT
    seed: int = 42
    run_label: str = "replay_run"
    ticker_whitelist: Optional[List[str]] = None

    @staticmethod
    def from_dict(data: Dict[str, Any], base_dir: Path) -> "InputsConfig":
        """Create an ``InputsConfig`` instance from a raw dictionary."""
        required = [
            "equity_trades_path",
            "equity_data_root",
            "options_data_root",
            "underlying_timeframe",
            "options_timeframe",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Missing required inputs config keys: {missing}")

        whitelist = data.get("ticker_whitelist")
        if whitelist is not None:
            if not isinstance(whitelist, list):
                raise TypeError("inputs.ticker_whitelist must be a list when provided")
            whitelist = [str(t).upper() for t in whitelist]

        option_timeframes = _parse_timeframes(data["options_timeframe"])
        return InputsConfig(
            equity_trades_path=_ensure_path(data["equity_trades_path"], base_dir),
            equity_data_root=_ensure_path(data["equity_data_root"], base_dir),
            options_data_root=_ensure_path(data["options_data_root"], base_dir),
            underlying_timeframe=str(data["underlying_timeframe"]),
            options_timeframe=option_timeframes[0],
            options_timeframes=option_timeframes,
            timezone=str(data.get("timezone", CONFIG_TIMEZONE_DEFAULT)),
            seed=int(data.get("seed", 42)),
            run_label=str(data.get("run_label", "replay_run")),
            ticker_whitelist=whitelist,
        )


@dataclass(frozen=True)
class SyntheticPricingConfig:
    """Synthetic pricing parameters used by the hybrid pricer."""

    volatility_model: str = "historical_20d"
    risk_free_rate: float = 0.06
    dividend_yield: float = 0.0
    vol_floor: float = 0.10
    vol_cap: float = 1.50

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SyntheticPricingConfig":
        return SyntheticPricingConfig(
            volatility_model=str(data.get("volatility_model", "historical_20d")),
            risk_free_rate=float(data.get("risk_free_rate", 0.06)),
            dividend_yield=float(data.get("dividend_yield", 0.0)),
            vol_floor=float(data.get("vol_floor", 0.10)),
            vol_cap=float(data.get("vol_cap", 1.50)),
        )


@dataclass(frozen=True)
class ActualPricingConfig:
    """Configuration controlling fills when using cached option bars."""

    fill_assumption: str = "mid"
    cache_dir: Optional[Path] = None

    @staticmethod
    def from_dict(data: Dict[str, Any], base_dir: Path) -> "ActualPricingConfig":
        cache_dir = data.get("cache_dir")
        return ActualPricingConfig(
            fill_assumption=str(data.get("fill_assumption", "mid")).lower(),
            cache_dir=_ensure_path(cache_dir, base_dir) if cache_dir else None,
        )


@dataclass(frozen=True)
class BrokerageChargesConfig:
    """Brokerage charge parameters per executed order."""

    flat_per_order: Optional[float]
    pct_of_premium: Optional[float]
    max_per_order: Optional[float]

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "BrokerageChargesConfig":
        def _as_optional_float(value: Any) -> Optional[float]:
            if value is None:
                return None
            converted = float(value)
            return converted if converted >= 0 else None

        flat = _as_optional_float(data.get("flat_per_order", 20.0))
        pct = _as_optional_float(data.get("pct_of_premium"))
        cap = _as_optional_float(data.get("max_per_order"))
        return BrokerageChargesConfig(
            flat_per_order=flat,
            pct_of_premium=pct,
            max_per_order=cap,
        )


@dataclass(frozen=True)
class TransactionCostConfig:
    """
    Deterministic transaction cost parameters for option trades.
    """

    enabled: bool
    default_exchange: str
    brokerage: BrokerageChargesConfig
    exchange_transaction_rates: Dict[str, float]
    stt_sell_rate: float
    stt_exercise_rate: float
    stamp_buy_rate: float
    sebi_rate: float
    sebi_minimum: float
    clearing_rate: float
    gst_rate: float
    gst_apply_to: Tuple[str, ...]
    rounding_precision: int

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TransactionCostConfig":
        enabled = bool(data.get("enabled", False))
        brokerage_cfg = BrokerageChargesConfig.from_dict(data.get("brokerage", {}))
        exchange_rates_raw = data.get(
            "exchange_transaction_rate",
            {"NSE": 0.0003503, "BSE": 0.000325},
        )
        exchange_rates = {
            str(name).upper(): float(rate)
            for name, rate in (exchange_rates_raw or {}).items()
        }
        if not exchange_rates:
            exchange_rates = {"NSE": 0.0}
        default_exchange = str(data.get("exchange", "NSE")).upper()
        if default_exchange not in exchange_rates:
            exchange_rates.setdefault(default_exchange, 0.0)
        gst_cfg = data.get("gst", {})
        gst_apply = tuple(
            str(component).lower()
            for component in gst_cfg.get("include_components", ["brokerage", "exchange_transaction"])
        )
        return TransactionCostConfig(
            enabled=enabled,
            default_exchange=default_exchange,
            brokerage=brokerage_cfg,
            exchange_transaction_rates=exchange_rates,
            stt_sell_rate=float(data.get("stt", {}).get("sell_premium_rate", 0.001)),
            stt_exercise_rate=float(data.get("stt", {}).get("exercise_intrinsic_rate", 0.00125)),
            stamp_buy_rate=float(data.get("stamp_duty_buy_rate", 0.00003)),
            sebi_rate=float(data.get("sebi", {}).get("rate", 0.000001)),
            sebi_minimum=float(data.get("sebi", {}).get("minimum", 10.0)),
            clearing_rate=float(data.get("clearing_rate", 0.0)),
            gst_rate=float(gst_cfg.get("rate", 0.18)),
            gst_apply_to=gst_apply,
            rounding_precision=int(data.get("rounding_precision", 2)),
        )


@dataclass(frozen=True)
class SlippageConfig:
    """Simple slippage assumptions applied per leg."""

    enabled: bool
    mode: str
    per_side_bps: float

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SlippageConfig":
        enabled = bool(data.get("enabled", False))
        mode = str(data.get("mode", "bps")).lower()
        if mode != "bps":
            raise ValueError("pricing.slippage.mode currently supports only 'bps'")
        return SlippageConfig(
            enabled=enabled,
            mode=mode,
            per_side_bps=float(data.get("per_side_bps", 0.0)),
        )


@dataclass(frozen=True)
class PricingConfig:
    """Unified pricing configuration."""

    mode: str
    synthetic: SyntheticPricingConfig
    actual: ActualPricingConfig
    costs: TransactionCostConfig
    slippage: SlippageConfig

    @staticmethod
    def from_dict(data: Dict[str, Any], base_dir: Path) -> "PricingConfig":
        if "mode" not in data:
            raise ValueError("pricing.mode must be specified")
        mode = str(data["mode"]).lower()
        if mode not in {"synthetic", "actual", "hybrid"}:
            raise ValueError(f"Unsupported pricing mode: {mode}")
        synthetic = SyntheticPricingConfig.from_dict(data.get("synthetic", {}))
        actual = ActualPricingConfig.from_dict(data.get("actual", {}), base_dir)
        costs = TransactionCostConfig.from_dict(data.get("costs", {}))
        slippage = SlippageConfig.from_dict(data.get("slippage", {}))
        return PricingConfig(
            mode=mode,
            synthetic=synthetic,
            actual=actual,
            costs=costs,
            slippage=slippage,
        )


@dataclass(frozen=True)
class DeltaSelectionConfig:
    target: float
    tolerance: float
    volatility: float

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "DeltaSelectionConfig":
        target = float(data.get("target", 0.3))
        tolerance = float(data.get("tolerance", 0.05))
        volatility = float(data.get("volatility", 0.25))
        if target <= 0 or target >= 1:
            raise ValueError("strike_selection.delta.target must be between 0 and 1")
        if tolerance <= 0 or tolerance >= 1:
            raise ValueError("strike_selection.delta.tolerance must be between 0 and 1")
        if volatility <= 0:
            raise ValueError("strike_selection.delta.volatility must be positive")
        return DeltaSelectionConfig(target=target, tolerance=tolerance, volatility=volatility)


@dataclass(frozen=True)
class StrikeSelectionConfig:
    """Strike selection behaviour."""

    method: str
    delta: DeltaSelectionConfig | None = None
    moneyness: Dict[str, Any] = field(default_factory=dict)
    premium_pct: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "StrikeSelectionConfig":
        if "method" not in data:
            raise ValueError("strike_selection.method must be provided")
        method = str(data["method"]).lower()
        if method not in {"atm", "delta", "moneyness", "premium_pct"}:
            raise ValueError(f"Unsupported strike selection method: {method}")
        delta_cfg = None
        if "delta" in data:
            delta_cfg = DeltaSelectionConfig.from_dict(data["delta"])
        return StrikeSelectionConfig(
            method=method,
            delta=delta_cfg,
            moneyness=dict(data.get("moneyness", {})),
            premium_pct=dict(data.get("premium_pct", {})),
        )


@dataclass(frozen=True)
class ExpirySelectionConfig:
    """Expiry selection heuristics."""

    method: str
    fixed_dte_target: Optional[int] = None
    fixed_dte_tolerance: Optional[int] = None
    next_expiry_min_dte: Optional[int] = None
    next_expiry_max_dte: Optional[int] = None

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ExpirySelectionConfig":
        if "method" not in data:
            raise ValueError("expiry_selection.method must be provided")
        method = str(data["method"]).lower()
        if method not in {"nearest_weekly", "nearest_monthly", "fixed_dte", "next_expiry"}:
            raise ValueError(f"Unsupported expiry selection method: {method}")
        fixed = data.get("fixed_dte", {})
        nxt = data.get("next_expiry", {})
        return ExpirySelectionConfig(
            method=method,
            fixed_dte_target=int(fixed.get("target_days", 0)) or None,
            fixed_dte_tolerance=int(fixed.get("tolerance_days", 0)) or None,
            next_expiry_min_dte=int(nxt.get("min_dte", 0)) or None,
            next_expiry_max_dte=int(nxt.get("max_dte", 0)) or None,
        )


@dataclass(frozen=True)
class OptionTypeConfig:
    """Mapping from equity signals to option type."""

    long_signal: str
    short_signal: str
    strategy: str

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "OptionTypeConfig":
        required = ["long_signal", "short_signal", "strategy"]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Missing option_type keys: {missing}")
        long_signal = str(data["long_signal"]).upper()
        short_signal = str(data["short_signal"]).upper()
        for label in (long_signal, short_signal):
            if label not in {"CE", "PE"}:
                raise ValueError(f"Invalid option signal type: {label}")
        return OptionTypeConfig(
            long_signal=long_signal,
            short_signal=short_signal,
            strategy=str(data["strategy"]).lower(),
        )


@dataclass(frozen=True)
class LotSizingFixedConfig:
    lots_per_trade: int = 1

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "LotSizingFixedConfig":
        lots = int(data.get("lots_per_trade", 1))
        if lots <= 0:
            raise ValueError("lot_sizing.fixed.lots_per_trade must be positive")
        return LotSizingFixedConfig(lots_per_trade=lots)


@dataclass(frozen=True)
class LotSizingConfig:
    method: str
    fixed: LotSizingFixedConfig
    capital_match: Dict[str, Any] = field(default_factory=dict)
    delta_match: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "LotSizingConfig":
        if "method" not in data:
            raise ValueError("lot_sizing.method must be provided")
        method = str(data["method"]).lower()
        if method not in {"fixed", "capital_match", "delta_match"}:
            raise ValueError(f"Unsupported lot sizing method: {method}")
        fixed_cfg = LotSizingFixedConfig.from_dict(data.get("fixed", {}))
        return LotSizingConfig(
            method=method,
            fixed=fixed_cfg,
            capital_match=dict(data.get("capital_match", {})),
            delta_match=dict(data.get("delta_match", {})),
        )


@dataclass(frozen=True)
class PositionEntryConfig:
    min_dte_to_enter: int
    max_dte_to_enter: int
    skip_if_illiquid: bool

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PositionEntryConfig":
        required = ["min_dte_to_enter", "max_dte_to_enter"]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Missing position_management.entry keys: {missing}")
        return PositionEntryConfig(
            min_dte_to_enter=int(data["min_dte_to_enter"]),
            max_dte_to_enter=int(data["max_dte_to_enter"]),
            skip_if_illiquid=bool(data.get("skip_if_illiquid", True)),
        )


@dataclass(frozen=True)
class ForceCloseConfig:
    enabled: bool
    hours_before: int

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ForceCloseConfig":
        return ForceCloseConfig(
            enabled=bool(data.get("enabled", False)),
            hours_before=int(data.get("hours_before", 24)),
        )


@dataclass(frozen=True)
class ThresholdConfig:
    enabled: bool
    threshold_pct: float

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ThresholdConfig":
        return ThresholdConfig(
            enabled=bool(data.get("enabled", False)),
            threshold_pct=float(data.get("threshold_pct", 0.0)),
        )


@dataclass(frozen=True)
class TimeBasedExitConfig:
    enabled: bool
    max_hold_hours: int

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TimeBasedExitConfig":
        return TimeBasedExitConfig(
            enabled=bool(data.get("enabled", False)),
            max_hold_hours=int(data.get("max_hold_hours", 0)),
        )


@dataclass(frozen=True)
class PositionExitConfig:
    follow_equity_signal: bool
    force_close_before_expiry: ForceCloseConfig
    stop_loss: ThresholdConfig
    take_profit: ThresholdConfig
    time_based: TimeBasedExitConfig

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PositionExitConfig":
        return PositionExitConfig(
            follow_equity_signal=bool(data.get("follow_equity_signal", True)),
            force_close_before_expiry=ForceCloseConfig.from_dict(data.get("force_close_before_expiry", {})),
            stop_loss=ThresholdConfig.from_dict(data.get("stop_loss", {})),
            take_profit=ThresholdConfig.from_dict(data.get("take_profit", {})),
            time_based=TimeBasedExitConfig.from_dict(data.get("time_based", {})),
        )


@dataclass(frozen=True)
class PositionManagementConfig:
    entry: PositionEntryConfig
    exit: PositionExitConfig

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PositionManagementConfig":
        return PositionManagementConfig(
            entry=PositionEntryConfig.from_dict(data.get("entry", {})),
            exit=PositionExitConfig.from_dict(data.get("exit", {})),
        )


@dataclass(frozen=True)
class LiquidityConfig:
    min_open_interest: int
    max_spread_pct: float
    min_volume: int
    on_filter_fail: str

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "LiquidityConfig":
        required = ["min_open_interest", "max_spread_pct", "min_volume", "on_filter_fail"]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Missing liquidity keys: {missing}")
        behaviour = str(data["on_filter_fail"]).lower()
        if behaviour not in {"skip_trade", "use_synthetic", "ignore_filters"}:
            raise ValueError(f"Unsupported liquidity.on_filter_fail: {behaviour}")
        return LiquidityConfig(
            min_open_interest=int(data["min_open_interest"]),
            max_spread_pct=float(data["max_spread_pct"]),
            min_volume=int(data["min_volume"]),
            on_filter_fail=behaviour,
        )


@dataclass(frozen=True)
class GreeksConfig:
    calculate: bool
    metrics: List[str]
    frequency: str

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "GreeksConfig":
        metrics = [str(metric) for metric in data.get("metrics", [])]
        frequency = str(data.get("frequency", "every_bar")).lower()
        if frequency not in {"every_bar", "hourly", "entry_exit_only"}:
            raise ValueError(f"Unsupported greeks.frequency: {frequency}")
        return GreeksConfig(
            calculate=bool(data.get("calculate", True)),
            metrics=metrics,
            frequency=frequency,
        )


@dataclass(frozen=True)
class RiskKillSwitchConfig:
    enabled: bool
    max_intraday_loss_pct: float
    max_single_trade_loss_pct: float
    reason_codes: bool

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RiskKillSwitchConfig":
        return RiskKillSwitchConfig(
            enabled=bool(data.get("enabled", False)),
            max_intraday_loss_pct=float(data.get("max_intraday_loss_pct", 0.0)),
            max_single_trade_loss_pct=float(data.get("max_single_trade_loss_pct", 0.0)),
            reason_codes=bool(data.get("reason_codes", False)),
        )


@dataclass(frozen=True)
class AssignmentRiskConfig:
    enabled: bool
    dte_hours_threshold: float
    intrinsic_buffer: float

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AssignmentRiskConfig":
        return AssignmentRiskConfig(
            enabled=bool(data.get("enabled", True)),
            dte_hours_threshold=float(data.get("dte_hours_threshold", 24.0)),
            intrinsic_buffer=float(data.get("intrinsic_buffer", 0.0)),
        )


@dataclass(frozen=True)
class RiskConfig:
    initial_portfolio_value: float
    max_portfolio_allocation: float
    max_concurrent_positions: int
    max_position_size_per_trade: float
    max_drawdown_pct: float
    stop_trading_on_drawdown: bool
    kill_switch: RiskKillSwitchConfig
    assignment_risk: AssignmentRiskConfig

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RiskConfig":
        required = [
            "initial_portfolio_value",
            "max_portfolio_allocation",
            "max_concurrent_positions",
            "max_position_size_per_trade",
            "max_drawdown_pct",
            "stop_trading_on_drawdown",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Missing risk configuration keys: {missing}")
        max_alloc = float(data["max_portfolio_allocation"])
        if not 0.0 <= max_alloc <= 1.0:
            raise ValueError("risk.max_portfolio_allocation must be between 0 and 1")
        max_position = float(data["max_position_size_per_trade"])
        if not 0.0 <= max_position <= 1.0:
            raise ValueError("risk.max_position_size_per_trade must be between 0 and 1")
        kill_switch = RiskKillSwitchConfig.from_dict(data.get("kill_switch", {}))
        assignment_risk = AssignmentRiskConfig.from_dict(data.get("assignment_risk", {}))
        return RiskConfig(
            initial_portfolio_value=float(data["initial_portfolio_value"]),
            max_portfolio_allocation=max_alloc,
            max_concurrent_positions=int(data["max_concurrent_positions"]),
            max_position_size_per_trade=max_position,
            max_drawdown_pct=float(data["max_drawdown_pct"]),
            stop_trading_on_drawdown=bool(data.get("stop_trading_on_drawdown", False)),
            kill_switch=kill_switch,
            assignment_risk=assignment_risk,
        )


@dataclass(frozen=True)
class DataQualityConfig:
    log_pricing_mode: bool
    log_synthetic_fallbacks: bool
    generate_quality_report: bool

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "DataQualityConfig":
        return DataQualityConfig(
            log_pricing_mode=bool(data.get("log_pricing_mode", True)),
            log_synthetic_fallbacks=bool(data.get("log_synthetic_fallbacks", True)),
            generate_quality_report=bool(data.get("generate_quality_report", True)),
        )


@dataclass(frozen=True)
class OutputFilesConfig:
    trades: bool
    base_data: bool
    metrics: bool
    comparison: bool
    position_lifecycle: bool
    pricing_validation: bool
    health_cards: bool
    lifecycle_summary: bool

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "OutputFilesConfig":
        return OutputFilesConfig(
            trades=bool(data.get("trades", True)),
            base_data=bool(data.get("base_data", True)),
            metrics=bool(data.get("metrics", True)),
            comparison=bool(data.get("comparison", True)),
            position_lifecycle=bool(data.get("position_lifecycle", True)),
            pricing_validation=bool(data.get("pricing_validation", True)),
            health_cards=bool(data.get("health_cards", True)),
            lifecycle_summary=bool(data.get("lifecycle_summary", True)),
        )


@dataclass(frozen=True)
class OutputFormatsConfig:
    trades: str
    base_data: str
    metrics: str

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "OutputFormatsConfig":
        return OutputFormatsConfig(
            trades=str(data.get("trades", "csv")),
            base_data=str(data.get("base_data", "csv")),
            metrics=str(data.get("metrics", "json")),
        )


@dataclass(frozen=True)
class OutputConfig:
    output_dir: Path
    files: OutputFilesConfig
    formats: OutputFormatsConfig
    compress: bool

    @staticmethod
    def from_dict(data: Dict[str, Any], base_dir: Path) -> "OutputConfig":
        if "output_dir" not in data:
            raise ValueError("output.output_dir must be provided")
        return OutputConfig(
            output_dir=_ensure_path(data["output_dir"], base_dir),
            files=OutputFilesConfig.from_dict(data.get("files", {})),
            formats=OutputFormatsConfig.from_dict(data.get("formats", {})),
            compress=bool(data.get("compress", False)),
        )


@dataclass(frozen=True)
class ControlBoardSLOConfig:
    """
    Service-level objectives for the options control board.
    Thresholds represent the point at which the board should raise warn/fail states.
    """

    assignment_warn_count: int
    assignment_fail_count: int
    liquidity_warn_count: int
    liquidity_fail_count: int
    pricing_fallback_warn_pct: float
    pricing_fallback_fail_pct: float
    warning_log_warn_count: int
    warning_log_fail_count: int
    risk_event_warn_count: int
    risk_event_fail_count: int

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ControlBoardSLOConfig":
        return ControlBoardSLOConfig(
            assignment_warn_count=int(data.get("assignment_warn_count", 3)),
            assignment_fail_count=int(data.get("assignment_fail_count", 8)),
            liquidity_warn_count=int(data.get("liquidity_warn_count", 5)),
            liquidity_fail_count=int(data.get("liquidity_fail_count", 15)),
            pricing_fallback_warn_pct=float(data.get("pricing_fallback_warn_pct", 0.20)),
            pricing_fallback_fail_pct=float(data.get("pricing_fallback_fail_pct", 0.35)),
            warning_log_warn_count=int(data.get("warning_log_warn_count", 10)),
            warning_log_fail_count=int(data.get("warning_log_fail_count", 30)),
            risk_event_warn_count=int(data.get("risk_event_warn_count", 5)),
            risk_event_fail_count=int(data.get("risk_event_fail_count", 15)),
        )


@dataclass(frozen=True)
class ControlBoardConfig:
    enabled: bool
    output_formats: Tuple[str, ...]
    archive_dir: Path
    slo: ControlBoardSLOConfig

    @staticmethod
    def from_dict(
        data: Dict[str, Any],
        base_dir: Path,
    ) -> "ControlBoardConfig":
        enabled = bool(data.get("enabled", True))

        raw_formats = data.get("output_formats", ["json", "markdown"])
        if isinstance(raw_formats, str):
            raw_formats = [raw_formats]
        normalised: List[str] = []
        for item in raw_formats:
            token = str(item).strip().lower()
            if not token:
                continue
            if token == "both":
                normalised = ["json", "markdown"]
                break
            if token not in {"json", "markdown"}:
                raise ValueError(f"Unsupported control_board output format: {token}")
            if token not in normalised:
                normalised.append(token)
        if not normalised:
            normalised = ["json"]

        archive_dir = _ensure_path(
            data.get("archive_dir", "outputs/control_board_archive"),
            base_dir,
        )

        slo_cfg = ControlBoardSLOConfig.from_dict(data.get("slo", {}))
        return ControlBoardConfig(
            enabled=enabled,
            output_formats=tuple(normalised),
            archive_dir=archive_dir,
            slo=slo_cfg,
        )


@dataclass(frozen=True)
class VisualizationConfig:
    enabled: bool
    plots: List[str]
    format: str
    dpi: int
    save_dir: str

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "VisualizationConfig":
        return VisualizationConfig(
            enabled=bool(data.get("enabled", True)),
            plots=[str(plot) for plot in data.get("plots", [])],
            format=str(data.get("format", "png")),
            dpi=int(data.get("dpi", 300)),
            save_dir=str(data.get("save_dir", "outputs/{run_id}/plots")),
        )


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    log_to_file: bool
    log_file: Path
    formatter: str
    max_file_size: str
    backup_count: int

    @staticmethod
    def from_dict(data: Dict[str, Any], base_dir: Path) -> "LoggingConfig":
        return LoggingConfig(
            level=str(data.get("level", "INFO")),
            log_to_file=bool(data.get("log_to_file", True)),
            log_file=_ensure_path(data.get("log_file", "logs/options_replay.log"), base_dir),
            formatter=str(data.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")),
            max_file_size=str(data.get("max_file_size", "15MB")),
            backup_count=int(data.get("backup_count", 7)),
        )


@dataclass(frozen=True)
class ParallelConfig:
    enabled: bool
    max_workers: int

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ParallelConfig":
        return ParallelConfig(
            enabled=bool(data.get("enabled", False)),
            max_workers=int(data.get("max_workers", 1)),
        )


@dataclass(frozen=True)
class PerformanceConfig:
    enable_cache: bool
    chunk_size: int
    parallel: ParallelConfig

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PerformanceConfig":
        return PerformanceConfig(
            enable_cache=bool(data.get("enable_cache", True)),
            chunk_size=int(data.get("chunk_size", 1000)),
            parallel=ParallelConfig.from_dict(data.get("parallel", {})),
        )


@dataclass(frozen=True)
class ValidationConfig:
    enabled: bool
    checks: List[str]
    on_failure: str

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ValidationConfig":
        checks = [str(check) for check in data.get("checks", [])]
        on_failure = str(data.get("on_failure", "raise_error")).lower()
        if on_failure not in {"raise_error", "log_warning", "continue"}:
            raise ValueError(f"Unsupported validation.on_failure: {on_failure}")
        return ValidationConfig(
            enabled=bool(data.get("enabled", True)),
            checks=checks,
            on_failure=on_failure,
        )


@dataclass(frozen=True)
class ReadinessConfig:
    min_ready_ratio: float
    max_blocked: int
    max_synthetic_ratio: float
    allow_needs_attention: bool
    synthetic_warn_ratio: float
    synthetic_info_ratio: float

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ReadinessConfig":
        max_syn = float(data.get("max_synthetic_ratio", 0.3))
        warn = data.get("synthetic_warn_ratio")
        info = data.get("synthetic_info_ratio")
        synthetic_warn = float(warn) if warn is not None else min(max_syn * 0.6, max_syn)
        synthetic_info = float(info) if info is not None else min(max_syn * 0.3, synthetic_warn)
        return ReadinessConfig(
            min_ready_ratio=float(data.get("min_ready_ratio", 0.8)),
            max_blocked=int(data.get("max_blocked", 0)),
            max_synthetic_ratio=max_syn,
            allow_needs_attention=bool(data.get("allow_needs_attention", True)),
            synthetic_warn_ratio=synthetic_warn,
            synthetic_info_ratio=synthetic_info,
        )


@dataclass(frozen=True)
class OptionsReplayConfig:
    """
    Root configuration object used by the replay engine.
    """

    inputs: InputsConfig
    pricing: PricingConfig
    strike_selection: StrikeSelectionConfig
    expiry_selection: ExpirySelectionConfig
    option_type: OptionTypeConfig
    lot_sizing: LotSizingConfig
    position_management: PositionManagementConfig
    liquidity: LiquidityConfig
    greeks: GreeksConfig
    risk: RiskConfig
    data_quality: DataQualityConfig
    output: OutputConfig
    visualization: VisualizationConfig
    logging: LoggingConfig
    performance: PerformanceConfig
    validation: ValidationConfig
    readiness: "ReadinessConfig"
    control_board: "ControlBoardConfig"
    repo_root: Path
    experimental: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)
    config_hash: str = field(init=False)

    def __post_init__(self):  # pragma: no cover - dataclass hook
        object.__setattr__(self, "config_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        """
        Compute a deterministic hash of the raw configuration dictionary.

        The hash is used for determinism checks and saved alongside run artefacts.
        """
        payload = json.dumps(self.raw, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def from_yaml(path: Path) -> "OptionsReplayConfig":
        """Load and validate configuration from a YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Options config file not found: {path}")
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            raise TypeError("Options config file must contain a mapping at the root level")
        resolved = path.resolve()
        # Climb to repository root (config -> options -> core -> src -> repo root)
        try:
            repo_root = resolved.parents[4]
        except IndexError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Unable to determine repository root from config path: {path}") from exc
        return OptionsReplayConfig.from_dict(data, base_dir=repo_root)

    @staticmethod
    def from_dict(data: Dict[str, Any], base_dir: Path) -> "OptionsReplayConfig":
        """
        Build the configuration object from a raw dictionary and repository base.

        Args:
            data: Raw configuration dictionary.
            base_dir: Repository base directory used to resolve relative paths.
        """
        # Base dir is the repository root; ensure type
        base_dir = base_dir.resolve()
        inputs = InputsConfig.from_dict(data.get("inputs", {}), base_dir)
        pricing = PricingConfig.from_dict(data.get("pricing", {}), base_dir)
        strike = StrikeSelectionConfig.from_dict(data.get("strike_selection", {}))
        expiry = ExpirySelectionConfig.from_dict(data.get("expiry_selection", {}))
        option_type = OptionTypeConfig.from_dict(data.get("option_type", {}))
        lot_sizing = LotSizingConfig.from_dict(data.get("lot_sizing", {}))
        position_mgmt = PositionManagementConfig.from_dict(data.get("position_management", {}))
        liquidity = LiquidityConfig.from_dict(data.get("liquidity", {}))
        greeks = GreeksConfig.from_dict(data.get("greeks", {}))
        risk = RiskConfig.from_dict(data.get("risk", {}))
        data_quality = DataQualityConfig.from_dict(data.get("data_quality", {}))
        output = OutputConfig.from_dict(data.get("output", {}), base_dir)
        visualization = VisualizationConfig.from_dict(data.get("visualization", {}))
        logging_cfg = LoggingConfig.from_dict(data.get("logging", {}), base_dir)
        performance = PerformanceConfig.from_dict(data.get("performance", {}))
        validation = ValidationConfig.from_dict(data.get("validation", {}))
        readiness = ReadinessConfig.from_dict(data.get("readiness", {}))
        control_board = ControlBoardConfig.from_dict(data.get("control_board", {}), base_dir)
        experimental = dict(data.get("experimental", {}))

        return OptionsReplayConfig(
            inputs=inputs,
            pricing=pricing,
            strike_selection=strike,
            expiry_selection=expiry,
            option_type=option_type,
            lot_sizing=lot_sizing,
            position_management=position_mgmt,
            liquidity=liquidity,
            greeks=greeks,
            risk=risk,
            data_quality=data_quality,
            output=output,
            visualization=visualization,
            logging=logging_cfg,
            performance=performance,
            validation=validation,
            readiness=readiness,
            control_board=control_board,
            repo_root=base_dir,
            experimental=experimental,
            raw=data,
        )
