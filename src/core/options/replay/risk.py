"""
Risk management primitives for the options replay engine.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.core.options.replay.config import RiskConfig
from .models import RiskEvent


@dataclass
class OpenPosition:
    trade_id: str
    ticker: str
    entry_time: pd.Timestamp
    entry_cost: float
    quantity: int


class RiskManager:
    """
    Stateful risk manager enforcing portfolio and kill-switch rules.
    """

    _SEVERITY_MAP = {
        "entry_registered": "info",
        "entry_rejected": "warning",
        "exit_missing_position": "warning",
        "position_closed": "info",
        "mtm_delta_warning": "warning",
        "theta_decay_warning": "warning",
        "mtm_drawdown_warning": "warning",
        "assignment_risk": "warning",
        "risk_alert": "warning",
        "kill_switch": "error",
        "kill_switch_status": "info",
    }

    def __init__(self, config: RiskConfig):
        self.config = config
        self.initial_capital = float(config.initial_portfolio_value)
        self.open_positions: Dict[str, OpenPosition] = {}
        self.ticker_allocations: Dict[str, float] = defaultdict(float)
        self.ticker_realized_pnl: Dict[str, float] = defaultdict(float)
        self.ticker_mtm_metrics: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.ticker_assignment_flags: Dict[str, int] = defaultdict(int)
        self.realized_pnl: float = 0.0
        self.max_equity: float = self.initial_capital
        self.min_equity: float = self.initial_capital
        self.kill_switch_triggered: bool = False
        self.kill_switch_reason: str | None = None
        self.kill_switch_timestamp: Optional[pd.Timestamp] = None
        self.risk_events: List[RiskEvent] = []
        self.kill_switch_status_logged: bool = False

    @property
    def deployed_capital(self) -> float:
        return sum(pos.entry_cost for pos in self.open_positions.values())

    @property
    def current_equity(self) -> float:
        return self.initial_capital + self.realized_pnl

    def _record_event(
        self,
        event_type: str,
        message: str,
        details: Dict[str, object],
        *,
        severity: Optional[str] = None,
    ) -> None:
        event_severity = severity or self._SEVERITY_MAP.get(event_type, "info")
        self.risk_events.append(
            RiskEvent(
                timestamp=pd.Timestamp.now(tz="UTC"),
                event_type=event_type,
                message=message,
                details=details,
                severity=event_severity,
            )
        )

    def evaluate_entry(
        self,
        trade_id: str,
        entry_cost: float,
        timestamp: pd.Timestamp,
        ticker: Optional[str] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Evaluate whether a new position can be opened.
        """
        reasons: List[str] = []
        if self.kill_switch_triggered:
            reasons.append("kill_switch_engaged")
            return False, reasons
        if trade_id in self.open_positions:
            reasons.append("duplicate_trade_id")
            return False, reasons
        if len(self.open_positions) >= self.config.max_concurrent_positions:
            reasons.append("max_concurrent_positions")
        max_position_value = self.initial_capital * self.config.max_position_size_per_trade
        if entry_cost > max_position_value:
            reasons.append("max_position_size_per_trade")
        if self.deployed_capital + entry_cost > self.initial_capital * self.config.max_portfolio_allocation:
            reasons.append("max_portfolio_allocation")
        if reasons:
            details = {
                "trade_id": trade_id,
                "reasons": reasons,
                "timestamp": timestamp.isoformat(),
                "entry_cost": entry_cost,
                "portfolio_allocation": self.deployed_capital,
            }
            if ticker:
                details["ticker"] = ticker
                details["ticker_allocation"] = self.ticker_allocations.get(ticker, 0.0)
            self._record_event(
                "entry_rejected",
                "Entry rejected by risk manager",
                details,
            )
            return False, reasons
        return True, []

    def register_entry(
        self,
        trade_id: str,
        entry_cost: float,
        quantity: int,
        timestamp: pd.Timestamp,
        ticker: str,
    ) -> None:
        """
        Add a new open position to the risk book.
        """
        self.open_positions[trade_id] = OpenPosition(
            trade_id=trade_id,
            ticker=ticker,
            entry_time=timestamp,
            entry_cost=entry_cost,
            quantity=quantity,
        )
        self.ticker_allocations[ticker] += entry_cost
        self.ticker_mtm_metrics[ticker]["position_value"] += entry_cost
        self._record_event(
            "entry_registered",
            "Position opened",
            {
                "trade_id": trade_id,
                "ticker": ticker,
                "entry_cost": entry_cost,
                "quantity": quantity,
                "timestamp": timestamp.isoformat(),
                "ticker_allocation": self.ticker_allocations[ticker],
                "portfolio_allocation": self.deployed_capital,
            },
        )

    def register_exit(
        self,
        trade_id: str,
        realized_pnl: float,
        timestamp: pd.Timestamp,
        ticker: Optional[str] = None,
    ) -> List[RiskEvent]:
        """
        Close an existing position and update drawdown metrics.
        """
        events: List[RiskEvent] = []
        position = self.open_positions.pop(trade_id, None)
        if position is None:
            self._record_event(
                "exit_missing_position",
                "Exit received for unknown position",
                {"trade_id": trade_id, "timestamp": timestamp.isoformat(), "ticker": ticker},
            )
            return events

        position_ticker = ticker or position.ticker
        if position_ticker:
            self.ticker_allocations[position_ticker] = max(
                self.ticker_allocations.get(position_ticker, 0.0) - position.entry_cost,
                0.0,
            )
            self.ticker_realized_pnl[position_ticker] += realized_pnl
            metrics = self.ticker_mtm_metrics[position_ticker]
            metrics["position_value"] = max(metrics.get("position_value", 0.0) - position.entry_cost, 0.0)

        self.realized_pnl += realized_pnl
        equity = self.current_equity
        self.max_equity = max(self.max_equity, equity)
        self.min_equity = min(self.min_equity, equity)

        drawdown = 0.0
        if self.max_equity > 0:
            drawdown = (equity - self.max_equity) / self.max_equity

        if (
            self.config.stop_trading_on_drawdown
            and drawdown < -abs(self.config.max_drawdown_pct)
            and not self.kill_switch_triggered
        ):
            self.kill_switch_triggered = True
            self.kill_switch_reason = "max_drawdown_breach"
            self.kill_switch_timestamp = timestamp
            events.append(
                RiskEvent(
                    timestamp=timestamp,
                    event_type="kill_switch",
                    message="Kill switch engaged: max drawdown breached",
                    details={"drawdown": drawdown, "trade_id": trade_id, "ticker": position_ticker},
                    severity="error",
                )
            )

        if self.config.kill_switch.enabled and not self.kill_switch_triggered:
            # Intraday loss check based on realized equity
            threshold_equity = self.initial_capital * (1 - self.config.kill_switch.max_intraday_loss_pct)
            if equity <= threshold_equity:
                self.kill_switch_triggered = True
                self.kill_switch_reason = "intraday_loss_threshold"
                self.kill_switch_timestamp = timestamp
                event = RiskEvent(
                    timestamp=timestamp,
                    event_type="kill_switch",
                    message="Kill switch engaged: intraday loss threshold breached",
                    details={"equity": equity, "threshold": threshold_equity, "trade_id": trade_id, "ticker": position_ticker},
                    severity="error",
                )
                events.append(event)

        if self.config.kill_switch.enabled and position.entry_cost > 0:
            loss_pct = realized_pnl / position.entry_cost
            if loss_pct < -abs(self.config.kill_switch.max_single_trade_loss_pct):
                event = RiskEvent(
                    timestamp=timestamp,
                    event_type="risk_alert",
                    message="Single trade loss exceeded threshold",
                    details={"trade_id": trade_id, "loss_pct": loss_pct, "ticker": position_ticker},
                    severity="warning",
                )
                events.append(event)

        exit_event_details = {
            "trade_id": trade_id,
            "ticker": position_ticker,
            "realized_pnl": realized_pnl,
            "timestamp": timestamp.isoformat(),
            "ticker_allocation": self.ticker_allocations.get(position_ticker, 0.0) if position_ticker else None,
            "portfolio_realized_pnl": self.realized_pnl,
        }
        events.append(
            RiskEvent(
                timestamp=timestamp,
                event_type="position_closed",
                message="Position closed",
                details=exit_event_details,
                severity="info",
            )
        )

        for event in events:
            self._record_event(event.event_type, event.message, event.details, severity=event.severity)

        return events

    def record_lifecycle_metrics(
        self,
        ticker: str,
        delta_peak: float,
        delta_drift: float,
        theta_cumulative: float,
        gamma_peak: float,
        unrealized_drawdown: float,
        assignment_risk: bool,
        assignment_intrinsic: float,
        assignment_dte_hours: float,
        timestamp: pd.Timestamp,
        trade_id: str,
    ) -> None:
        metrics = self.ticker_mtm_metrics[ticker]
        metrics["delta_peak_abs"] = max(metrics.get("delta_peak_abs", 0.0), abs(delta_peak))
        metrics["delta_drift"] = max(metrics.get("delta_drift", 0.0), abs(delta_drift))
        metrics["theta_cumulative"] += theta_cumulative
        metrics["gamma_peak_abs"] = max(metrics.get("gamma_peak_abs", 0.0), abs(gamma_peak))
        metrics["max_unrealized_drawdown"] = min(metrics.get("max_unrealized_drawdown", 0.0), unrealized_drawdown)

        delta_limit = self.initial_capital * max(self.config.max_position_size_per_trade, 0.05)
        if metrics["delta_peak_abs"] > delta_limit:
            self._record_event(
                "mtm_delta_warning",
                "Delta exposure exceeded soft limit",
                {
                    "trade_id": trade_id,
                    "ticker": ticker,
                    "delta_peak_abs": metrics["delta_peak_abs"],
                    "threshold": delta_limit,
                    "timestamp": timestamp.isoformat(),
                },
            )

        theta_limit = -self.initial_capital * max(self.config.max_portfolio_allocation, 0.05)
        if metrics["theta_cumulative"] < theta_limit:
            self._record_event(
                "theta_decay_warning",
                "Cumulative theta below soft limit",
                {
                    "trade_id": trade_id,
                    "ticker": ticker,
                    "theta_cumulative": metrics["theta_cumulative"],
                    "threshold": theta_limit,
                    "timestamp": timestamp.isoformat(),
                },
            )

        drawdown_floor = -abs(self.config.max_drawdown_pct or 0.25)
        if metrics["max_unrealized_drawdown"] < drawdown_floor and not self.kill_switch_triggered:
            self._record_event(
                "mtm_drawdown_warning",
                "Unrealized drawdown breached soft limit",
                {
                    "trade_id": trade_id,
                    "ticker": ticker,
                    "drawdown": metrics["max_unrealized_drawdown"],
                    "threshold": drawdown_floor,
                    "timestamp": timestamp.isoformat(),
                },
            )

        assignment_cfg = self.config.assignment_risk
        if assignment_cfg.enabled and assignment_risk:
            self.ticker_assignment_flags[ticker] += 1
            self._record_event(
                "assignment_risk",
                "Assignment risk detected near expiry",
                {
                    "trade_id": trade_id,
                    "ticker": ticker,
                    "intrinsic": assignment_intrinsic,
                    "dte_hours": assignment_dte_hours,
                    "threshold_hours": assignment_cfg.dte_hours_threshold,
                    "timestamp": timestamp.isoformat(),
                },
            )

    def summary(self) -> Dict[str, object]:
        """
        Return a serialisable snapshot of portfolio and per-ticker risk state.
        """
        portfolio_drawdown = 0.0
        if self.max_equity > 0:
            portfolio_drawdown = (self.min_equity - self.max_equity) / self.max_equity
        severity_counts: Dict[str, int] = defaultdict(int)
        for event in self.risk_events:
            severity_counts[event.severity] += 1
        portfolio = {
            "initial_capital": self.initial_capital,
            "deployed_capital": self.deployed_capital,
            "deployed_capital_pct": (self.deployed_capital / self.initial_capital) if self.initial_capital else 0.0,
            "realized_pnl": self.realized_pnl,
            "current_equity": self.current_equity,
            "max_equity": self.max_equity,
            "min_equity": self.min_equity,
            "max_drawdown_pct": portfolio_drawdown * 100.0,
            "kill_switch_triggered": self.kill_switch_triggered,
            "kill_switch_reason": self.kill_switch_reason,
            "kill_switch_timestamp": self.kill_switch_timestamp.isoformat() if self.kill_switch_timestamp else None,
            "risk_events": {
                "total": len(self.risk_events),
                "by_severity": dict(severity_counts),
            },
        }

        ticker_keys = set(self.ticker_allocations.keys()) | set(self.ticker_realized_pnl.keys())
        ticker_keys |= {position.ticker for position in self.open_positions.values()}
        per_ticker: Dict[str, Dict[str, float | int]] = {}
        for ticker in sorted(ticker_keys):
            per_ticker[ticker] = {
                "open_capital": float(self.ticker_allocations.get(ticker, 0.0)),
                "open_positions": sum(1 for position in self.open_positions.values() if position.ticker == ticker),
                "realized_pnl": float(self.ticker_realized_pnl.get(ticker, 0.0)),
                "mtm_metrics": dict(self.ticker_mtm_metrics.get(ticker, {})),
                "assignment_risk_count": int(self.ticker_assignment_flags.get(ticker, 0)),
            }

        return {
            "portfolio": portfolio,
            "per_ticker": per_ticker,
        }

    def log_kill_switch_status(self) -> None:
        if self.kill_switch_status_logged:
            return
        details = {
            "kill_switch_triggered": self.kill_switch_triggered,
            "reason": self.kill_switch_reason,
        }
        severity = "error" if self.kill_switch_triggered else "info"
        self._record_event(
            "kill_switch_status",
            "Kill switch drill status recorded",
            details,
            severity=severity,
        )
        self.kill_switch_status_logged = True
