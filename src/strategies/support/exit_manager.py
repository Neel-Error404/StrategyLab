import logging
from datetime import datetime, time
from typing import Dict, Optional, Tuple

import pandas as pd


class ExitManager:
    """Evaluate stop/target/timeout exits defined in configuration."""

    def __init__(self, exit_config):
        self.config = exit_config
        self.logger = logging.getLogger("ExitManager")
        self._intraday_cutoff = self._parse_time(
            getattr(exit_config.square_off, 'intraday_cutoff', None)
            or getattr(exit_config.timeout, 'intraday_cutoff', None)
        )

    def is_active(self) -> bool:
        return bool(self.config and self.config.mode.lower() != 'manual')

    def start_trade(self, trade_type: str, row: pd.Series) -> Dict:
        context = {
            'entry_price': row['close'],
            'entry_time': row['timestamp'],
            'trade_type': trade_type,
            'stop_price': None,
            'target_price': None,
        }

        context['stop_price'] = self._compute_threshold(
            trade_type,
            row['close'],
            getattr(self.config, 'stop_loss', None)
        )
        context['target_price'] = self._compute_threshold(
            trade_type,
            row['close'],
            getattr(self.config, 'take_profit', None),
            is_target=True
        )

        return context

    def should_exit(self, trade: Dict, row: pd.Series) -> Tuple[bool, Optional[str]]:
        if not self.is_active():
            return False, None

        context = trade.get('_exit_context', {})
        price = row.get('close')
        trade_type = context.get('trade_type', trade.get('Trade Type'))

        if self._check_stop(context, trade_type, price):
            return True, 'stop_loss'
        if self._check_target(context, trade_type, price):
            return True, 'take_profit'
        if self._check_timeout(context, row.get('timestamp')):
            return True, 'timeout'
        if self._check_intraday_cutoff(row.get('timestamp')):
            return True, 'intraday_square_off'
        if self._check_delivery_horizon(context, row.get('timestamp')):
            return True, 'delivery_square_off'

        return False, None

    def _compute_threshold(self, trade_type: str, entry_price: float, cfg, is_target: bool = False):
        if not cfg or not getattr(cfg, 'enabled', False):
            return None

        threshold_type = getattr(cfg, 'type', 'percent')
        value = getattr(cfg, 'value', 0.0)

        if threshold_type == 'percent':
            if trade_type == 'Buy':
                return entry_price * (1 + value) if is_target else entry_price * (1 - value)
            return entry_price * (1 - value) if is_target else entry_price * (1 + value)

        return None

    def _check_stop(self, context: Dict, trade_type: str, price: float) -> bool:
        stop_price = context.get('stop_price')
        if stop_price is None or price is None:
            return False
        if trade_type == 'Buy':
            return price <= stop_price
        return price >= stop_price

    def _check_target(self, context: Dict, trade_type: str, price: float) -> bool:
        target_price = context.get('target_price')
        if target_price is None or price is None:
            return False
        if trade_type == 'Buy':
            return price >= target_price
        return price <= target_price

    def _check_timeout(self, context: Dict, current_time: Optional[pd.Timestamp]) -> bool:
        timeout_cfg = getattr(self.config, 'timeout', None)
        if not timeout_cfg or not timeout_cfg.enabled or not current_time:
            return False
        entry_time = context.get('entry_time')
        if entry_time is None:
            return False
        elapsed = (current_time - entry_time).total_seconds() / 60.0
        return timeout_cfg.max_minutes > 0 and elapsed >= timeout_cfg.max_minutes

    def _check_intraday_cutoff(self, current_time: Optional[pd.Timestamp]) -> bool:
        if not current_time or not self._intraday_cutoff:
            return False
        if getattr(self.config.square_off, 'mode', 'none') != 'intraday':
            return False
        return current_time.time() >= self._intraday_cutoff

    def _check_delivery_horizon(self, context: Dict, current_time: Optional[pd.Timestamp]) -> bool:
        delivery_days = getattr(self.config.square_off, 'delivery_horizon_days', 0)
        if getattr(self.config.square_off, 'mode', 'none') != 'delivery':
            return False
        if delivery_days <= 0 or not current_time:
            return False
        entry_time = context.get('entry_time')
        if entry_time is None:
            return False
        exit_time = entry_time + pd.Timedelta(days=delivery_days)
        return current_time >= exit_time

    @staticmethod
    def _parse_time(value: Optional[str]) -> Optional[time]:
        if not value:
            return None
        try:
            hour, minute = value.split(":")
            return time(int(hour), int(minute))
        except Exception:
            return None
