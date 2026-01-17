import logging
from typing import Dict, List, Tuple, Union

import pandas as pd

from src.indicators import IndicatorLibrary


class IndicatorRegistry:
    """Compute and attach indicators declared in configuration."""

    _PANDAS_FREQ = {
        '1m': '1T',
        '2m': '2T',
        '3m': '3T',
        '5m': '5T',
        '10m': '10T',
        '15m': '15T',
        '30m': '30T',
        '1h': '1H',
        '2h': '2H',
        '4h': '4H',
        'day': '1D',
    }

    def __init__(self, strategy, indicator_map: Dict[str, List]):
        self.strategy = strategy
        self.indicator_map = indicator_map or {"entry": [], "exit": []}
        self.logger = logging.getLogger(f"indicator_registry.{strategy.name}")
        self.library = IndicatorLibrary()
        self.required_warmup = self._estimate_warmup()

    def apply(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]):
        if not any(self.indicator_map.values()):
            return data

        data_map = data if isinstance(data, dict) else {"entry": data}

        for role, specs in self.indicator_map.items():
            for spec in specs:
                self._apply_single_indicator(role, spec, data_map)

        return data_map if isinstance(data, dict) else data_map.get("entry", data)

    def _normalize_spec(self, spec):
        if isinstance(spec, dict):
            return spec
        if hasattr(spec, "__dict__"):
            return {
                "name": getattr(spec, "name", None),
                "type": getattr(spec, "type", None),
                "timeframe": getattr(spec, "timeframe", None),
                "params": getattr(spec, "params", {}) or {},
            }
        return {}

    def _apply_single_indicator(self, role: str, spec: Dict, data_map: Dict[str, pd.DataFrame]):
        spec_dict = self._normalize_spec(spec)
        name = spec_dict.get("name")
        indicator_type = spec_dict.get("type")
        timeframe = spec_dict.get("timeframe")
        params = spec_dict.get("params", {})

        if not name or not indicator_type:
            self.logger.warning("Indicator spec missing name/type, skipping.")
            return

        target_frame = timeframe or self._default_timeframe(role, data_map)
        df = data_map.get(target_frame)
        if df is None:
            self.logger.warning(
                f"No dataframe available for timeframe '{target_frame}' when computing indicator '{name}'."
            )
            return

        indicator_key = indicator_type.lower()
        try:
            series, warmup = self._calculate_builtin_indicator(indicator_key, df, params)
            if series is None:
                if not self.library.is_supported(indicator_key):
                    raise ValueError(f"Unsupported indicator type '{indicator_type}'")
                series = self.library.compute(indicator_key, df)
                warmup = self.library.warmup_period(indicator_key)
            self._attach_series(df, name, series)
            if warmup:
                self.required_warmup = max(self.required_warmup, warmup)
        except Exception as exc:
            self.logger.error(f"Failed to compute indicator '{name}': {exc}")

    def _attach_series(self, df: pd.DataFrame, name: str, series):
        if isinstance(series, dict):
            for key, value in series.items():
                column = f"{name}_{key}"
                df[column] = value
        else:
            df[name] = series

    def _calculate_builtin_indicator(
        self, indicator_type: str, df: pd.DataFrame, params: Dict
    ) -> Tuple[Union[pd.Series, Dict[str, pd.Series], None], int]:
        close = df['close']

        if indicator_type == 'sma':
            period = params.get('period', 20)
            return self.strategy.calculate_sma(close, period), period

        if indicator_type == 'ema':
            period = params.get('period', 20)
            return self.strategy.calculate_ema(close, period), period

        if indicator_type == 'rsi':
            period = params.get('period', 14)
            return self.strategy.calculate_rsi(close, period), period

        if indicator_type == 'macd':
            fast = params.get('fast', 12)
            slow = params.get('slow', 26)
            signal = params.get('signal', 9)
            macd_values = self.strategy.calculate_macd(close, fast=fast, slow=slow, signal=signal)
            component = params.get('component')
            warmup = max(slow, signal)
            if component and component in macd_values:
                return macd_values[component], warmup
            return macd_values, warmup

        if indicator_type == 'bollinger':
            period = params.get('period', 20)
            std_dev = params.get('std_dev', 2.0)
            return self.strategy.calculate_bollinger_bands(close, period=period, std_dev=std_dev), period

        if indicator_type == 'atr':
            period = params.get('period', 14)
            return (
                self.strategy.calculate_atr(df['high'], df['low'], close, period=period),
                period,
            )

        if indicator_type == 'stochastic':
            period = params.get('period', 14)
            return (
                self.strategy.calculate_stochastic(df['high'], df['low'], close, period=period),
                period,
            )

        return None, 0

    def _default_timeframe(self, role: str, data_map: Dict[str, pd.DataFrame]) -> str:
        """
        Determine which timeframe to use when the spec omits one.
        Preference order:
          1. Strategy timeframe config for the given role (entry/exit/confirmation)
          2. A dataframe keyed by the role name
          3. The first available dataframe in the map
        """
        config = getattr(self.strategy, "timeframe_config", None)
        candidates: List[str] = []
        if config:
            if role == 'entry' and getattr(config, 'entry', None):
                candidates.extend(config.entry)
            elif role == 'exit' and getattr(config, 'exit', None):
                candidates.extend(config.exit)
            elif role == 'confirmation' and getattr(config, 'confirmation', None):
                candidates.extend(config.confirmation)
        if role in data_map:
            candidates.append(role)
        if not candidates and data_map:
            candidates.append(next(iter(data_map.keys())))
        return candidates[0] if candidates else 'entry'

    def _estimate_warmup(self) -> int:
        max_warmup = 0
        for specs in self.indicator_map.values():
            for raw_spec in specs:
                spec = self._normalize_spec(raw_spec)
                indicator_type = (spec.get("type") or "").lower()
                params = spec.get("params", {})
                warmup = self._warmup_for_indicator(indicator_type, params)
                max_warmup = max(max_warmup, warmup)
        return max_warmup

    def _warmup_for_indicator(self, indicator_type: str, params: Dict) -> int:
        builtin_warmup = self._builtin_warmup(indicator_type, params)
        if builtin_warmup:
            return builtin_warmup
        if self.library.is_supported(indicator_type):
            return self.library.warmup_period(indicator_type)
        return 0

    @staticmethod
    def _builtin_warmup(indicator_type: str, params: Dict) -> int:
        if indicator_type in {'sma', 'ema', 'bollinger'}:
            return params.get('period', 20)
        if indicator_type in {'rsi', 'atr', 'stochastic'}:
            return params.get('period', 14)
        if indicator_type == 'macd':
            slow = params.get('slow', 26)
            signal = params.get('signal', 9)
            return max(slow, signal)
        return 0
