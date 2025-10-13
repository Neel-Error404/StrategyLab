"""
Configuration loader for options validation.

Loads and validates validation_config.yaml.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class ValidationConfig:
    """Wrapper for validation configuration with easy access to nested values."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Load validation configuration.

        Args:
            config_path: Path to validation_config.yaml (default: auto-detect)
        """
        self.logger = logging.getLogger(__name__)

        # Auto-detect config path if not provided
        if config_path is None:
            config_path = Path(__file__).parent / "validation_config.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Validation config not found: {config_path}")

        # Load YAML
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        self.logger.info(f"Loaded validation config from {config_path}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get config value using dot-notation path.

        Args:
            key_path: Dot-separated path (e.g., 'api.rate_limit.requests_per_second')
            default: Default value if key not found

        Returns:
            Config value or default

        Example:
            >>> config = ValidationConfig()
            >>> config.get('api.rate_limit.requests_per_second')
            5
        """
        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    # Convenience properties for commonly used config values

    @property
    def tickers(self):
        """Get list of validation tickers."""
        return self.get('validation.tickers', ['RELIANCE'])

    @property
    def timeframe(self):
        """Get timeframe for data fetching."""
        return self.get('validation.timeframe', '1day')

    @property
    def strike_range_pct(self):
        """Get strike range percentage."""
        return self.get('validation.strike_range.percentage_range', 0.20)

    @property
    def min_open_interest(self):
        """Get minimum open interest filter."""
        return self.get('validation.filters.min_open_interest', 100)

    @property
    def max_strikes(self):
        """Get maximum strikes per expiry."""
        return self.get('validation.filters.max_strikes', None)

    @property
    def manual_reference_prices(self):
        """Get manual override reference prices."""
        return self.get('validation.manual_reference_prices', {}) or {}

    @property
    def max_spread_pct(self):
        """Get maximum spread percentage filter."""
        return self.get('validation.filters.max_spread_pct', None)

    @property
    def min_volume(self):
        """Get minimum volume filter."""
        return self.get('validation.filters.min_volume', 10)

    @property
    def exclude_dte_below(self):
        """Get minimum days-to-expiry threshold."""
        return self.get('validation.filters.exclude_dte_below', None)

    @property
    def exclude_dte_above(self):
        """Get maximum days-to-expiry threshold."""
        return self.get('validation.filters.exclude_dte_above', None)

    @property
    def requests_per_second(self):
        """Get API rate limit (requests per second)."""
        return self.get('api.rate_limit.requests_per_second', 5)

    @property
    def requests_per_minute(self):
        """Get API rate limit (requests per minute)."""
        return self.get('api.rate_limit.requests_per_minute', 100)

    @property
    def retry_attempts(self):
        """Get API retry attempts."""
        return self.get('api.rate_limit.retry_attempts', 3)

    @property
    def retry_backoff_factor(self):
        """Get API retry backoff factor."""
        return self.get('api.rate_limit.retry_backoff_factor', 2)

    @property
    def output_dir(self):
        """Get output directory for validation results."""
        return self.get('output.output_dir', 'src/core/options/data/validation_results')

    @property
    def log_level(self):
        """Get logging level."""
        return self.get('logging.level', 'INFO')

    @property
    def log_file(self):
        """Get log file path."""
        return self.get('logging.log_file', 'logs/options_validation.log')

    @property
    def enable_parallel(self):
        """Check if parallel processing is enabled."""
        return self.get('parallel.enable', True)

    @property
    def log_to_file(self) -> bool:
        """Return whether file logging is enabled."""
        return self.get('logging.log_to_file', False)

    @property
    def max_workers(self):
        """Get maximum number of parallel workers."""
        return self.get('parallel.max_workers', 3)

    @property
    def date_range(self):
        """Get date range for organizing fetched data."""
        explicit = self.get('validation.date_range', None)
        if explicit:
            return explicit

        # Attempt to auto-generate from configured start/end dates
        start_str = self.get('validation.time_period.start_date', None)
        end_str = self.get('validation.time_period.end_date', None)
        if not start_str and not end_str:
            return None

        from datetime import datetime, date
        start = datetime.strptime(start_str, '%Y-%m-%d').date() if start_str else None
        if end_str:
            end = datetime.strptime(end_str, '%Y-%m-%d').date()
        else:
            end = date.today()

        if start is None:
            # Cannot build range without start date
            return None

        return f"{start.isoformat()}_to_{end.isoformat()}"

    @property
    def equity_pool(self):
        """Get equity pool path (for reference prices). None = auto-detect."""
        return self.get('validation.equity_pool', None)

    @property
    def reference_date_start(self):
        """Get reference date lower bound (for reference price lookups)."""
        start_str = self.get('validation.time_period.start_date', None)
        if start_str:
            from datetime import datetime
            return datetime.strptime(start_str, '%Y-%m-%d').date()
        return None

    @property
    def reference_date_end(self):
        """Get reference date upper bound."""
        end_str = self.get('validation.time_period.end_date', None)
        if end_str:
            from datetime import datetime
            return datetime.strptime(end_str, '%Y-%m-%d').date()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Get full config as dictionary."""
        return self._config.copy()


# Module-level singleton instance
_config_instance = None


def get_validation_config(config_path: Optional[str] = None) -> ValidationConfig:
    """
    Get or create validation config instance (singleton pattern).

    Args:
        config_path: Path to config file (only used on first call)

    Returns:
        ValidationConfig instance
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = ValidationConfig(config_path)

    return _config_instance


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = ValidationConfig()

    print("=== Validation Config Test ===")
    print(f"Tickers: {config.tickers}")
    print(f"Timeframe: {config.timeframe}")
    print(f"Strike Range: ±{config.strike_range_pct * 100}%")
    print(f"Min OI: {config.min_open_interest}")
    print(f"API Rate Limit: {config.requests_per_second} req/sec")
    print(f"Retry Attempts: {config.retry_attempts}")
    print(f"Log Level: {config.log_level}")
    print(f"Parallel Enabled: {config.enable_parallel}")
    print(f"Max Workers: {config.max_workers}")
