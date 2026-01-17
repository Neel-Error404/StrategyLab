from __future__ import annotations

import re
from typing import Callable, Dict

import pandas as pd

from .indicator_catalog import INDICATOR_FUNCTIONS as CATALOG_FUNCTIONS


class IndicatorLibrary:
    """Catalog of advanced indicator functions plus warm-up inference."""

    def __init__(self) -> None:
        self._functions: Dict[str, Callable[[pd.DataFrame], pd.Series]] = CATALOG_FUNCTIONS

    def is_supported(self, name: str) -> bool:
        return name in self._functions

    def compute(self, name: str, data: pd.DataFrame):
        if name not in self._functions:
            raise KeyError(f"Indicator '{name}' is not registered")
        # Work on a copy to avoid mutating shared dataframes
        return self._functions[name](data.copy())

    def warmup_period(self, name: str) -> int:
        """Best-effort warm-up estimate based on indicator metadata."""
        lower_name = name.lower()
        if "ichimoku" in lower_name:
            return 52
        if "macd" in lower_name:
            digits = _extract_digits(lower_name)
            return max(digits) if digits else 26
        if "bollinger" in lower_name or lower_name in {"bbw", "blc", "bmc", "bhc"}:
            return 20
        if lower_name.startswith("atr") or lower_name == "adr":
            return 14
        digits = _extract_digits(lower_name)
        if digits:
            return max(digits)
        # Fallback for indicators without explicit window sizes
        return 30

    @property
    def names(self):
        return list(self._functions.keys())


def _extract_digits(name: str) -> list[int]:
    return [int(match) for match in re.findall(r"(\d+)", name)]
