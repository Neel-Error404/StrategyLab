"""Compatibility shim: expose MSEUniversalStrategy for tests and tools

Some tests and legacy code import `strategies.example_mse_strategy.MSEUniversalStrategy`.
The new strategy implementation lives in `src/strategies/strategy_mse.py` as `MSEStrategy`.
This module provides a lightweight alias so imports keep working without changing tests.
"""
from .strategy_mse import MSEStrategy


class MSEUniversalStrategy(MSEStrategy):
    """Backward-compatible alias of MSEStrategy.

    This subclass exists only to preserve the historical class name expected by
    tests and external scripts. It behaves identically to MSEStrategy.
    """
    pass


__all__ = ["MSEUniversalStrategy"]
