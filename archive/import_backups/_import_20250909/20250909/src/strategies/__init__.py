# Backtester/strategies/__init__.py
# Export key classes for easier imports
from .strategy_base import StrategyBase
from .strategy_factory import StrategyFactory
from .strategy_sma_crossover import SMAcrossoverStrategy
from .strategy_bollinger_bands import BollingerBandsStrategy

__all__ = ['StrategyBase', 'StrategyFactory', 'SMAcrossoverStrategy', 'BollingerBandsStrategy']