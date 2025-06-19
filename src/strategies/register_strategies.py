# Backtester/strategies/register_strategies.py
from .strategy_factory import StrategyFactory
from .strategy_sma_crossover import SMAcrossoverStrategy
from .strategy_bollinger_bands import BollingerBandsStrategy

def register_all_strategies():
    """
    Register all available strategies with the factory.
    """
    try:
        # Register template strategies for public use
        StrategyFactory.register_strategy('sma_crossover', SMAcrossoverStrategy)
        StrategyFactory.register_strategy('bollinger_bands', BollingerBandsStrategy)
        
        # Note: Add your custom strategies here
        # StrategyFactory.register_strategy('your_strategy', YourStrategyClass)
        
        return True
    except Exception as e:
        print(f"Error registering strategies: {e}")
        return False