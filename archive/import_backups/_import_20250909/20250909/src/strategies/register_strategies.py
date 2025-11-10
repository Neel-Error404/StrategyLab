# Backtester/strategies/register_strategies.py
from .strategy_factory import StrategyFactory
from .strategy_sma_crossover import SMAcrossoverStrategy
from .strategy_mse import MSEStrategy
from .strategy_bollinger_bands import BollingerBandsStrategy
from .mse_20pct_with_cascade import MSEStrategy as MSE20WithCascade
from .mse_20pct_no_cascade import MSEStrategy as MSE20NoCascade
from .mse_80pct_with_cascade import MSEStrategy as MSE80WithCascade
from .mse_80pct_no_cascade import MSEStrategy as MSE80NoCascade
from .mse_20pct_no_cascade_with_macd_exit import MSEStrategy as MSE20NoCascadeWithMACDExit
from .mse_80pct_no_cascade_with_macd_exit import MSEStrategy as MSE80NoCascadeWithMACDExit
from .mse_80pct_no_cascade_live_matching import MSEStrategy as MSE80NoCascadeLiveMatching
from .mse_5min_validation import MSEStrategy as MSE5MinValidation

def register_all_strategies():
    """
    Register all available strategies with the factory.
    """
    try:
        # Register template strategies for public use
        StrategyFactory.register_strategy('mse', MSEStrategy)
        #StrategyFactory.register_strategy('bollinger_bands', BollingerBandsStrategy)
        
        # Register the 4 MSE strategy variations
        StrategyFactory.register_strategy('mse_20pct_with_cascade', MSE20WithCascade)
        StrategyFactory.register_strategy('mse_20pct_no_cascade', MSE20NoCascade)
        StrategyFactory.register_strategy('mse_80pct_with_cascade', MSE80WithCascade)
        StrategyFactory.register_strategy('mse_80pct_no_cascade', MSE80NoCascade)
        
        # Register the 2 new MSE strategy variations with MACD crossover exits
        StrategyFactory.register_strategy('mse_20pct_no_cascade_with_macd_exit', MSE20NoCascadeWithMACDExit)
        StrategyFactory.register_strategy('mse_80pct_no_cascade_with_macd_exit', MSE80NoCascadeWithMACDExit)
        
        # Register live-matching strategy (no shift for current bar indicators)
        StrategyFactory.register_strategy('mse_80pct_no_cascade_live_matching', MSE80NoCascadeLiveMatching)
        StrategyFactory.register_strategy('mse_5min_validation', MSE5MinValidation)
        
        
        # Note: Add your custom strategies here
        # StrategyFactory.register_strategy('your_strategy', YourStrategyClass)
        
        return True
    except Exception as e:
        print(f"Error registering strategies: {e}")
        return False