# Backtester/strategies/register_strategies.py
from .strategy_factory import StrategyFactory
from .strategy_sma_crossover import SMAcrossoverStrategy
from .strategy_bollinger_bands import BollingerBandsStrategy

# Enhanced MSE strategy (audit compliant version 2.0)
try:
    from .enhanced_mse_strategy import EnhancedMSEStrategy  # Audit compliant MSE strategy
except Exception:
    # Keep registration resilient if strategy is missing
    EnhancedMSEStrategy = None

# Legacy MSE strategy (for comparison if available)
try:
    from .strategy_mse import MSEStrategy as LegacyMSEStrategy  # Legacy version for comparison
except Exception:
    LegacyMSEStrategy = None

# Backtesting MSE strategy (bias-free implementation)
try:
    from .mse_strategy_backtesting import MSEStrategyBacktesting  # Bias-free MSE for backtesting
except Exception:
    MSEStrategyBacktesting = None

def register_all_strategies():
    """
    Register all available strategies with the factory.
    
    After cleanup: Only core strategies remain, eliminating 87% code duplication
    from MSE variants as identified in the analysis report.
    """
    try:
        # Register core strategies
        StrategyFactory.register_strategy('sma_crossover', SMAcrossoverStrategy)
        StrategyFactory.register_strategy('bollinger_bands', BollingerBandsStrategy)

        # Register backtesting MSE strategy (bias-free implementation) - PRIMARY MSE STRATEGY
        if MSEStrategyBacktesting:
            StrategyFactory.register_strategy('mse', MSEStrategyBacktesting)  # Primary MSE strategy
            StrategyFactory.register_strategy('mse_backtesting', MSEStrategyBacktesting)  # Alias
            
        # Register Enhanced MSE strategy (audit compliant) - if available
        if EnhancedMSEStrategy:
            StrategyFactory.register_strategy('mse_enhanced', EnhancedMSEStrategy)
            
        # Register legacy MSE strategy for comparison testing - if available
        if LegacyMSEStrategy:
            StrategyFactory.register_strategy('mse_legacy', LegacyMSEStrategy)
        
        # Note: Previous MSE variants have been removed to eliminate code duplication
        # Parameters can now be configured via the unified MSE strategy
        # Add your custom strategies here:
        # StrategyFactory.register_strategy('your_strategy', YourStrategyClass)
        
        return True
    except Exception as e:
        print(f"Error registering strategies: {e}")
        return False
