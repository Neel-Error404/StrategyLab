# Backtester/strategies/support/register_strategies.py
from .strategy_factory import StrategyFactory
from ..mse_strategy_backtesting import MSEStrategyBacktesting
from ..strategy_mac import StrategyMAC
from ..ema_pvt_strategy import EmaPivotStrategy
from ..bollinger_squeeze_strategy import BollingerSqueezeStrategy
from ..marketcap_boundary_strategy import MarketCapBoundaryStrategy

def register_all_strategies():
    """
    Register all available strategies with the factory.
    
    After cleanup: Only core strategies remain, eliminating 87% code duplication
    from MSE variants as identified in the analysis report.
    """
    try:
        # Register core strategies
        StrategyFactory.register_strategy('mse_strategy_backtesting', MSEStrategyBacktesting)
        StrategyFactory.register_strategy('mse', MSEStrategyBacktesting)  # Alias for CLI/docs compatibility
        StrategyFactory.register_strategy('strategy_mac', StrategyMAC)
        StrategyFactory.register_strategy('ema_pvt_strategy', EmaPivotStrategy)
        StrategyFactory.register_strategy('bollinger_squeeze_strategy', BollingerSqueezeStrategy)
        StrategyFactory.register_strategy('marketcap_boundary', MarketCapBoundaryStrategy)

        # Note: Proprietary MSE implementations removed from public release.
        # Add your custom strategies here:
        # StrategyFactory.register_strategy('your_strategy', YourStrategyClass)
        
        return True
    except Exception as e:
        print(f"Error registering strategies: {e}")
        return False
