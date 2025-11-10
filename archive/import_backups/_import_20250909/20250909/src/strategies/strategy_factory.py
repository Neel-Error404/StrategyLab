# Backtester/strategies/strategy_factory.py
from typing import Dict, Any, Type, Optional
import logging
import inspect
from .strategy_base import StrategyBase

class StrategyFactory:
    """
    Factory class for creating strategy instances.
    """
    _strategies = {}
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class):
        """
        Register a strategy with the factory.
        """
        # More robust class checking
        if not inspect.isclass(strategy_class):
            raise TypeError(f"Expected a class, got {type(strategy_class)}")
            
        # Try to check inheritance, but with a fallback
        try:
            is_subclass = issubclass(strategy_class, StrategyBase)
        except TypeError:
            # If type checking fails, check for the required methods
            required_methods = ['prepare_data', 'generate_signals']
            if all(hasattr(strategy_class, method) for method in required_methods):
                is_subclass = True
            else:
                is_subclass = False
        
        if not is_subclass:
            raise TypeError(f"Strategy class must inherit from StrategyBase, got {strategy_class}")
            
        cls._strategies[name.lower()] = strategy_class
        logging.info(f"Strategy '{name}' registered successfully")
    
    @classmethod
    def create_strategy(cls, name: str, parameters: Optional[Dict[str, Any]] = None) -> StrategyBase:
        """
        Create a strategy instance.
        """
        name = name.lower()
        if name not in cls._strategies:
            available = ", ".join(cls._strategies.keys())
            raise ValueError(f"Strategy '{name}' not registered. Available: {available}")
        
        strategy_class = cls._strategies[name]
        return strategy_class(name, parameters)
    
    @classmethod
    def list_strategies(cls) -> dict:
        """
        List all available registered strategies.

        Returns:
            Dictionary mapping strategy names to strategy classes
        """
        return cls._strategies

    @classmethod
    def get_strategy(cls, name: str, parameters=None):
        """
        Get a strategy instance by name.

        Args:
            name: The name of the strategy to get
            parameters: Optional parameters to pass to the strategy constructor

        Returns:
            An instance of the requested strategy
        """
        name = name.lower()
        if name not in cls._strategies:
            logging.error(f"Strategy '{name}' not found in registered strategies")
            return None

        strategy_class = cls._strategies[name]
        try:
            return strategy_class(name, parameters)
        except Exception as e:
            logging.error(f"Error creating strategy '{name}': {e}")
            return None
# Register built-in strategies
#StrategyFactory.register_strategy('mse', MSEStrategy)