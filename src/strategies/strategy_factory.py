# Backtester/strategies/strategy_factory.py
from typing import Dict, Any, Type, Optional, List, Set
import logging
import inspect
from .strategy_base import StrategyBase

class StrategyFactory:
    """
    Enhanced factory class for creating strategy instances with multi-timeframe support.
    
    Features:
    - Strategy registration with timeframe validation
    - Strategy-timeframe mapping registry
    - Timeframe requirement discovery
    - CLI integration support
    """
    _strategies = {}  # Maps strategy_name -> strategy_class
    _strategy_timeframes = {}  # Maps strategy_name -> List[timeframes]
    _timeframe_strategies = {}  # Maps timeframe -> List[strategy_names]
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class):
        """
        Register a strategy with enhanced timeframe validation and mapping.
        
        Args:
            name: Strategy name (e.g., 'mse', 'sma_crossover')
            strategy_class: Strategy class that inherits from StrategyBase
        """
        logger = logging.getLogger("StrategyFactory")
        
        # Validate strategy class
        if not inspect.isclass(strategy_class):
            raise TypeError(f"Expected a class, got {type(strategy_class)}")
            
        # Check inheritance with fallback
        try:
            is_subclass = issubclass(strategy_class, StrategyBase)
        except TypeError:
            # Fallback: check for required methods
            required_methods = ['prepare_data', 'generate_signals', 'required_timeframes']
            is_subclass = all(hasattr(strategy_class, method) for method in required_methods)
        
        if not is_subclass:
            raise TypeError(f"Strategy class must inherit from StrategyBase, got {strategy_class}")
        
        # ENFORCED VALIDATION: Timeframe requirements and warmup periods
        try:
            # Create temporary instance to get timeframe requirements
            temp_instance = strategy_class(f"temp_{name}")
            required_timeframes = temp_instance.required_timeframes
            
            # ENFORCE: required_timeframes must be a non-empty list
            if not isinstance(required_timeframes, list):
                raise ValueError(f"REGISTRATION FAILED - Strategy {name}: required_timeframes must be a list, got {type(required_timeframes)}")
            
            if not required_timeframes:
                raise ValueError(f"REGISTRATION FAILED - Strategy {name}: required_timeframes cannot be empty")
            
            # ENFORCE: Multi-timeframe strategies must have warmup periods defined
            if len(required_timeframes) > 1:
                if not hasattr(temp_instance, 'warmup_periods'):
                    raise ValueError(f"REGISTRATION FAILED - Multi-timeframe strategy {name}: warmup_periods must be defined for all timeframes")
                
                # Validate warmup period for each timeframe
                for timeframe in required_timeframes:
                    if timeframe not in temp_instance.warmup_periods:
                        raise ValueError(f"REGISTRATION FAILED - Strategy {name}: warmup period missing for timeframe {timeframe}")
                    
                    if temp_instance.warmup_periods[timeframe] <= 0:
                        raise ValueError(f"REGISTRATION FAILED - Strategy {name}: warmup period for {timeframe} must be positive")
                
                logger.info(f"✅ Multi-timeframe validation passed for {name}: warmup periods {temp_instance.warmup_periods}")
            
            # ENFORCE: Validate timeframe strings
            valid_timeframes = {'1m', '2m', '3m', '5m', '10m', '15m', '30m', '1h', '2h', '4h', 'day', 'week', 'month'}
            invalid_timeframes = set(required_timeframes) - valid_timeframes
            if invalid_timeframes:
                logger.warning(f"Strategy {name}: Non-standard timeframes detected: {invalid_timeframes}")
            
            # ENFORCE: Multi-timeframe strategies must implement required methods
            if len(required_timeframes) > 1:
                required_methods = ['_validate_timeframes', 'prepare_data', 'generate_signals']
                missing_methods = [method for method in required_methods if not hasattr(temp_instance, method)]
                if missing_methods:
                    raise ValueError(f"REGISTRATION FAILED - Multi-timeframe strategy {name}: missing required methods {missing_methods}")
            
        except Exception as e:
            raise ValueError(f"STRATEGY REGISTRATION ENFORCEMENT FAILED for {name}: {e}")
        
        # Register strategy
        name_lower = name.lower()
        cls._strategies[name_lower] = strategy_class
        cls._strategy_timeframes[name_lower] = required_timeframes
        
        # Update reverse mapping (timeframe -> strategies)
        for timeframe in required_timeframes:
            if timeframe not in cls._timeframe_strategies:
                cls._timeframe_strategies[timeframe] = []
            if name_lower not in cls._timeframe_strategies[timeframe]:
                cls._timeframe_strategies[timeframe].append(name_lower)
        
        logger.info(f"✅ Strategy '{name}' registered successfully with timeframes: {required_timeframes}")
    
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
    def list_strategies(cls) -> Dict[str, Type[StrategyBase]]:
        """
        List all available registered strategies.

        Returns:
            Dictionary mapping strategy names to strategy classes
        """
        return cls._strategies
    
    @classmethod
    def get_strategy_timeframes(cls, name: str) -> List[str]:
        """
        Get timeframe requirements for a specific strategy.
        
        Args:
            name: Strategy name
            
        Returns:
            List of required timeframes
        """
        name_lower = name.lower()
        if name_lower not in cls._strategy_timeframes:
            raise ValueError(f"Strategy '{name}' not registered")
        return cls._strategy_timeframes[name_lower]
    
    @classmethod
    def get_strategies_by_timeframe(cls, timeframe: str) -> List[str]:
        """
        Get all strategies that use a specific timeframe.
        
        Args:
            timeframe: Timeframe string (e.g., '5m', '15m')
            
        Returns:
            List of strategy names that require this timeframe
        """
        return cls._timeframe_strategies.get(timeframe, [])
    
    @classmethod
    def get_all_timeframes(cls) -> Set[str]:
        """
        Get all timeframes used by registered strategies.
        
        Returns:
            Set of all timeframes across all strategies
        """
        all_timeframes = set()
        for timeframes in cls._strategy_timeframes.values():
            all_timeframes.update(timeframes)
        return all_timeframes
    
    @classmethod
    def validate_timeframe_availability(cls, strategy_name: str, available_timeframes: List[str]) -> Dict[str, Any]:
        """
        Validate if required timeframes are available for a strategy.
        
        Args:
            strategy_name: Name of the strategy to validate
            available_timeframes: List of timeframes available in the data
            
        Returns:
            Dictionary with validation results
        """
        name_lower = strategy_name.lower()
        
        if name_lower not in cls._strategy_timeframes:
            return {
                'valid': False,
                'error': f"Strategy '{strategy_name}' not registered",
                'required': [],
                'missing': [],
                'available': available_timeframes
            }
        
        required = set(cls._strategy_timeframes[name_lower])
        available = set(available_timeframes)
        missing = required - available
        
        return {
            'valid': len(missing) == 0,
            'error': f"Missing timeframes: {list(missing)}" if missing else None,
            'required': list(required),
            'missing': list(missing),
            'available': list(available)
        }
    
    @classmethod
    def get_strategy_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive information about all registered strategies.
        
        Returns:
            Dictionary with strategy information including timeframes
        """
        info = {}
        for name, strategy_class in cls._strategies.items():
            info[name] = {
                'class': strategy_class.__name__,
                'timeframes': cls._strategy_timeframes.get(name, []),
                'module': strategy_class.__module__,
                'description': getattr(strategy_class, '__doc__', 'No description available')
            }
        return info

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
    
    @classmethod
    def clear_registry(cls):
        """
        Clear all registered strategies (useful for testing).
        """
        cls._strategies.clear()
        cls._strategy_timeframes.clear()
        cls._timeframe_strategies.clear()
        logging.info("Strategy registry cleared")

# Built-in strategies will be registered via register_strategies.py