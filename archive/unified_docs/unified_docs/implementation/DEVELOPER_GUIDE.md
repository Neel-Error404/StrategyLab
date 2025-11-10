# 👩‍💻 Unified Trading System Developer Guide

## 🎯 Development Philosophy

### **Core Principles**

1. **Single Source of Truth**: One strategy implementation for both environments
2. **Environment Agnostic**: Strategies shouldn't know if they're running in backtest or live
3. **Clean Abstractions**: Hide complexity behind well-defined interfaces
4. **Fail Fast**: Early detection of issues through comprehensive validation
5. **Performance First**: Maintain low-latency operation for live trading

### **Code Quality Standards**

- **Type Hints**: Required for all public interfaces and methods
- **Documentation**: Comprehensive docstrings following Google style
- **Testing**: >90% test coverage for core components
- **Error Handling**: Explicit error handling with meaningful messages
- **Performance**: Benchmark critical paths and optimize hotspots

## 🏗️ Development Environment Setup

### **Prerequisites**

- Python 3.8+ (recommended: 3.9)
- Git with submodule support
- IDE with Python support (recommended: VS Code)
- Windows PowerShell or Linux/macOS terminal

### **Initial Setup**

```bash
# 1. Clone the unified repository
git clone https://github.com/your-org/unified-trading.git
cd unified-trading

# 2. Initialize and update submodules
git submodule init
git submodule update --recursive

# 3. Create Python virtual environment
python -m venv venv

# 4. Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 6. Install pre-commit hooks
pre-commit install

# 7. Verify installation
python -c "from core.engine.unified_engine import UnifiedTradingEngine; print('Setup successful!')"
```

### **Development Dependencies**

```python
# requirements-dev.txt
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0
pytest-benchmark>=4.0.0
black>=22.0.0
flake8>=5.0.0
mypy>=1.0.0
isort>=5.12.0
pre-commit>=3.0.0
sphinx>=5.0.0
streamlit>=1.20.0  # For validation dashboard
plotly>=5.0.0      # For visualization
```

### **IDE Configuration (VS Code)**

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.path": "isort",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestArgs": [
        "tests"
    ]
}
```

## 🏛️ Architecture Overview for Developers

### **Project Structure**

```
unified_trading/
├── core/                          # Core system components
│   ├── engine/                    # Main trading engine
│   │   ├── unified_engine.py      # Primary orchestration
│   │   └── execution_manager.py   # Execution coordination
│   ├── interfaces/                # Abstract interfaces
│   │   ├── strategy_interface.py  # Universal strategy base
│   │   ├── data_provider.py       # Data abstraction
│   │   └── execution_adapter.py   # Execution abstraction
│   ├── context/                   # Context management
│   │   ├── market_context.py      # Market data context
│   │   └── strategy_state.py      # Strategy state management
│   └── configuration/             # Configuration system
│       ├── config_manager.py      # Configuration loading
│       └── validation.py          # Config validation
├── strategies/                    # Strategy implementations
│   ├── base/                      # Base strategy classes
│   │   └── universal_strategy.py  # Universal base class
│   └── mse/                       # MSE strategy
│       └── unified_mse_strategy.py # Unified MSE implementation
├── adapters/                      # Environment adapters
│   ├── backtester/               # Backtester integration
│   │   ├── backtester_adapter.py # Main adapter
│   │   └── data_provider.py      # Historical data provider
│   └── live_trading/             # Live trading integration
│       ├── live_adapter.py       # Main adapter
│       └── data_provider.py      # Real-time data provider
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   ├── validation/               # Validation tests
│   └── performance/              # Performance tests
├── config/                       # Configuration files
│   ├── templates/                # Configuration templates
│   └── examples/                 # Example configurations
├── submodules/                   # Git submodules
│   ├── backtester/              # Existing backtester
│   └── live_module/             # Existing live module
└── docs/                        # Documentation
```

## 🧩 Core Components Development

### **1. Universal Strategy Interface**

#### **Interface Definition**

```python
# core/interfaces/strategy_interface.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class StrategyRequirements:
    """Strategy data and computational requirements"""
    timeframes: List[str]                    # Required timeframes ['5m', '15m']
    warmup_periods: Dict[str, int]          # Warmup periods per timeframe
    minimum_candles: Dict[str, int]         # Minimum candles needed per timeframe
    symbols_per_execution: int = 1          # Number of symbols processed together
    requires_position_context: bool = True   # Whether strategy needs position info

@dataclass
class Signal:
    """Universal signal format"""
    action: str                             # 'BUY', 'SELL', 'HOLD'
    symbol: str                             # Symbol to trade
    price: float                            # Signal price
    confidence: float                       # Confidence 0.0-1.0
    reason: str                             # Human-readable reason
    timestamp: datetime                     # Signal generation time
    is_exit: bool = False                   # Whether this is an exit signal
    metadata: Dict[str, Any] = None         # Additional signal data

    def __post_init__(self):
        """Validate signal data"""
        if self.action not in ['BUY', 'SELL', 'HOLD']:
            raise ValueError(f"Invalid action: {self.action}")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")

        if self.price <= 0:
            raise ValueError(f"Price must be positive, got {self.price}")

class UniversalStrategy(ABC):
    """
    Base class for all unified trading strategies

    This interface ensures strategies can run identically in both
    backtesting and live trading environments.
    """

    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.parameters = parameters or {}
        self.version = "1.0"

    @abstractmethod
    def get_requirements(self) -> StrategyRequirements:
        """
        Declare strategy requirements

        Returns:
            StrategyRequirements object specifying data and computational needs
        """
        pass

    @abstractmethod
    def generate_signal(self,
                       context: 'MarketContext',
                       state: 'StrategyState') -> Optional[Signal]:
        """
        Generate trading signal based on market context and strategy state

        Args:
            context: Current market data and environment context
            state: Strategy state and position information

        Returns:
            Signal object if conditions met, None otherwise
        """
        pass

    def initialize_state(self) -> Dict[str, Any]:
        """
        Initialize strategy-specific state variables

        Returns:
            Dictionary of initial state variables
        """
        return {}

    def validate_signal(self, signal: Signal) -> bool:
        """
        Validate generated signal (override for custom validation)

        Args:
            signal: Signal to validate

        Returns:
            True if signal is valid, False otherwise
        """
        return True

    def on_execution_result(self, signal: Signal, result: Dict[str, Any]) -> None:
        """
        React to execution result (override for post-execution logic)

        Args:
            signal: Original signal that was executed
            result: Execution result with status and details
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Return strategy metadata for monitoring and debugging"""
        return {
            'name': self.name,
            'version': self.version,
            'parameters': self.parameters,
            'requirements': self.get_requirements().__dict__
        }
```

#### **Implementation Example**

```python
# strategies/mse/unified_mse_strategy.py
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from core.interfaces.strategy_interface import (
    UniversalStrategy, StrategyRequirements, Signal
)
from core.context.market_context import MarketContext
from core.context.strategy_state import StrategyState

class UnifiedMSEStrategy(UniversalStrategy):
    """
    Unified MSE Strategy Implementation

    Multi-Signal Entry strategy using 4 indicators across 2 timeframes:
    - 5min MACD vs Signal
    - 15min MACD vs Signal
    - 5min EMA9 vs EMA20
    - 15min EMA9 vs EMA20

    Entry: ALL 4 indicators must align (bullish or bearish)
    Exit: 80% peak/valley MACD histogram logic
    """

    def __init__(self, parameters: Dict[str, Any] = None):
        super().__init__("MSE", parameters)

        # Strategy-specific parameters
        self.exit_threshold = self.parameters.get('exit_threshold', 0.80)
        self.confidence_base = self.parameters.get('confidence_base', 0.70)

        # MACD parameters
        self.macd_fast = self.parameters.get('macd_fast', 12)
        self.macd_slow = self.parameters.get('macd_slow', 26)
        self.macd_signal = self.parameters.get('macd_signal', 9)

        # EMA parameters
        self.ema_fast = self.parameters.get('ema_fast', 9)
        self.ema_slow = self.parameters.get('ema_slow', 20)

    def get_requirements(self) -> StrategyRequirements:
        """Declare MSE strategy requirements"""
        return StrategyRequirements(
            timeframes=['5m', '15m'],
            warmup_periods={
                '5m': 175,   # 35 periods * 5 minutes
                '15m': 525   # 35 periods * 15 minutes
            },
            minimum_candles={
                '5m': 40,    # Minimum for reliable calculations
                '15m': 40
            },
            symbols_per_execution=1,
            requires_position_context=True
        )

    def generate_signal(self, context: MarketContext, state: StrategyState) -> Optional[Signal]:
        """
        Generate MSE signal using 4-indicator system

        This is the core logic that runs identically in both environments
        """

        # Get timeframe data
        try:
            data_5m = context.get_timeframe_data('5m')
            data_15m = context.get_timeframe_data('15m')
        except Exception as e:
            # Log error but don't crash
            context.log_warning(f"Failed to get timeframe data: {e}")
            return None

        # Validate data availability
        if data_5m is None or data_15m is None or data_5m.empty or data_15m.empty:
            return None

        # Calculate indicators for both timeframes
        indicators_5m = self._calculate_indicators(data_5m)
        indicators_15m = self._calculate_indicators(data_15m)

        # Get current position context
        position = state.get_position_info()
        current_price = context.get_current_price()
        current_time = context.get_current_time()

        if not position.has_position:
            # Entry signal logic
            return self._check_entry_conditions(
                indicators_5m, indicators_15m,
                context.symbol, current_price, current_time
            )
        else:
            # Exit signal logic
            return self._check_exit_conditions(
                indicators_15m, position, state,
                current_price, current_time
            )

    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate MACD and EMA indicators for given timeframe"""

        if len(df) < max(self.macd_slow, self.ema_slow):
            return {'valid': False}

        closes = df['close'].values

        # MACD calculation
        ema_fast = self._calculate_ema(closes, self.macd_fast)
        ema_slow = self._calculate_ema(closes, self.macd_slow)
        macd_line = ema_fast - ema_slow

        # Signal line (EMA of MACD line)
        macd_values = []
        for i in range(len(closes)):
            if i >= self.macd_slow - 1:
                window_closes = closes[:i+1]
                fast_ema = self._calculate_ema(window_closes, self.macd_fast)
                slow_ema = self._calculate_ema(window_closes, self.macd_slow)
                macd_values.append(fast_ema - slow_ema)

        if len(macd_values) >= self.macd_signal:
            signal_line = self._calculate_ema(np.array(macd_values), self.macd_signal)
        else:
            signal_line = 0

        histogram = macd_line - signal_line

        # EMA calculation
        ema_fast_val = self._calculate_ema(closes, self.ema_fast)
        ema_slow_val = self._calculate_ema(closes, self.ema_slow)

        return {
            'valid': True,
            'macd_line': macd_line,
            'signal_line': signal_line,
            'macd_histogram': histogram,
            'macd_bullish': macd_line > signal_line,
            'ema_fast': ema_fast_val,
            'ema_slow': ema_slow_val,
            'ema_bullish': ema_fast_val > ema_slow_val
        }

    def _calculate_ema(self, values: np.ndarray, period: int) -> float:
        """Calculate EMA using standard 2/(period+1) smoothing"""
        if len(values) < period:
            return 0

        alpha = 2.0 / (period + 1)
        ema = values[0]

        for value in values[1:]:
            ema = alpha * value + (1 - alpha) * ema

        return ema

    def _check_entry_conditions(self, indicators_5m: Dict, indicators_15m: Dict,
                               symbol: str, price: float, timestamp) -> Optional[Signal]:
        """Check 4-indicator entry conditions"""

        # Validate indicators
        if not indicators_5m.get('valid') or not indicators_15m.get('valid'):
            return None

        # BUY: ALL 4 indicators bullish
        if (indicators_5m['macd_bullish'] and indicators_15m['macd_bullish'] and
            indicators_5m['ema_bullish'] and indicators_15m['ema_bullish']):

            confidence = self._calculate_confidence(indicators_5m, indicators_15m, 'BUY')

            return Signal(
                action='BUY',
                symbol=symbol,
                price=price,
                confidence=confidence,
                reason='4-indicator bullish alignment',
                timestamp=timestamp,
                metadata={
                    '5m_macd_hist': indicators_5m['macd_histogram'],
                    '15m_macd_hist': indicators_15m['macd_histogram'],
                    '5m_ema_diff': indicators_5m['ema_fast'] - indicators_5m['ema_slow'],
                    '15m_ema_diff': indicators_15m['ema_fast'] - indicators_15m['ema_slow']
                }
            )

        # SELL: ALL 4 indicators bearish
        elif (not indicators_5m['macd_bullish'] and not indicators_15m['macd_bullish'] and
              not indicators_5m['ema_bullish'] and not indicators_15m['ema_bullish']):

            confidence = self._calculate_confidence(indicators_5m, indicators_15m, 'SELL')

            return Signal(
                action='SELL',
                symbol=symbol,
                price=price,
                confidence=confidence,
                reason='4-indicator bearish alignment',
                timestamp=timestamp,
                metadata={
                    '5m_macd_hist': indicators_5m['macd_histogram'],
                    '15m_macd_hist': indicators_15m['macd_histogram']
                }
            )

        return None

    def _check_exit_conditions(self, indicators_15m: Dict, position, state: StrategyState,
                              price: float, timestamp) -> Optional[Signal]:
        """Check 80% peak/valley exit conditions"""

        if not indicators_15m.get('valid'):
            return None

        current_macd = indicators_15m['macd_histogram']

        # Get or initialize peak tracking
        peak_value = state.get('macd_peak', current_macd)
        peak_initialized = state.get('peak_initialized', False)

        # Initialize peak on first call
        if not peak_initialized:
            state.set('macd_peak', current_macd)
            state.set('peak_initialized', True)
            return None

        if position.side == 'LONG':
            # Track highest peak for long positions
            if current_macd > peak_value:
                peak_value = current_macd
                state.set('macd_peak', peak_value)

            # Exit when MACD drops below 80% of peak
            exit_threshold = peak_value * self.exit_threshold
            if current_macd <= exit_threshold:
                return Signal(
                    action='SELL',
                    symbol=position.symbol,
                    price=price,
                    confidence=0.90,
                    reason=f'80% peak exit: {current_macd:.4f} <= {exit_threshold:.4f}',
                    timestamp=timestamp,
                    is_exit=True
                )

        elif position.side == 'SHORT':
            # Track lowest valley for short positions
            if current_macd < peak_value:
                peak_value = current_macd
                state.set('macd_peak', peak_value)

            # Exit when MACD rises above 80% of valley
            exit_threshold = peak_value * self.exit_threshold
            if current_macd >= exit_threshold:
                return Signal(
                    action='BUY',
                    symbol=position.symbol,
                    price=price,
                    confidence=0.90,
                    reason=f'80% valley exit: {current_macd:.4f} >= {exit_threshold:.4f}',
                    timestamp=timestamp,
                    is_exit=True
                )

        return None

    def _calculate_confidence(self, ind_5m: Dict, ind_15m: Dict, direction: str) -> float:
        """Calculate dynamic confidence based on MACD histogram strength"""

        base_confidence = self.confidence_base

        hist_5m = ind_5m['macd_histogram']
        hist_15m = ind_15m['macd_histogram']

        if direction == 'BUY':
            strength_5m = max(0, hist_5m)
            strength_15m = max(0, hist_15m)
        else:  # SELL
            strength_5m = max(0, -hist_5m)
            strength_15m = max(0, -hist_15m)

        # Normalize and add bonus (up to 25% total bonus)
        bonus_5m = min(strength_5m / 1.0, 1.0) * 0.125
        bonus_15m = min(strength_15m / 0.5, 1.0) * 0.125

        final_confidence = base_confidence + bonus_5m + bonus_15m

        # Bound between 0.5 and 0.95
        return max(0.5, min(final_confidence, 0.95))

    def initialize_state(self) -> Dict[str, Any]:
        """Initialize MSE-specific state variables"""
        return {
            'macd_peak': 0.0,
            'peak_initialized': False,
            'last_signal_time': None,
            'signal_count': 0
        }

    def validate_signal(self, signal: Signal) -> bool:
        """Validate MSE signal"""

        # Basic validation from parent
        if not super().validate_signal(signal):
            return False

        # MSE-specific validation
        if signal.action not in ['BUY', 'SELL']:
            return False

        # Check confidence is reasonable
        if signal.confidence < 0.5:
            return False

        # Validate metadata if present
        if signal.metadata:
            required_fields = ['5m_macd_hist', '15m_macd_hist'] if not signal.is_exit else []
            for field in required_fields:
                if field not in signal.metadata:
                    return False

        return True
```

### **2. Market Context Development**

```python
# core/context/market_context.py
from typing import Dict, Any, Optional, List
from datetime import datetime, time
import pandas as pd
import logging

class MarketContext:
    """
    Unified market data context

    Provides consistent interface to market data regardless of source
    (historical files vs real-time streams)
    """

    def __init__(self, data_provider: 'DataProvider', symbol: str):
        self.data_provider = data_provider
        self.symbol = symbol
        self.current_time = None
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{symbol}")

    def get_timeframe_data(self, timeframe: str, periods: int = None) -> Optional[pd.DataFrame]:
        """
        Get OHLCV data for specific timeframe

        Args:
            timeframe: Timeframe string ('1m', '5m', '15m', etc.)
            periods: Number of periods to retrieve (None = all available)

        Returns:
            DataFrame with OHLCV data or None if not available
        """
        try:
            data = self.data_provider.get_data(self.symbol, timeframe, periods)

            if data is None or data.empty:
                self.logger.debug(f"No data available for {timeframe}")
                return None

            # Validate data structure
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                self.logger.error(f"Missing columns in {timeframe} data: {missing_columns}")
                return None

            return data

        except Exception as e:
            self.logger.error(f"Error getting {timeframe} data: {e}")
            return None

    def get_current_price(self) -> float:
        """Get current/latest price for the symbol"""
        try:
            return self.data_provider.get_current_price(self.symbol)
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return 0.0

    def get_current_time(self) -> datetime:
        """Get current timestamp in strategy execution"""
        return self.current_time

    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            return self.data_provider.is_market_open()
        except Exception:
            # Default to market hours check
            current = self.get_current_time()
            if current:
                current_time = current.time()
                return time(9, 15) <= current_time <= time(15, 15)
            return False

    def log_info(self, message: str):
        """Log informational message with context"""
        self.logger.info(f"{self.symbol}: {message}")

    def log_warning(self, message: str):
        """Log warning message with context"""
        self.logger.warning(f"{self.symbol}: {message}")

    def log_error(self, message: str):
        """Log error message with context"""
        self.logger.error(f"{self.symbol}: {message}")

    def get_metadata(self) -> Dict[str, Any]:
        """Get context metadata for debugging"""
        return {
            'symbol': self.symbol,
            'current_time': self.current_time,
            'data_provider_type': type(self.data_provider).__name__,
            'market_open': self.is_market_open(),
            'current_price': self.get_current_price()
        }
```

### **3. Strategy State Development**

```python
# core/context/strategy_state.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class PositionInfo:
    """Unified position information"""
    symbol: str
    has_position: bool = False
    side: str = "FLAT"  # 'LONG', 'SHORT', 'FLAT'
    quantity: int = 0
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    unrealized_pnl: float = 0.0

    def is_long(self) -> bool:
        return self.side == "LONG"

    def is_short(self) -> bool:
        return self.side == "SHORT"

    def is_flat(self) -> bool:
        return self.side == "FLAT"

class StrategyState(ABC):
    """
    Abstract strategy state management

    Provides unified interface for strategy state regardless of storage backend
    (in-memory for backtesting vs external files for live trading)
    """

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get strategy variable"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set strategy variable"""
        pass

    @abstractmethod
    def get_position_info(self) -> PositionInfo:
        """Get current position information"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all strategy state (when position closes)"""
        pass

    def get_all_variables(self) -> Dict[str, Any]:
        """Get all strategy variables (for debugging)"""
        return {}

class BacktestStrategyState(StrategyState):
    """Strategy state stored in memory for backtesting"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.variables: Dict[str, Any] = {}
        self.position = PositionInfo(symbol=symbol)
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{symbol}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get variable from memory"""
        return self.variables.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set variable in memory"""
        self.variables[key] = value
        self.logger.debug(f"Set {key} = {value}")

    def get_position_info(self) -> PositionInfo:
        """Get position from internal tracking"""
        return self.position

    def clear(self) -> None:
        """Clear all state"""
        self.variables.clear()
        self.position = PositionInfo(symbol=self.symbol)
        self.logger.debug("State cleared")

    def get_all_variables(self) -> Dict[str, Any]:
        """Get all variables for debugging"""
        return self.variables.copy()

    def update_position(self, side: str, quantity: int, entry_price: float, entry_time: datetime):
        """Update position information (backtesting only)"""
        self.position.side = side
        self.position.quantity = quantity
        self.position.entry_price = entry_price
        self.position.entry_time = entry_time
        self.position.has_position = abs(quantity) > 0

class LiveStrategyState(StrategyState):
    """Strategy state stored externally for live trading"""

    def __init__(self, position_manager: 'UnifiedPositionManager', strategy_name: str, symbol: str):
        self.position_manager = position_manager
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{strategy_name}.{symbol}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get variable from external storage"""
        try:
            return self.position_manager.get_strategy_variable(
                self.strategy_name, self.symbol, key, default
            )
        except Exception as e:
            self.logger.error(f"Error getting variable {key}: {e}")
            return default

    def set(self, key: str, value: Any) -> None:
        """Set variable in external storage"""
        try:
            self.position_manager.set_strategy_variable(
                self.strategy_name, self.symbol, key, value
            )
            self.logger.debug(f"Set {key} = {value}")
        except Exception as e:
            self.logger.error(f"Error setting variable {key}: {e}")

    def get_position_info(self) -> PositionInfo:
        """Get position from external position manager"""
        try:
            return self.position_manager.get_position_info(self.strategy_name, self.symbol)
        except Exception as e:
            self.logger.error(f"Error getting position info: {e}")
            return PositionInfo(symbol=self.symbol)

    def clear(self) -> None:
        """Clear state in external storage"""
        try:
            self.position_manager.clear_strategy_state(self.strategy_name, self.symbol)
            self.logger.debug("State cleared")
        except Exception as e:
            self.logger.error(f"Error clearing state: {e}")

    def get_all_variables(self) -> Dict[str, Any]:
        """Get all variables from external storage"""
        try:
            return self.position_manager.get_all_strategy_variables(
                self.strategy_name, self.symbol
            )
        except Exception as e:
            self.logger.error(f"Error getting all variables: {e}")
            return {}
```

## 🧪 Testing Development Guidelines

### **Test Structure**

```
tests/
├── unit/                          # Unit tests
│   ├── test_strategy_interface.py # Interface tests
│   ├── test_market_context.py     # Context tests
│   └── test_strategy_state.py     # State tests
├── integration/                   # Integration tests
│   ├── test_backtester_adapter.py # Backtester integration
│   ├── test_live_adapter.py       # Live trading integration
│   └── test_end_to_end.py         # Full pipeline tests
├── validation/                    # Validation tests
│   ├── test_signal_parity.py      # Signal comparison tests
│   ├── test_performance_parity.py # Performance comparison tests
│   └── test_state_consistency.py  # State synchronization tests
└── performance/                   # Performance tests
    ├── test_latency.py            # Latency benchmarks
    └── test_memory_usage.py       # Memory usage tests
```

### **Unit Testing Guidelines**

```python
# tests/unit/test_unified_mse_strategy.py
import pytest
from unittest.mock import Mock, MagicMock
import pandas as pd
from datetime import datetime

from strategies.mse.unified_mse_strategy import UnifiedMSEStrategy
from core.context.market_context import MarketContext
from core.context.strategy_state import BacktestStrategyState, PositionInfo

class TestUnifiedMSEStrategy:
    """Comprehensive unit tests for MSE strategy"""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance for testing"""
        return UnifiedMSEStrategy()

    @pytest.fixture
    def mock_context(self):
        """Create mock market context"""
        context = Mock(spec=MarketContext)
        context.symbol = "TEST_SYMBOL"
        context.get_current_price.return_value = 100.0
        context.get_current_time.return_value = datetime.now()
        return context

    @pytest.fixture
    def mock_state(self):
        """Create mock strategy state"""
        state = BacktestStrategyState("TEST_SYMBOL")
        return state

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing"""
        dates = pd.date_range('2024-01-01 09:15', periods=100, freq='5min')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': [100 + i*0.1 for i in range(100)],
            'high': [101 + i*0.1 for i in range(100)],
            'low': [99 + i*0.1 for i in range(100)],
            'close': [100.5 + i*0.1 for i in range(100)],
            'volume': [1000] * 100
        })
        return data

    def test_strategy_requirements(self, strategy):
        """Test strategy requirements specification"""
        requirements = strategy.get_requirements()

        assert requirements.timeframes == ['5m', '15m']
        assert '5m' in requirements.warmup_periods
        assert '15m' in requirements.warmup_periods
        assert requirements.warmup_periods['15m'] == 525  # Critical for MACD
        assert requirements.requires_position_context is True

    def test_indicator_calculation(self, strategy, sample_ohlcv_data):
        """Test MACD and EMA calculations"""

        indicators = strategy._calculate_indicators(sample_ohlcv_data)

        assert indicators['valid'] is True
        assert 'macd_line' in indicators
        assert 'signal_line' in indicators
        assert 'macd_histogram' in indicators
        assert 'ema_fast' in indicators
        assert 'ema_slow' in indicators
        assert isinstance(indicators['macd_bullish'], bool)
        assert isinstance(indicators['ema_bullish'], bool)

    def test_entry_signal_generation_bullish(self, strategy, mock_context, mock_state, sample_ohlcv_data):
        """Test bullish entry signal generation"""

        # Mock timeframe data
        mock_context.get_timeframe_data.side_effect = lambda tf: sample_ohlcv_data

        # Mock position (no position)
        mock_state.get_position_info.return_value = PositionInfo("TEST_SYMBOL")

        # Create conditions for bullish signal
        # This would require mocking the indicator calculations to return bullish conditions

        signal = strategy.generate_signal(mock_context, mock_state)

        # Test depends on mock data - adjust based on actual indicator calculations
        if signal:
            assert signal.action == 'BUY'
            assert signal.symbol == "TEST_SYMBOL"
            assert 0.5 <= signal.confidence <= 0.95
            assert 'bullish alignment' in signal.reason.lower()

    def test_exit_signal_generation(self, strategy, mock_context, mock_state, sample_ohlcv_data):
        """Test exit signal generation"""

        # Mock timeframe data
        mock_context.get_timeframe_data.side_effect = lambda tf: sample_ohlcv_data

        # Mock position (has LONG position)
        position = PositionInfo(
            symbol="TEST_SYMBOL",
            has_position=True,
            side="LONG",
            quantity=100,
            entry_price=99.0
        )
        mock_state.get_position_info.return_value = position

        # Mock state variables for peak tracking
        mock_state.get.side_effect = lambda key, default=None: {
            'macd_peak': 0.5,  # Mock peak value
            'peak_initialized': True
        }.get(key, default)

        signal = strategy.generate_signal(mock_context, mock_state)

        # Exit signal depends on MACD histogram vs peak - adjust based on test data
        if signal:
            assert signal.action == 'SELL'
            assert signal.is_exit is True

    def test_signal_validation(self, strategy):
        """Test signal validation"""

        # Valid signal
        valid_signal = Signal(
            action='BUY',
            symbol='TEST',
            price=100.0,
            confidence=0.8,
            reason='Test signal',
            timestamp=datetime.now()
        )
        assert strategy.validate_signal(valid_signal) is True

        # Invalid action
        invalid_signal = Signal(
            action='INVALID',
            symbol='TEST',
            price=100.0,
            confidence=0.8,
            reason='Test signal',
            timestamp=datetime.now()
        )
        assert strategy.validate_signal(invalid_signal) is False

        # Low confidence
        low_confidence_signal = Signal(
            action='BUY',
            symbol='TEST',
            price=100.0,
            confidence=0.3,
            reason='Test signal',
            timestamp=datetime.now()
        )
        assert strategy.validate_signal(low_confidence_signal) is False

    def test_state_initialization(self, strategy):
        """Test strategy state initialization"""

        initial_state = strategy.initialize_state()

        assert 'macd_peak' in initial_state
        assert 'peak_initialized' in initial_state
        assert initial_state['peak_initialized'] is False
        assert initial_state['macd_peak'] == 0.0

    @pytest.mark.parametrize("timeframe", ['5m', '15m'])
    def test_insufficient_data_handling(self, strategy, mock_context, mock_state, timeframe):
        """Test handling of insufficient data"""

        # Create insufficient data
        insufficient_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='5min'),
            'open': [100] * 5,
            'high': [101] * 5,
            'low': [99] * 5,
            'close': [100] * 5,
            'volume': [1000] * 5
        })

        mock_context.get_timeframe_data.return_value = insufficient_data
        mock_state.get_position_info.return_value = PositionInfo("TEST_SYMBOL")

        signal = strategy.generate_signal(mock_context, mock_state)

        # Should return None for insufficient data
        assert signal is None

    def test_confidence_calculation(self, strategy):
        """Test confidence calculation logic"""

        # Mock indicators with strong signals
        strong_indicators_5m = {
            'valid': True,
            'macd_histogram': 1.5,  # Strong positive
            'macd_bullish': True
        }

        strong_indicators_15m = {
            'valid': True,
            'macd_histogram': 0.8,  # Strong positive
            'macd_bullish': True
        }

        confidence = strategy._calculate_confidence(
            strong_indicators_5m, strong_indicators_15m, 'BUY'
        )

        assert 0.5 <= confidence <= 0.95
        assert confidence > strategy.confidence_base  # Should be higher than base
```

### **Integration Testing Guidelines**

```python
# tests/integration/test_strategy_adapter_integration.py
import pytest
import tempfile
import os
from datetime import datetime, timedelta
import pandas as pd

from core.engine.unified_engine import UnifiedTradingEngine
from strategies.mse.unified_mse_strategy import UnifiedMSEStrategy
from adapters.backtester.backtester_adapter import BacktesterAdapter

class TestStrategyAdapterIntegration:
    """Integration tests between strategies and adapters"""

    @pytest.fixture
    def test_config(self):
        """Create test configuration"""
        return {
            'execution': {
                'mode': 'backtest'
            },
            'portfolio': {
                'symbols': ['TEST_SYMBOL'],
                'initial_capital': 100000
            },
            'data_source': {
                'type': 'csv',
                'path': self.create_test_data()
            }
        }

    def create_test_data(self):
        """Create test CSV data"""
        temp_dir = tempfile.mkdtemp()

        # Create sample data
        dates = pd.date_range('2024-01-01 09:15', periods=1000, freq='5min')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': [100 + i*0.01 for i in range(1000)],
            'high': [101 + i*0.01 for i in range(1000)],
            'low': [99 + i*0.01 for i in range(1000)],
            'close': [100.5 + i*0.01 for i in range(1000)],
            'volume': [1000] * 1000
        })

        # Save as CSV
        csv_path = os.path.join(temp_dir, 'TEST_SYMBOL_5m.csv')
        data.to_csv(csv_path, index=False)

        return temp_dir

    def test_end_to_end_backtest_execution(self, test_config):
        """Test complete end-to-end backtest execution"""

        # Create unified engine
        engine = UnifiedTradingEngine(test_config)

        # Add strategy
        strategy = UnifiedMSEStrategy()
        engine.add_strategy(strategy)

        # Execute backtest
        results = engine.run()

        # Validate results
        assert results is not None
        assert 'strategies' in results
        assert 'MSE' in results['strategies']

        strategy_results = results['strategies']['MSE']
        assert 'TEST_SYMBOL' in strategy_results

        symbol_results = strategy_results['TEST_SYMBOL']
        assert 'trades' in symbol_results
        assert 'performance_metrics' in symbol_results

    def test_adapter_data_consistency(self, test_config):
        """Test data consistency between adapters"""

        # Test that same data produces same results through different paths
        adapter = BacktesterAdapter(test_config)
        strategy = UnifiedMSEStrategy()

        # Execute twice - should be identical
        results1 = adapter.execute_strategy(strategy, 'TEST_SYMBOL')
        results2 = adapter.execute_strategy(strategy, 'TEST_SYMBOL')

        # Compare results
        assert len(results1['trades']) == len(results2['trades'])

        for trade1, trade2 in zip(results1['trades'], results2['trades']):
            assert trade1['entry_timestamp'] == trade2['entry_timestamp']
            assert trade1['entry_price'] == trade2['entry_price']
            assert trade1.get('exit_timestamp') == trade2.get('exit_timestamp')
```

## 🚀 Performance Optimization Guidelines

### **Critical Performance Areas**

1. **Signal Generation Latency**: <10ms per symbol
2. **Data Provider Access**: <5ms per timeframe request
3. **State Management**: <1ms per get/set operation
4. **Memory Usage**: <100MB total system memory

### **Performance Testing**

```python
# tests/performance/test_strategy_performance.py
import pytest
import time
import psutil
import os
from memory_profiler import profile

from strategies.mse.unified_mse_strategy import UnifiedMSEStrategy
from tests.fixtures.performance_data import create_performance_test_data

class TestStrategyPerformance:
    """Performance benchmarks for strategy execution"""

    @pytest.fixture
    def strategy(self):
        return UnifiedMSEStrategy()

    @pytest.fixture
    def performance_data(self):
        return create_performance_test_data(symbols=10, periods=10000)

    @pytest.mark.benchmark
    def test_signal_generation_latency(self, benchmark, strategy, performance_data):
        """Benchmark signal generation latency"""

        def generate_signal():
            return strategy.generate_signal(
                performance_data['context'],
                performance_data['state']
            )

        result = benchmark(generate_signal)

        # Assert latency requirements
        assert benchmark.stats['mean'] < 0.01  # 10ms
        assert benchmark.stats['max'] < 0.05   # 50ms max

    def test_memory_usage(self, strategy, performance_data):
        """Test memory usage during strategy execution"""

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Execute strategy multiple times
        for _ in range(1000):
            signal = strategy.generate_signal(
                performance_data['context'],
                performance_data['state']
            )

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Assert memory requirements
        assert memory_increase < 50  # Less than 50MB increase
        assert final_memory < 200    # Total less than 200MB

    @profile
    def test_memory_profiling(self, strategy, performance_data):
        """Detailed memory profiling (run manually)"""

        for _ in range(100):
            signal = strategy.generate_signal(
                performance_data['context'],
                performance_data['state']
            )
```

## 📚 Documentation Standards

### **Code Documentation**

```python
def generate_signal(self, context: MarketContext, state: StrategyState) -> Optional[Signal]:
    """
    Generate trading signal based on market context and strategy state.

    This method implements the core MSE strategy logic using a 4-indicator system:
    1. 5-minute MACD line vs signal line comparison
    2. 15-minute MACD line vs signal line comparison
    3. 5-minute EMA9 vs EMA20 comparison
    4. 15-minute EMA9 vs EMA20 comparison

    Entry signals require ALL 4 indicators to align (bullish or bearish).
    Exit signals use 80% peak/valley MACD histogram logic.

    Args:
        context: Market data context providing access to timeframe data,
                current prices, and market status
        state: Strategy state providing access to position information
               and strategy-specific variables (e.g., MACD peak tracking)

    Returns:
        Signal object if conditions are met, None otherwise.
        Signal includes action (BUY/SELL), price, confidence, and reasoning.

    Raises:
        None. Method handles all exceptions internally and logs errors.

    Example:
        >>> context = MarketContext(data_provider, "RELIANCE")
        >>> state = StrategyState()
        >>> signal = strategy.generate_signal(context, state)
        >>> if signal:
        ...     print(f"Generated {signal.action} signal at {signal.price}")

    Note:
        This method runs identically in both backtesting and live trading
        environments. The context and state abstractions hide the underlying
        implementation differences.
    """
```

## 🔧 Development Workflow

### **Daily Development Process**

1. **Morning Setup** (5 minutes)
   ```bash
   git pull origin main
   git submodule update --remote
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

2. **Feature Development** (Development cycle)
   - Create feature branch: `git checkout -b feature/description`
   - Write failing tests first (TDD approach)
   - Implement feature code
   - Run tests: `pytest tests/`
   - Run linting: `flake8 core/ strategies/`
   - Run type checking: `mypy core/ strategies/`

3. **Code Quality Checks** (Before commit)
   ```bash
   # Format code
   black core/ strategies/ tests/
   isort core/ strategies/ tests/

   # Run full test suite
   pytest tests/ --cov=core --cov=strategies --cov-report=html

   # Performance benchmarks (if changed performance-critical code)
   pytest tests/performance/ --benchmark-only
   ```

4. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: description of change"
   git push origin feature/description
   ```

5. **Code Review Process**
   - Create pull request with description and test results
   - Request review from technical lead
   - Address review comments
   - Merge after approval

### **Debugging Guidelines**

#### **Strategy Signal Issues**
```python
# Add debugging to strategy
import logging
logger = logging.getLogger(__name__)

def generate_signal(self, context, state):
    logger.debug(f"Signal generation for {context.symbol} at {context.get_current_time()}")

    # Log intermediate calculations
    indicators_5m = self._calculate_indicators(context.get_timeframe_data('5m'))
    logger.debug(f"5m indicators: {indicators_5m}")

    # Log decision logic
    if all_bullish_conditions:
        logger.info(f"Generating BUY signal: all conditions met")
        return Signal(...)
    else:
        logger.debug("No signal: conditions not met")
        return None
```

#### **Performance Issues**
```python
# Use cProfile for performance debugging
import cProfile
import pstats

pr = cProfile.Profile()
pr.enable()

# Your code here
signal = strategy.generate_signal(context, state)

pr.disable()
stats = pstats.Stats(pr)
stats.sort_stats('cumulative').print_stats(10)
```

#### **Memory Issues**
```python
# Use tracemalloc for memory debugging
import tracemalloc

tracemalloc.start()

# Your code here
signal = strategy.generate_signal(context, state)

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage is {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage was {peak / 1024 / 1024:.1f} MB")
tracemalloc.stop()
```

## 🎯 Next Steps for Developers

### **Week 1: Environment Setup**
- [ ] Complete development environment setup
- [ ] Run through all examples in this guide
- [ ] Create your first test strategy following the patterns
- [ ] Submit a small test PR to validate workflow

### **Week 2: Core Component Understanding**
- [ ] Study the UniversalStrategy interface thoroughly
- [ ] Understand MarketContext and StrategyState abstractions
- [ ] Read through MSE strategy implementation
- [ ] Write unit tests for one component

### **Week 3: Integration Development**
- [ ] Start working on assigned component (engine, adapter, or strategy)
- [ ] Write comprehensive tests for your component
- [ ] Integrate with other developers' components
- [ ] Participate in code reviews

### **Ongoing: Quality and Performance**
- [ ] Monitor performance benchmarks for your components
- [ ] Maintain >90% test coverage
- [ ] Follow documentation standards
- [ ] Participate in validation testing

---

This developer guide provides the foundation for building high-quality, performant components for the unified trading system. Follow these patterns and standards to ensure your code integrates seamlessly with the overall architecture.