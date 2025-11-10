# Strategy Virtual Machine - Implementation Guide

## 📋 Overview

This guide provides step-by-step instructions for implementing the Strategy Virtual Machine (SVM) that unifies backtesting and live trading environments. The implementation follows a phased approach to minimize risk and ensure system stability.

## 🎯 Implementation Goals

1. **Minimal System Modification**: Preserve existing proven systems
2. **Gradual Integration**: Phased rollout with extensive validation
3. **Complete Parity**: Identical strategy behavior across environments
4. **Production Readiness**: Enterprise-grade reliability and monitoring

## 📊 System Analysis Summary

### Current Backtesting System
- **Status**: ✅ Production-ready, mature system
- **Architecture**: Modular with unified runner (`src/runners/unified_runner.py`)
- **Configuration**: Comprehensive YAML-based config system (`config/unified_config.py`)
- **Strategy Framework**: Multi-timeframe support with base classes (`src/strategies/strategy_base.py`)
- **Data Integration**: Multi-broker support (Upstox, Zerodha, Binance)

### Current Live Trading System
- **Status**: ✅ Operational
- **Location**: `../live_module_14-04/`
- **Integration Points**: To be analyzed and documented during Phase 1

## 🏗️ Phase 1: Foundation Architecture (Weeks 1-3)

### 1.1 Create Unified Interface Layer

#### 1.1.1 Strategy Virtual Machine Core
```powershell
# Create SVM core directory structure
mkdir src/svm
mkdir src/svm/core
mkdir src/svm/adapters
mkdir src/svm/interfaces
mkdir src/svm/utils
```

#### 1.1.2 Define Core Interfaces

**File: `src/svm/interfaces/strategy_interface.py`**
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import pandas as pd

class UnifiedStrategyInterface(ABC):
    """
    Unified interface for strategies running in both backtest and live environments.
    This interface abstracts away the execution environment details.
    """

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize strategy with unified configuration."""
        pass

    @abstractmethod
    def on_data(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Process incoming data and generate signals."""
        pass

    @abstractmethod
    def on_order_update(self, order_update: Dict[str, Any]) -> None:
        """Handle order execution updates."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources when strategy stops."""
        pass
```

**File: `src/svm/interfaces/data_interface.py`**
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import pandas as pd

class UnifiedDataInterface(ABC):
    """
    Unified interface for data access across backtest and live environments.
    """

    @abstractmethod
    def get_historical_data(self, ticker: str, start_date: str, end_date: str,
                          timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """Get historical data for the specified ticker and timeframes."""
        pass

    @abstractmethod
    def get_current_data(self, ticker: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """Get current/latest data for the specified ticker and timeframes."""
        pass

    @abstractmethod
    def subscribe_to_updates(self, tickers: List[str], callback) -> bool:
        """Subscribe to real-time data updates (live mode only)."""
        pass
```

**File: `src/svm/interfaces/execution_interface.py`**
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from enum import Enum

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class UnifiedExecutionInterface(ABC):
    """
    Unified interface for order execution across backtest and live environments.
    """

    @abstractmethod
    def place_order(self, order: Dict[str, Any]) -> str:
        """Place an order and return order ID."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get current status of an order."""
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, Any]:
        """Get current positions."""
        pass

    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information including balance, buying power, etc."""
        pass
```

#### 1.1.3 Create SVM Core Engine

**File: `src/svm/core/strategy_virtual_machine.py`**
```python
import logging
from typing import Dict, Any, Optional, Type
from datetime import datetime
import threading
import queue

from ..interfaces.strategy_interface import UnifiedStrategyInterface
from ..interfaces.data_interface import UnifiedDataInterface
from ..interfaces.execution_interface import UnifiedExecutionInterface

class StrategyVirtualMachine:
    """
    Core Strategy Virtual Machine that provides unified runtime for strategies.
    Handles strategy lifecycle, data flow, and execution coordination.
    """

    def __init__(self,
                 strategy: UnifiedStrategyInterface,
                 data_adapter: UnifiedDataInterface,
                 execution_adapter: UnifiedExecutionInterface,
                 config: Dict[str, Any]):
        """
        Initialize SVM with strategy and adapters.

        Args:
            strategy: Strategy implementation following UnifiedStrategyInterface
            data_adapter: Data adapter for the target environment
            execution_adapter: Execution adapter for the target environment
            config: Unified configuration dictionary
        """
        self.strategy = strategy
        self.data_adapter = data_adapter
        self.execution_adapter = execution_adapter
        self.config = config
        self.logger = logging.getLogger(f"svm.{strategy.__class__.__name__}")

        # Runtime state
        self.is_running = False
        self.is_live_mode = config.get('mode') == 'live'
        self.current_positions = {}
        self.pending_orders = {}

        # Event handling
        self.event_queue = queue.Queue()
        self.event_handlers = {}

        # Performance tracking
        self.start_time = None
        self.total_signals = 0
        self.total_trades = 0
        self.errors = []

    def initialize(self) -> bool:
        """
        Initialize the SVM and all components.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Strategy Virtual Machine")

            # Initialize strategy
            self.strategy.initialize(self.config)

            # Setup event handlers
            self._setup_event_handlers()

            # Initialize adapters
            if hasattr(self.data_adapter, 'initialize'):
                self.data_adapter.initialize(self.config)

            if hasattr(self.execution_adapter, 'initialize'):
                self.execution_adapter.initialize(self.config)

            self.logger.info("SVM initialization completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"SVM initialization failed: {e}")
            self.errors.append(f"Initialization error: {e}")
            return False

    def start(self) -> None:
        """Start the strategy virtual machine."""
        if not self.initialize():
            raise RuntimeError("Failed to initialize SVM")

        self.is_running = True
        self.start_time = datetime.now()

        self.logger.info(f"Starting SVM in {'LIVE' if self.is_live_mode else 'BACKTEST'} mode")

        if self.is_live_mode:
            self._start_live_mode()
        else:
            self._start_backtest_mode()

    def stop(self) -> None:
        """Stop the strategy virtual machine."""
        self.logger.info("Stopping Strategy Virtual Machine")
        self.is_running = False

        # Cleanup strategy
        try:
            self.strategy.cleanup()
        except Exception as e:
            self.logger.error(f"Error during strategy cleanup: {e}")

        # Generate performance report
        self._generate_performance_report()

    def _start_live_mode(self) -> None:
        """Start SVM in live trading mode."""
        # Subscribe to data updates
        tickers = self.config.get('tickers', [])
        self.data_adapter.subscribe_to_updates(tickers, self._on_data_update)

        # Start event processing loop
        self._start_event_loop()

    def _start_backtest_mode(self) -> None:
        """Start SVM in backtesting mode."""
        # Get historical data and process sequentially
        tickers = self.config.get('tickers', [])
        start_date = self.config.get('start_date')
        end_date = self.config.get('end_date')
        timeframes = self.config.get('timeframes', ['1m'])

        for ticker in tickers:
            data = self.data_adapter.get_historical_data(ticker, start_date, end_date, timeframes)
            self._process_historical_data(ticker, data)

    def _on_data_update(self, data_update: Dict[str, Any]) -> None:
        """Handle incoming data updates in live mode."""
        try:
            # Process data through strategy
            signals = self.strategy.on_data(data_update)

            # Handle any generated signals
            if signals:
                self._process_signals(signals)

        except Exception as e:
            self.logger.error(f"Error processing data update: {e}")
            self.errors.append(f"Data processing error: {e}")

    def _process_signals(self, signals: Dict[str, Any]) -> None:
        """Process signals generated by strategy."""
        for signal in signals.get('orders', []):
            try:
                order_id = self.execution_adapter.place_order(signal)
                self.pending_orders[order_id] = signal
                self.total_signals += 1

            except Exception as e:
                self.logger.error(f"Error placing order: {e}")
                self.errors.append(f"Order placement error: {e}")

    def _setup_event_handlers(self) -> None:
        """Setup event handlers for different event types."""
        self.event_handlers = {
            'data_update': self._on_data_update,
            'order_update': self._on_order_update,
            'position_update': self._on_position_update,
            'error': self._on_error
        }

    def _on_order_update(self, order_update: Dict[str, Any]) -> None:
        """Handle order execution updates."""
        try:
            self.strategy.on_order_update(order_update)

            # Update internal state
            order_id = order_update.get('order_id')
            if order_id in self.pending_orders:
                if order_update.get('status') in ['filled', 'cancelled', 'rejected']:
                    del self.pending_orders[order_id]

                if order_update.get('status') == 'filled':
                    self.total_trades += 1

        except Exception as e:
            self.logger.error(f"Error processing order update: {e}")
            self.errors.append(f"Order update error: {e}")

    def _on_position_update(self, position_update: Dict[str, Any]) -> None:
        """Handle position updates."""
        ticker = position_update.get('ticker')
        if ticker:
            self.current_positions[ticker] = position_update

    def _on_error(self, error_event: Dict[str, Any]) -> None:
        """Handle error events."""
        self.logger.error(f"Error event: {error_event}")
        self.errors.append(error_event)

    def _start_event_loop(self) -> None:
        """Start the main event processing loop for live mode."""
        def event_loop():
            while self.is_running:
                try:
                    event = self.event_queue.get(timeout=1.0)
                    event_type = event.get('type')

                    if event_type in self.event_handlers:
                        self.event_handlers[event_type](event.get('data', {}))

                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error in event loop: {e}")

        # Start event loop in separate thread
        threading.Thread(target=event_loop, daemon=True).start()

    def _process_historical_data(self, ticker: str, data: Dict[str, Any]) -> None:
        """Process historical data for backtesting."""
        # This will be implemented based on how the current backtesting system works
        # For now, this is a placeholder that shows the concept
        pass

    def _generate_performance_report(self) -> Dict[str, Any]:
        """Generate performance report for the SVM session."""
        runtime = datetime.now() - self.start_time if self.start_time else None

        report = {
            'session_start': self.start_time,
            'session_end': datetime.now(),
            'runtime_seconds': runtime.total_seconds() if runtime else 0,
            'mode': 'live' if self.is_live_mode else 'backtest',
            'total_signals': self.total_signals,
            'total_trades': self.total_trades,
            'errors_count': len(self.errors),
            'errors': self.errors[-10:],  # Last 10 errors
            'final_positions': self.current_positions.copy()
        }

        self.logger.info(f"SVM Performance Report: {report}")
        return report
```

### 1.2 Create Adapter Framework

#### 1.2.1 Backtesting Adapter

**File: `src/svm/adapters/backtest_adapter.py`**
```python
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import logging
from pathlib import Path

from ..interfaces.data_interface import UnifiedDataInterface
from ..interfaces.execution_interface import UnifiedExecutionInterface, OrderStatus

class BacktestDataAdapter(UnifiedDataInterface):
    """
    Adapter for backtesting data access using existing CSV/Parquet data pools.
    Integrates with the current backtesting system's data loading mechanisms.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_pool_dir = Path(config.get('data_pool_dir', 'data/pools'))
        self.logger = logging.getLogger('svm.backtest_data_adapter')

        # Cache for loaded data
        self._data_cache = {}

    def get_historical_data(self, ticker: str, start_date: str, end_date: str,
                          timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Get historical data using existing backtesting system's data loading.

        Returns:
            Dictionary with timeframe as key and DataFrame as value
            e.g., {'1m': df_1m, '5m': df_5m}
        """
        try:
            # Use existing data loading logic from current system
            from src.core.etl.data_loader import load_historical_data

            data = {}
            date_range = f"{start_date}_to_{end_date}"

            for timeframe in timeframes:
                df = load_historical_data(ticker, date_range, timeframe)
                if df is not None and not df.empty:
                    data[timeframe] = df
                else:
                    self.logger.warning(f"No data found for {ticker} {timeframe} {date_range}")

            return data

        except Exception as e:
            self.logger.error(f"Error loading historical data: {e}")
            return {}

    def get_current_data(self, ticker: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """
        In backtest mode, this returns the latest available historical data.
        """
        # For backtesting, "current" data is just the latest historical data
        # This would be used for real-time strategy decisions in backtest simulation
        return self._data_cache.get(ticker, {})

    def subscribe_to_updates(self, tickers: List[str], callback) -> bool:
        """
        Not applicable in backtest mode - data is processed sequentially.
        """
        self.logger.info("Data subscription not needed in backtest mode")
        return True

class BacktestExecutionAdapter(UnifiedExecutionInterface):
    """
    Adapter for backtesting execution using existing backtesting system's order simulation.
    Integrates with the current risk management and execution logic.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('svm.backtest_execution_adapter')

        # Simulation state
        self.orders = {}
        self.positions = {}
        self.account_balance = config.get('initial_capital', 1000000)
        self.order_counter = 0

        # Integration with existing risk management
        try:
            from src.core.risk.risk_manager import RiskManager
            self.risk_manager = RiskManager(config)
        except ImportError:
            self.logger.warning("Risk manager not available - using basic validation")
            self.risk_manager = None

    def place_order(self, order: Dict[str, Any]) -> str:
        """
        Simulate order placement using existing backtesting logic.
        """
        try:
            # Generate order ID
            self.order_counter += 1
            order_id = f"BT_{self.order_counter:06d}"

            # Validate order with risk manager if available
            if self.risk_manager:
                validation_result = self.risk_manager.validate_order(order)
                if not validation_result.get('approved', False):
                    raise ValueError(f"Order rejected by risk manager: {validation_result.get('reason')}")

            # Store order for simulation
            order['order_id'] = order_id
            order['status'] = OrderStatus.PENDING
            order['timestamp'] = pd.Timestamp.now()

            self.orders[order_id] = order

            # In backtest mode, orders are filled immediately or according to simulation rules
            # This would integrate with existing backtesting execution logic
            self._simulate_order_execution(order_id)

            return order_id

        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order in simulation."""
        if order_id in self.orders:
            self.orders[order_id]['status'] = OrderStatus.CANCELLED
            return True
        return False

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status from simulation."""
        return self.orders.get(order_id, {})

    def get_positions(self) -> Dict[str, Any]:
        """Get current simulated positions."""
        return self.positions.copy()

    def get_account_info(self) -> Dict[str, Any]:
        """Get simulated account information."""
        return {
            'balance': self.account_balance,
            'buying_power': self.account_balance,  # Simplified
            'positions_value': sum(pos.get('market_value', 0) for pos in self.positions.values()),
            'total_equity': self.account_balance
        }

    def _simulate_order_execution(self, order_id: str) -> None:
        """
        Simulate order execution based on backtesting rules.
        This would integrate with existing execution simulation logic.
        """
        order = self.orders[order_id]

        # Simple simulation - in reality this would use proper backtesting execution rules
        order['status'] = OrderStatus.FILLED
        order['fill_price'] = order.get('price', 0)
        order['fill_quantity'] = order.get('quantity', 0)
        order['fill_timestamp'] = pd.Timestamp.now()

        # Update positions
        ticker = order.get('ticker')
        if ticker:
            if ticker not in self.positions:
                self.positions[ticker] = {'quantity': 0, 'avg_price': 0}

            # Update position (simplified)
            current_qty = self.positions[ticker]['quantity']
            order_qty = order.get('quantity', 0)

            if order.get('side') == 'sell':
                order_qty = -order_qty

            self.positions[ticker]['quantity'] = current_qty + order_qty
```

#### 1.2.2 Live Trading Adapter

**File: `src/svm/adapters/live_adapter.py`**
```python
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import logging
from datetime import datetime, timedelta
import threading
import time

from ..interfaces.data_interface import UnifiedDataInterface
from ..interfaces.execution_interface import UnifiedExecutionInterface, OrderStatus

class LiveDataAdapter(UnifiedDataInterface):
    """
    Adapter for live trading data access using existing broker integrations.
    Integrates with the live trading system's real-time data feeds.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('svm.live_data_adapter')

        # Integration with existing live trading system
        self.broker_client = None
        self.subscriptions = {}
        self.data_callbacks = {}

        # Initialize broker connection
        self._initialize_broker_connection()

    def _initialize_broker_connection(self):
        """Initialize connection to the live trading system."""
        try:
            # This would integrate with the existing live trading system
            # Placeholder for actual implementation
            broker_type = self.config.get('broker_type', 'upstox')

            if broker_type == 'upstox':
                from live_module_14_04.brokers.upstox_client import UpstoxClient
                self.broker_client = UpstoxClient(self.config)
            elif broker_type == 'zerodha':
                from live_module_14_04.brokers.zerodha_client import ZerodhaClient
                self.broker_client = ZerodhaClient(self.config)
            else:
                raise ValueError(f"Unsupported broker type: {broker_type}")

            self.logger.info(f"Initialized {broker_type} broker connection")

        except Exception as e:
            self.logger.error(f"Failed to initialize broker connection: {e}")
            raise

    def get_historical_data(self, ticker: str, start_date: str, end_date: str,
                          timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """Get historical data from broker API."""
        try:
            data = {}

            for timeframe in timeframes:
                df = self.broker_client.get_historical_data(
                    ticker, start_date, end_date, timeframe
                )
                if df is not None and not df.empty:
                    data[timeframe] = df

            return data

        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            return {}

    def get_current_data(self, ticker: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """Get current market data from broker."""
        try:
            data = {}

            for timeframe in timeframes:
                # Get latest bars for the timeframe
                df = self.broker_client.get_current_data(ticker, timeframe)
                if df is not None and not df.empty:
                    data[timeframe] = df

            return data

        except Exception as e:
            self.logger.error(f"Error fetching current data: {e}")
            return {}

    def subscribe_to_updates(self, tickers: List[str], callback) -> bool:
        """Subscribe to real-time data updates."""
        try:
            for ticker in tickers:
                self.subscriptions[ticker] = callback
                self.broker_client.subscribe_to_ticker(ticker, self._on_data_update)

            self.logger.info(f"Subscribed to real-time data for {len(tickers)} tickers")
            return True

        except Exception as e:
            self.logger.error(f"Error subscribing to data updates: {e}")
            return False

    def _on_data_update(self, ticker: str, data: Dict[str, Any]):
        """Handle incoming real-time data updates."""
        if ticker in self.subscriptions:
            callback = self.subscriptions[ticker]
            callback({'ticker': ticker, 'data': data, 'timestamp': datetime.now()})

class LiveExecutionAdapter(UnifiedExecutionInterface):
    """
    Adapter for live trading execution using existing broker integrations.
    Integrates with the live trading system's order management.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('svm.live_execution_adapter')

        # Integration with existing live trading system
        self.broker_client = None
        self.orders = {}

        # Initialize broker connection (same as data adapter)
        self._initialize_broker_connection()

        # Start order status monitoring
        self._start_order_monitoring()

    def _initialize_broker_connection(self):
        """Initialize connection to the live trading system."""
        try:
            # This would reuse the same broker client as the data adapter
            # or create a new instance for execution
            broker_type = self.config.get('broker_type', 'upstox')

            if broker_type == 'upstox':
                from live_module_14_04.brokers.upstox_client import UpstoxClient
                self.broker_client = UpstoxClient(self.config)
            elif broker_type == 'zerodha':
                from live_module_14_04.brokers.zerodha_client import ZerodhaClient
                self.broker_client = ZerodhaClient(self.config)
            else:
                raise ValueError(f"Unsupported broker type: {broker_type}")

            self.logger.info(f"Initialized {broker_type} execution connection")

        except Exception as e:
            self.logger.error(f"Failed to initialize execution connection: {e}")
            raise

    def place_order(self, order: Dict[str, Any]) -> str:
        """Place order through broker API."""
        try:
            # Validate order format
            required_fields = ['ticker', 'side', 'quantity', 'order_type']
            for field in required_fields:
                if field not in order:
                    raise ValueError(f"Missing required field: {field}")

            # Place order through broker
            order_id = self.broker_client.place_order(order)

            # Store order for tracking
            order['order_id'] = order_id
            order['status'] = OrderStatus.PENDING
            order['timestamp'] = datetime.now()
            self.orders[order_id] = order

            self.logger.info(f"Placed order {order_id}: {order}")
            return order_id

        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        """Cancel order through broker API."""
        try:
            success = self.broker_client.cancel_order(order_id)

            if success and order_id in self.orders:
                self.orders[order_id]['status'] = OrderStatus.CANCELLED

            return success

        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status from broker."""
        try:
            # Get latest status from broker
            broker_status = self.broker_client.get_order_status(order_id)

            # Update local order record
            if order_id in self.orders:
                self.orders[order_id].update(broker_status)
                return self.orders[order_id]

            return broker_status

        except Exception as e:
            self.logger.error(f"Error getting order status for {order_id}: {e}")
            return {}

    def get_positions(self) -> Dict[str, Any]:
        """Get current positions from broker."""
        try:
            return self.broker_client.get_positions()
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {}

    def get_account_info(self) -> Dict[str, Any]:
        """Get account information from broker."""
        try:
            return self.broker_client.get_account_info()
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}

    def _start_order_monitoring(self):
        """Start background thread to monitor order status updates."""
        def monitor_orders():
            while True:
                try:
                    # Check status of pending orders
                    pending_orders = [
                        order_id for order_id, order in self.orders.items()
                        if order.get('status') == OrderStatus.PENDING
                    ]

                    for order_id in pending_orders:
                        self.get_order_status(order_id)

                    time.sleep(1)  # Check every second

                except Exception as e:
                    self.logger.error(f"Error in order monitoring: {e}")
                    time.sleep(5)  # Wait longer if there's an error

        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_orders, daemon=True)
        monitor_thread.start()
        self.logger.info("Started order monitoring thread")
```

### 1.3 Integration Points Definition

#### 1.3.1 Configuration Bridge

**File: `src/svm/utils/config_bridge.py`**
```python
from typing import Dict, Any
import logging
from pathlib import Path

class ConfigurationBridge:
    """
    Bridge between existing configuration systems and SVM unified configuration.
    Handles translation between backtesting and live trading configurations.
    """

    def __init__(self):
        self.logger = logging.getLogger('svm.config_bridge')

    def create_unified_config(self, mode: str, **kwargs) -> Dict[str, Any]:
        """
        Create unified configuration for SVM from various sources.

        Args:
            mode: 'backtest' or 'live'
            **kwargs: Additional configuration parameters

        Returns:
            Unified configuration dictionary
        """
        if mode == 'backtest':
            return self._create_backtest_config(**kwargs)
        elif mode == 'live':
            return self._create_live_config(**kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _create_backtest_config(self, **kwargs) -> Dict[str, Any]:
        """Create configuration for backtesting mode."""
        # Load existing backtesting configuration
        try:
            from config.unified_config import BacktestConfig
            backtest_config = BacktestConfig()

            # Convert to dictionary and merge with kwargs
            config_dict = backtest_config.to_dict()
            config_dict.update(kwargs)
            config_dict['mode'] = 'backtest'

            return config_dict

        except Exception as e:
            self.logger.error(f"Error creating backtest config: {e}")
            # Fallback to basic configuration
            return {
                'mode': 'backtest',
                'data_pool_dir': 'data/pools',
                'initial_capital': 1000000,
                **kwargs
            }

    def _create_live_config(self, **kwargs) -> Dict[str, Any]:
        """Create configuration for live trading mode."""
        # Load existing live trading configuration
        try:
            # This would load configuration from the live trading system
            # Placeholder for actual implementation
            live_config = {
                'mode': 'live',
                'broker_type': kwargs.get('broker_type', 'upstox'),
                'api_credentials': kwargs.get('api_credentials', {}),
                'risk_limits': kwargs.get('risk_limits', {}),
                **kwargs
            }

            return live_config

        except Exception as e:
            self.logger.error(f"Error creating live config: {e}")
            raise

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate unified configuration."""
        try:
            required_fields = ['mode']

            if config['mode'] == 'backtest':
                required_fields.extend(['data_pool_dir', 'initial_capital'])
            elif config['mode'] == 'live':
                required_fields.extend(['broker_type', 'api_credentials'])

            for field in required_fields:
                if field not in config:
                    self.logger.error(f"Missing required configuration field: {field}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating config: {e}")
            return False
```

### 1.4 Strategy Adapter Creation

#### 1.4.1 Existing Strategy Wrapper

**File: `src/svm/adapters/strategy_adapter.py`**
```python
from typing import Dict, Any, Union, Optional
import pandas as pd
import logging

from ..interfaces.strategy_interface import UnifiedStrategyInterface

class ExistingStrategyAdapter(UnifiedStrategyInterface):
    """
    Adapter that wraps existing strategy implementations to work with SVM.
    Allows gradual migration of strategies to the unified interface.
    """

    def __init__(self, strategy_class, strategy_params: Dict[str, Any] = None):
        """
        Initialize with an existing strategy class.

        Args:
            strategy_class: Existing strategy class (e.g., MSEStrategy)
            strategy_params: Parameters for the strategy
        """
        self.strategy_class = strategy_class
        self.strategy_params = strategy_params or {}
        self.strategy_instance = None
        self.logger = logging.getLogger(f'svm.strategy_adapter.{strategy_class.__name__}')

        # State tracking
        self.current_positions = {}
        self.pending_orders = {}

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the wrapped strategy."""
        try:
            # Create strategy instance
            self.strategy_instance = self.strategy_class(
                name=self.strategy_class.__name__,
                parameters=self.strategy_params
            )

            # Pass configuration to strategy if it supports it
            if hasattr(self.strategy_instance, 'set_config'):
                self.strategy_instance.set_config(config)

            self.logger.info(f"Initialized strategy adapter for {self.strategy_class.__name__}")

        except Exception as e:
            self.logger.error(f"Error initializing strategy adapter: {e}")
            raise

    def on_data(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """
        Process data through the wrapped strategy and convert signals to unified format.
        """
        try:
            if not self.strategy_instance:
                raise RuntimeError("Strategy not initialized")

            # Determine ticker and pull_date from data
            # This would need to be extracted from the data structure
            ticker = self._extract_ticker_from_data(data)
            pull_date = self._extract_date_from_data(data)

            # Execute strategy
            signals_df = self.strategy_instance.execute(data, ticker, pull_date)

            # Convert strategy output to unified signal format
            unified_signals = self._convert_signals_to_unified_format(signals_df, ticker)

            return unified_signals

        except Exception as e:
            self.logger.error(f"Error processing data: {e}")
            return {}

    def on_order_update(self, order_update: Dict[str, Any]) -> None:
        """Handle order execution updates."""
        try:
            order_id = order_update.get('order_id')
            status = order_update.get('status')

            self.logger.info(f"Order update: {order_id} -> {status}")

            # Update internal tracking
            if order_id in self.pending_orders:
                self.pending_orders[order_id].update(order_update)

                # If order is filled, update positions
                if status == 'filled':
                    self._update_positions_from_fill(order_update)

        except Exception as e:
            self.logger.error(f"Error handling order update: {e}")

    def cleanup(self) -> None:
        """Cleanup strategy resources."""
        try:
            if self.strategy_instance and hasattr(self.strategy_instance, 'cleanup'):
                self.strategy_instance.cleanup()

            self.logger.info("Strategy adapter cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def _extract_ticker_from_data(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> str:
        """Extract ticker symbol from data structure."""
        # This would need to be implemented based on how data is structured
        # Placeholder implementation
        if isinstance(data, dict):
            # Multi-timeframe data
            for timeframe_data in data.values():
                if not timeframe_data.empty and 'ticker' in timeframe_data.columns:
                    return timeframe_data['ticker'].iloc[0]
        elif isinstance(data, pd.DataFrame):
            # Single timeframe data
            if 'ticker' in data.columns:
                return data['ticker'].iloc[0]

        return 'UNKNOWN'

    def _extract_date_from_data(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> str:
        """Extract date from data structure."""
        # Placeholder implementation
        if isinstance(data, dict):
            for timeframe_data in data.values():
                if not timeframe_data.empty and 'timestamp' in timeframe_data.columns:
                    return timeframe_data['timestamp'].iloc[-1].strftime('%Y-%m-%d')
        elif isinstance(data, pd.DataFrame):
            if 'timestamp' in data.columns:
                return data['timestamp'].iloc[-1].strftime('%Y-%m-%d')

        return pd.Timestamp.now().strftime('%Y-%m-%d')

    def _convert_signals_to_unified_format(self, signals_df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """
        Convert strategy-specific signal format to unified SVM format.
        """
        unified_signals = {
            'orders': [],
            'indicators': {},
            'metadata': {
                'ticker': ticker,
                'timestamp': pd.Timestamp.now(),
                'strategy': self.strategy_class.__name__
            }
        }

        if signals_df.empty:
            return unified_signals

        try:
            # Extract entry and exit signals
            # This would need to be customized based on how each strategy outputs signals

            # Look for common signal columns
            signal_columns = ['entry_signal', 'exit_signal', 'signal', 'position']

            for idx, row in signals_df.iterrows():
                for col in signal_columns:
                    if col in row and pd.notna(row[col]) and row[col] != 0:
                        order = self._create_order_from_signal(row, col, ticker)
                        if order:
                            unified_signals['orders'].append(order)

            # Extract indicators for monitoring
            indicator_columns = [col for col in signals_df.columns
                               if col not in signal_columns and
                               col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            for col in indicator_columns:
                if col in signals_df.columns:
                    unified_signals['indicators'][col] = signals_df[col].iloc[-1]

        except Exception as e:
            self.logger.error(f"Error converting signals: {e}")

        return unified_signals

    def _create_order_from_signal(self, signal_row: pd.Series, signal_col: str, ticker: str) -> Optional[Dict[str, Any]]:
        """Create order dictionary from signal row."""
        try:
            signal_value = signal_row[signal_col]

            # Determine order side based on signal
            if signal_value > 0 or signal_col == 'entry_signal':
                side = 'buy'
            elif signal_value < 0 or signal_col == 'exit_signal':
                side = 'sell'
            else:
                return None

            # Create order dictionary
            order = {
                'ticker': ticker,
                'side': side,
                'quantity': abs(int(signal_value)) if abs(signal_value) > 1 else 100,  # Default quantity
                'order_type': 'market',  # Default to market order
                'timestamp': signal_row.get('timestamp', pd.Timestamp.now()),
                'signal_source': signal_col,
                'price': signal_row.get('close', 0)  # Reference price
            }

            return order

        except Exception as e:
            self.logger.error(f"Error creating order from signal: {e}")
            return None

    def _update_positions_from_fill(self, order_update: Dict[str, Any]) -> None:
        """Update position tracking from order fill."""
        try:
            ticker = order_update.get('ticker')
            side = order_update.get('side')
            quantity = order_update.get('fill_quantity', 0)
            price = order_update.get('fill_price', 0)

            if not ticker:
                return

            # Initialize position if not exists
            if ticker not in self.current_positions:
                self.current_positions[ticker] = {
                    'quantity': 0,
                    'avg_price': 0,
                    'market_value': 0
                }

            position = self.current_positions[ticker]

            # Update position based on order side
            if side == 'buy':
                new_quantity = position['quantity'] + quantity
                if new_quantity > 0:
                    position['avg_price'] = (
                        (position['quantity'] * position['avg_price'] + quantity * price) / new_quantity
                    )
                position['quantity'] = new_quantity
            elif side == 'sell':
                position['quantity'] = max(0, position['quantity'] - quantity)

            # Update market value (simplified)
            position['market_value'] = position['quantity'] * price

        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
```

## ✅ Phase 1 Validation and Testing

### 1.5 Create Test Framework

**File: `tests/test_svm_foundation.py`**
```python
import unittest
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import json

from src.svm.core.strategy_virtual_machine import StrategyVirtualMachine
from src.svm.adapters.backtest_adapter import BacktestDataAdapter, BacktestExecutionAdapter
from src.svm.adapters.strategy_adapter import ExistingStrategyAdapter
from src.svm.utils.config_bridge import ConfigurationBridge

class TestSVMFoundation(unittest.TestCase):
    """Test suite for SVM foundation components."""

    def setUp(self):
        """Set up test environment."""
        self.config_bridge = ConfigurationBridge()
        self.test_config = {
            'mode': 'backtest',
            'data_pool_dir': 'data/pools',
            'initial_capital': 100000,
            'tickers': ['RELIANCE'],
            'start_date': '2024-01-01',
            'end_date': '2024-01-02',
            'timeframes': ['1m']
        }

    def test_configuration_bridge(self):
        """Test configuration bridge functionality."""
        # Test backtest configuration creation
        backtest_config = self.config_bridge.create_unified_config('backtest', **self.test_config)
        self.assertEqual(backtest_config['mode'], 'backtest')
        self.assertIn('data_pool_dir', backtest_config)

        # Test configuration validation
        self.assertTrue(self.config_bridge.validate_config(backtest_config))

        # Test invalid configuration
        invalid_config = {'mode': 'invalid'}
        self.assertFalse(self.config_bridge.validate_config(invalid_config))

    def test_backtest_adapters(self):
        """Test backtesting adapter functionality."""
        # Test data adapter
        data_adapter = BacktestDataAdapter(self.test_config)
        self.assertIsNotNone(data_adapter)

        # Test execution adapter
        execution_adapter = BacktestExecutionAdapter(self.test_config)
        self.assertIsNotNone(execution_adapter)

        # Test order placement simulation
        test_order = {
            'ticker': 'RELIANCE',
            'side': 'buy',
            'quantity': 100,
            'order_type': 'market'
        }

        order_id = execution_adapter.place_order(test_order)
        self.assertIsNotNone(order_id)

        # Test order status retrieval
        order_status = execution_adapter.get_order_status(order_id)
        self.assertEqual(order_status['order_id'], order_id)

    def test_strategy_adapter(self):
        """Test strategy adapter functionality."""
        # Create mock strategy class
        class MockStrategy:
            def __init__(self, name, parameters):
                self.name = name
                self.parameters = parameters

            def execute(self, data, ticker, pull_date):
                # Return mock signals
                return pd.DataFrame({
                    'timestamp': [pd.Timestamp.now()],
                    'entry_signal': [1],
                    'close': [100.0]
                })

        # Test adapter creation
        adapter = ExistingStrategyAdapter(MockStrategy, {'param1': 'value1'})
        adapter.initialize(self.test_config)

        # Test data processing
        mock_data = pd.DataFrame({
            'timestamp': [pd.Timestamp.now()],
            'close': [100.0],
            'ticker': ['RELIANCE']
        })

        signals = adapter.on_data(mock_data)
        self.assertIn('orders', signals)
        self.assertIn('indicators', signals)
        self.assertIn('metadata', signals)

    def test_svm_initialization(self):
        """Test SVM core initialization."""
        # Create mock components
        data_adapter = BacktestDataAdapter(self.test_config)
        execution_adapter = BacktestExecutionAdapter(self.test_config)

        class MockStrategy:
            def __init__(self, name, parameters):
                pass
            def execute(self, data, ticker, pull_date):
                return pd.DataFrame()

        strategy_adapter = ExistingStrategyAdapter(MockStrategy)

        # Test SVM creation
        svm = StrategyVirtualMachine(
            strategy=strategy_adapter,
            data_adapter=data_adapter,
            execution_adapter=execution_adapter,
            config=self.test_config
        )

        self.assertIsNotNone(svm)
        self.assertEqual(svm.config['mode'], 'backtest')
        self.assertFalse(svm.is_running)

        # Test initialization
        success = svm.initialize()
        self.assertTrue(success)

if __name__ == '__main__':
    unittest.main()
```

### 1.6 Integration Testing

**File: `tests/test_integration_foundation.py`**
```python
import unittest
import pandas as pd
from datetime import datetime
import os
import tempfile

from src.svm.core.strategy_virtual_machine import StrategyVirtualMachine
from src.svm.adapters.backtest_adapter import BacktestDataAdapter, BacktestExecutionAdapter
from src.svm.adapters.strategy_adapter import ExistingStrategyAdapter
from src.svm.utils.config_bridge import ConfigurationBridge

# Import existing strategy for integration testing
from src.strategies.strategy_mse import MSEStrategy

class TestSVMIntegration(unittest.TestCase):
    """Integration tests for SVM with existing systems."""

    def setUp(self):
        """Set up integration test environment."""
        self.config_bridge = ConfigurationBridge()
        self.test_config = self.config_bridge.create_unified_config(
            'backtest',
            tickers=['RELIANCE'],
            start_date='2024-01-01',
            end_date='2024-01-02',
            timeframes=['1m'],
            initial_capital=100000
        )

    def test_mse_strategy_integration(self):
        """Test integration with existing MSE strategy."""
        # Create adapters
        data_adapter = BacktestDataAdapter(self.test_config)
        execution_adapter = BacktestExecutionAdapter(self.test_config)

        # Create strategy adapter with MSE strategy
        strategy_adapter = ExistingStrategyAdapter(
            MSEStrategy,
            {'warmup_period': 50, 'exit_threshold': 0.8}
        )

        # Create SVM
        svm = StrategyVirtualMachine(
            strategy=strategy_adapter,
            data_adapter=data_adapter,
            execution_adapter=execution_adapter,
            config=self.test_config
        )

        # Test initialization
        self.assertTrue(svm.initialize())

        # Test with mock data (in real test, would use actual data)
        mock_data = self._create_mock_ohlcv_data()

        # Process data through SVM
        signals = strategy_adapter.on_data(mock_data)

        # Validate signal structure
        self.assertIn('orders', signals)
        self.assertIn('metadata', signals)
        self.assertEqual(signals['metadata']['strategy'], 'MSEStrategy')

    def test_config_bridge_integration(self):
        """Test configuration bridge with existing config system."""
        # Test loading existing backtest configuration
        try:
            from config.unified_config import BacktestConfig

            # Create existing config
            existing_config = BacktestConfig()

            # Convert through bridge
            unified_config = self.config_bridge.create_unified_config(
                'backtest',
                strategy_name='mse',
                risk_profile='conservative'
            )

            # Validate integration
            self.assertEqual(unified_config['mode'], 'backtest')
            self.assertIn('data_pool_dir', unified_config)

        except ImportError:
            self.skipTest("Existing config system not available")

    def test_data_adapter_integration(self):
        """Test data adapter integration with existing data loading."""
        data_adapter = BacktestDataAdapter(self.test_config)

        # Test historical data loading (would need actual data files)
        try:
            historical_data = data_adapter.get_historical_data(
                'RELIANCE', '2024-01-01', '2024-01-02', ['1m']
            )

            # Validate data structure
            self.assertIsInstance(historical_data, dict)

            if historical_data:  # Only test if data is available
                self.assertIn('1m', historical_data)
                self.assertIsInstance(historical_data['1m'], pd.DataFrame)

        except Exception as e:
            self.skipTest(f"Data loading not available: {e}")

    def test_execution_adapter_integration(self):
        """Test execution adapter integration with existing risk management."""
        execution_adapter = BacktestExecutionAdapter(self.test_config)

        # Test order placement with risk validation
        test_order = {
            'ticker': 'RELIANCE',
            'side': 'buy',
            'quantity': 100,
            'order_type': 'market',
            'price': 2500.0
        }

        # Place order
        order_id = execution_adapter.place_order(test_order)
        self.assertIsNotNone(order_id)

        # Check order status
        order_status = execution_adapter.get_order_status(order_id)
        self.assertEqual(order_status['ticker'], 'RELIANCE')

        # Test account info
        account_info = execution_adapter.get_account_info()
        self.assertIn('balance', account_info)
        self.assertIn('buying_power', account_info)

    def _create_mock_ohlcv_data(self) -> pd.DataFrame:
        """Create mock OHLCV data for testing."""
        dates = pd.date_range('2024-01-01 09:15', periods=100, freq='1min')

        # Generate realistic price data
        base_price = 2500.0
        prices = []
        current_price = base_price

        for _ in range(100):
            # Random walk with small variations
            change = (pd.np.random.random() - 0.5) * 2  # -1 to 1
            current_price += change
            prices.append(current_price)

        data = []
        for i, (timestamp, close) in enumerate(zip(dates, prices)):
            # Create OHLC from close price
            high = close + pd.np.random.random() * 2
            low = close - pd.np.random.random() * 2
            open_price = prices[i-1] if i > 0 else close
            volume = pd.np.random.randint(1000, 10000)

            data.append({
                'timestamp': timestamp,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume,
                'ticker': 'RELIANCE'
            })

        return pd.DataFrame(data)

if __name__ == '__main__':
    unittest.main()
```

## 📊 Phase 1 Deliverables and Checkpoints

### 1.7 Validation Checklist

**File: `docs/phase1_validation_checklist.md`**
```markdown
# Phase 1 Validation Checklist

## Foundation Components ✓

- [ ] **Core Interfaces Implemented**
  - [ ] UnifiedStrategyInterface defined and documented
  - [ ] UnifiedDataInterface defined and documented
  - [ ] UnifiedExecutionInterface defined and documented
  - [ ] All interfaces include comprehensive docstrings

- [ ] **Strategy Virtual Machine Core**
  - [ ] SVM class implemented with lifecycle management
  - [ ] Event handling system implemented
  - [ ] Performance tracking implemented
  - [ ] Error handling and logging implemented

- [ ] **Adapter Framework**
  - [ ] BacktestDataAdapter implemented and tested
  - [ ] BacktestExecutionAdapter implemented and tested
  - [ ] Strategy adapter for existing strategies implemented
  - [ ] Configuration bridge implemented and tested

## Integration Points ✓

- [ ] **Existing System Integration**
  - [ ] Backtesting system integration points identified
  - [ ] Live trading system integration points identified
  - [ ] Data loading integration working
  - [ ] Risk management integration working

- [ ] **Configuration Management**
  - [ ] Unified configuration format defined
  - [ ] Configuration bridge tested with existing configs
  - [ ] Validation rules implemented
  - [ ] Migration path from existing configs defined

## Testing ✓

- [ ] **Unit Tests**
  - [ ] All core components have unit tests
  - [ ] Test coverage > 80%
  - [ ] All tests passing
  - [ ] Mock objects for external dependencies

- [ ] **Integration Tests**
  - [ ] SVM initialization tested
  - [ ] Strategy adapter integration tested
  - [ ] Data flow integration tested
  - [ ] Configuration integration tested

## Documentation ✓

- [ ] **Architecture Documentation**
  - [ ] Interface specifications documented
  - [ ] Component interaction diagrams created
  - [ ] Data flow documentation complete
  - [ ] Error handling documentation complete

- [ ] **Development Documentation**
  - [ ] Code standards defined
  - [ ] Testing guidelines created
  - [ ] Integration procedures documented
  - [ ] Troubleshooting guide started

## Performance Validation ✓

- [ ] **Baseline Measurements**
  - [ ] Existing system performance measured
  - [ ] SVM overhead measured and documented
  - [ ] Memory usage profiled
  - [ ] Latency measurements taken

- [ ] **Acceptance Criteria**
  - [ ] Latency overhead < 5ms confirmed
  - [ ] Memory overhead < 10% confirmed
  - [ ] Error rate < 0.1% confirmed
  - [ ] Configuration migration 100% successful

## Risk Assessment ✓

- [ ] **Technical Risks**
  - [ ] Integration complexity assessed
  - [ ] Performance impact evaluated
  - [ ] Data consistency risks identified
  - [ ] Error handling gaps identified

- [ ] **Mitigation Strategies**
  - [ ] Rollback procedures defined
  - [ ] Monitoring strategies implemented
  - [ ] Testing strategies enhanced
  - [ ] Documentation completeness verified

## Go/No-Go Decision Criteria

### Go Criteria (All must be met):
1. All unit and integration tests passing
2. Performance overhead within acceptable limits (<5ms latency, <10% memory)
3. Successful integration with at least one existing strategy
4. Configuration bridge working with existing systems
5. Comprehensive error handling implemented
6. Documentation complete and reviewed

### No-Go Criteria (Any one triggers re-work):
1. Performance overhead exceeds limits
2. Integration failures with existing systems
3. Test coverage below 80%
4. Critical error handling gaps
5. Configuration migration failures
6. Unresolved architectural conflicts

## Phase 1 Success Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Test Coverage | >80% | __%  | ⏳ |
| Latency Overhead | <5ms | __ms | ⏳ |
| Memory Overhead | <10% | __%  | ⏳ |
| Integration Success | 100% | __%  | ⏳ |
| Config Migration | 100% | __%  | ⏳ |
| Error Rate | <0.1% | __%  | ⏳ |

## Sign-off

- [ ] **Technical Lead Approval**
  - [ ] Architecture review completed
  - [ ] Code review completed
  - [ ] Test review completed
  - [ ] Performance review completed

- [ ] **QA Approval**
  - [ ] Test plan executed
  - [ ] Integration testing completed
  - [ ] Performance testing completed
  - [ ] Documentation review completed

- [ ] **Project Manager Approval**
  - [ ] Milestone deliverables met
  - [ ] Timeline adherence confirmed
  - [ ] Resource utilization acceptable
  - [ ] Ready for Phase 2

**Date:** _______________
**Approved by:** _______________
**Next Phase Start Date:** _______________
```

---

**⚠️ Important Notes for Phase 1:**

1. **Minimal System Modification**: Phase 1 focuses on creating the abstraction layer without modifying existing systems
2. **Gradual Integration**: Start with simple strategies and expand complexity
3. **Comprehensive Testing**: Every component must have unit and integration tests
4. **Performance Monitoring**: Continuously monitor performance impact
5. **Documentation First**: Document interfaces before implementation
6. **Risk Mitigation**: Have rollback plans for every integration step

**Continue to Phase 2 only after all Phase 1 validation criteria are met.**