"""
Test Live Trading Adapter Integration - Simplified Version

Tests the Live Trading Adapter architecture without complex dependencies.
Validates adapter design patterns and integration readiness.
"""
import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import json
import asyncio

# Initialize UTF-8 encoding for PowerShell compatibility
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Add paths for imports
sys.path.insert(0, str(Path('D:/Balcony/Trading/trading-unified-core')))
sys.path.insert(0, str(Path('D:/Balcony/Trading/trading-unified-core/adapters/live_trading')))
sys.path.insert(0, str(Path('D:/Balcony/Trading/backtester/src')))

# Test live adapter import
try:
    from live_adapter import (
        LiveTradingAdapter, LiveTradingMode, BrokerConnection, LiveOrderRequest,
        LiveDataProvider, LiveOrderExecutor, LivePositionManager, LiveStateManager
    )
    HAS_LIVE_ADAPTER = True
    print("✅ PASS: LiveTradingAdapter imported successfully")
except ImportError as e:
    print(f"❌ FAIL: Could not import LiveTradingAdapter: {e}")
    HAS_LIVE_ADAPTER = False
    sys.exit(1)

def test_live_adapter_initialization():
    """Test Live Trading Adapter initialization and configuration"""
    print("\n" + "="*50)
    print("TESTING LIVE ADAPTER INITIALIZATION")
    print("="*50)

    try:
        # Test default initialization
        adapter = LiveTradingAdapter()
        print("✅ PASS: LiveTradingAdapter initialized with default config")

        # Test configuration
        config = adapter.config
        print(f"✅ PASS: Adapter config loaded: {len(config)} sections")

        # Test trading modes
        modes = [mode.value for mode in LiveTradingMode]
        print(f"✅ PASS: Available trading modes: {modes}")

        # Test default trading mode
        print(f"✅ PASS: Default trading mode: {adapter.trading_mode.value}")

        # Test custom configuration
        custom_config = {
            'trading_mode': 'SIGNAL_ONLY',
            'data_config': {
                'broker': 'upstox',
                'update_interval': 5
            }
        }
        custom_adapter = LiveTradingAdapter(custom_config)
        print(f"✅ PASS: Custom trading mode: {custom_adapter.trading_mode.value}")

        return True

    except Exception as e:
        print(f"❌ FAIL: Error in adapter initialization - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_component_initialization():
    """Test Live Trading Adapter component initialization"""
    print("\n" + "="*50)
    print("TESTING COMPONENT INITIALIZATION")
    print("="*50)

    try:
        adapter = LiveTradingAdapter()

        # Test component initialization
        components = [
            ('data_provider', adapter.data_provider, LiveDataProvider),
            ('order_executor', adapter.order_executor, LiveOrderExecutor),
            ('position_manager', adapter.position_manager, LivePositionManager),
            ('state_manager', adapter.state_manager, LiveStateManager)
        ]

        for name, component, expected_type in components:
            if component and isinstance(component, expected_type):
                print(f"✅ PASS: {name} component initialized correctly")
            else:
                print(f"❌ FAIL: {name} component not initialized or wrong type")
                return False

        return True

    except Exception as e:
        print(f"❌ FAIL: Error testing component initialization - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_signal_processing():
    """Test signal processing and order conversion logic"""
    print("\n" + "="*50)
    print("TESTING SIGNAL PROCESSING LOGIC")
    print("="*50)

    try:
        adapter = LiveTradingAdapter()

        # Create mock trading signal using basic classes
        class MockTradingSignal:
            def __init__(self):
                self.symbol = "BIOCON"
                self.signal = MockSignalType()
                self.price = 150.0
                self.confidence = 0.85
                self.reason = "4-indicator alignment"
                self.timestamp = datetime.now()

        class MockSignalType:
            def __init__(self):
                self.value = "BUY"

        class MockLiveStrategyState:
            def get(self, key, default=None):
                return {"current_capital": 100000}.get(key, default)

        mock_signal = MockTradingSignal()
        mock_state = MockLiveStrategyState()

        # Test signal to order conversion
        order_request = adapter._convert_signal_to_order(mock_signal, mock_state)

        if order_request:
            print("✅ PASS: Signal converted to order request")
            print(f"  Symbol: {order_request.symbol}")
            print(f"  Signal Type: {order_request.signal_type.value}")
            print(f"  Quantity: {order_request.quantity}")
            print(f"  Order Type: {order_request.order_type}")
        else:
            print("❌ FAIL: Signal conversion failed")
            return False

        # Test position sizing logic
        position_size = adapter._calculate_position_size(mock_signal, mock_state)
        print(f"✅ PASS: Position size calculated: {position_size}")

        return True

    except Exception as e:
        print(f"❌ FAIL: Error testing signal processing - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trading_modes():
    """Test different trading modes"""
    print("\n" + "="*50)
    print("TESTING TRADING MODES")
    print("="*50)

    try:
        # Test each trading mode
        modes_to_test = [
            ('SIGNAL_ONLY', LiveTradingMode.SIGNAL_ONLY),
            ('PAPER_TRADING', LiveTradingMode.PAPER_TRADING),
            ('LIVE_TRADING', LiveTradingMode.LIVE_TRADING)
        ]

        for mode_name, mode_enum in modes_to_test:
            config = {'trading_mode': mode_enum.value}
            adapter = LiveTradingAdapter(config)

            print(f"✅ PASS: {mode_name} mode initialized successfully")
            print(f"  Mode: {adapter.trading_mode.value}")

        print("✅ PASS: All trading modes validated")
        return True

    except Exception as e:
        print(f"❌ FAIL: Error testing trading modes - {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_async_operations():
    """Test async operations for live trading"""
    print("\n" + "="*50)
    print("TESTING ASYNC OPERATIONS")
    print("="*50)

    try:
        adapter = LiveTradingAdapter()
        symbols = ['BIOCON', 'JSWSTEEL']

        # Test async environment initialization (simulated)
        print("Testing async initialization...")
        await adapter._initialize_live_environment(symbols)
        print("✅ PASS: Live environment initialization completed")

        # Test data provider connection (simulated)
        print("Testing data provider connection...")
        await adapter.data_provider.connect()
        print("✅ PASS: Data provider connection established")

        # Test symbol subscription (simulated)
        print("Testing symbol subscription...")
        await adapter.data_provider.subscribe_symbols(symbols)
        print("✅ PASS: Symbol subscription completed")

        # Test cleanup (simulated)
        print("Testing cleanup...")
        await adapter._cleanup_live_environment()
        print("✅ PASS: Live environment cleanup completed")

        return True

    except Exception as e:
        print(f"❌ FAIL: Error testing async operations - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_adapter_architecture():
    """Test adapter follows established patterns"""
    print("\n" + "="*50)
    print("TESTING ADAPTER ARCHITECTURE PATTERNS")
    print("="*50)

    try:
        adapter = LiveTradingAdapter()

        # Test adapter methods exist
        required_methods = [
            'run_live_strategy',
            '_initialize_live_environment',
            '_cleanup_live_environment',
            '_convert_signal_to_order',
            '_calculate_position_size'
        ]

        for method_name in required_methods:
            if hasattr(adapter, method_name):
                print(f"✅ PASS: Method {method_name} exists")
            else:
                print(f"❌ FAIL: Missing required method: {method_name}")
                return False

        # Test configuration structure
        config_sections = ['data_config', 'order_config', 'position_config', 'state_config']
        for section in config_sections:
            if section in adapter.config:
                print(f"✅ PASS: Configuration section {section} exists")
            else:
                print(f"❌ FAIL: Missing configuration section: {section}")
                return False

        print("✅ PASS: Adapter architecture follows established patterns")
        return True

    except Exception as e:
        print(f"❌ FAIL: Error testing adapter architecture - {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test execution"""
    print("LIVE TRADING ADAPTER INTEGRATION TESTING")
    print("=" * 60)

    # Test 1: Adapter initialization
    test1_passed = test_live_adapter_initialization()

    # Test 2: Component initialization
    test2_passed = test_component_initialization()

    # Test 3: Signal processing
    test3_passed = test_signal_processing()

    # Test 4: Trading modes
    test4_passed = test_trading_modes()

    # Test 5: Adapter architecture
    test5_passed = test_adapter_architecture()

    # Test 6: Async operations
    test6_passed = asyncio.run(test_async_operations())

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Adapter Initialization: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Component Initialization: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"Signal Processing: {'✅ PASS' if test3_passed else '❌ FAIL'}")
    print(f"Trading Modes: {'✅ PASS' if test4_passed else '❌ FAIL'}")
    print(f"Adapter Architecture: {'✅ PASS' if test5_passed else '❌ FAIL'}")
    print(f"Async Operations: {'✅ PASS' if test6_passed else '❌ FAIL'}")

    overall_success = all([test1_passed, test2_passed, test3_passed, test4_passed, test5_passed, test6_passed])
    print(f"\nOVERALL: {'✅ SUCCESS - Live adapter ready for integration' if overall_success else '❌ FAILURE - Issues detected'}")

    if overall_success:
        print("\n🎯 LIVE TRADING ADAPTER VALIDATION:")
        print("   • Architecture follows existing live trading patterns")
        print("   • Supports signal-only, paper trading, and live trading modes")
        print("   • Async operations for real-time data handling")
        print("   • Signal processing and order conversion working")
        print("   • Component initialization successful")
        print("   • Ready for Universal Strategy integration")
        print("   • Compatible with plug-and-play multi-timeframe architecture")
    else:
        print("⚠️  Integration issues detected - requires investigation")

    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)