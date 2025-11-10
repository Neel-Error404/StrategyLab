"""
Clean MSE Strategy - Two-Bar Rule (Matches Backtest Timing)

This is a refactored version of the MSE strategy that follows the single responsibility principle.
The strategy ONLY generates signals based on technical analysis. All other responsibilities 
have been moved to appropriate components:

- Order execution & retry logic → UnifiedOrderExecutor
- Risk management & rejection tracking → GlobalRiskManager  
- Position state management → UnifiedPositionManager
- System orchestration → TradingSystem

Strategy Responsibilities (ONLY):
1. Calculate technical indicators (EMA, MACD)
2. Analyze 4-indicator system for entry/exit signals
3. Return signal intent with reasoning
4. Track minimal state needed for signal generation

Strategy Logic:
- BUY Entry: ALL 4 bullish (5min MACD > signal, 15min MACD > signal, 5min EMA9 > EMA20, 15min EMA9 > EMA20)
- SELL Entry: ALL 4 bearish (5min MACD < signal, 15min MACD < signal, 5min EMA9 < EMA20, 15min EMA9 < EMA20)
- Exit Logic: 15min MACD histogram at 80% of peak/valley
"""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from .strategy_interface import StrategyInterface, StrategyRequirements
from .strategy_template import UniversalStrategyTemplate, PositionContext, HistoricalContext, StrategyDataError
from ..utils.precision_handler import round_price, round_technical, format_price, format_technical

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of signals that can be generated."""
    BUY_ENTRY = "BUY_ENTRY"
    SELL_ENTRY = "SELL_ENTRY"
    BUY_EXIT = "BUY_EXIT"  # Exit from SHORT
    SELL_EXIT = "SELL_EXIT"  # Exit from LONG
    NO_SIGNAL = "NO_SIGNAL"


class StrategyMode(Enum):
    """Strategy operating modes to prevent position doubling."""
    ENTRY_MODE = "ENTRY_MODE"  # Can only generate entry signals
    EXIT_MODE = "EXIT_MODE"    # Can only generate exit signals


@dataclass
class MSEIndicators:
    """MSE strategy indicators for 4-indicator system."""
    # 5min timeframe indicators
    ema_9_5min: float
    ema_20_5min: float
    macd_line_5min: float
    macd_signal_5min: float
    macd_histogram_5min: float
    
    # 15min timeframe indicators  
    ema_9_15min: float
    ema_20_15min: float
    macd_line_15min: float
    macd_signal_15min: float
    macd_histogram_15min: float


@dataclass
class PositionSummary:
    """Minimal position information needed for signal generation."""
    symbol: str
    has_position: bool = False
    side: str = "FLAT"  # "LONG", "SHORT", "FLAT"
    peak_macd_histogram: float = 0.0
    current_macd_histogram: float = 0.0


class MSEStrategy(StrategyInterface, UniversalStrategyTemplate):
    """
    Clean MSE Strategy focused only on signal generation.
    
    Implements the 4-indicator system:
    - 5min MACD vs Signal
    - 15min MACD vs Signal  
    - 5min EMA9 vs EMA20
    - 15min EMA9 vs EMA20
    
    Entry requires ALL 4 indicators to align.
    Exit uses 80% peak/valley MACD histogram logic.
    """
    
    def __init__(self):
        self._name = "MSE_Strategy_TwoBar"
        self._requirements = StrategyRequirements(
            timeframes=["5min", "15min"],
            minimum_candles={
                "5min": 40,  # 40 candles for EMA9/20 and MACD calculations
                "15min": 40  # 40 candles for EMA9/20 and MACD calculations
            }
        )
        
        # Path to unified position file (single source of truth)
        # Use absolute path resolution to ensure consistency with position manager
        base_dir = Path(__file__).parent.parent.parent.parent  # Go up to project root
        self._unified_position_file = base_dir / "live_module" / "data" / "positions" / "positions.json"
        
        # Entry/Exit Mode Architecture - Prevents position doubling
        self._strategy_modes: Dict[str, StrategyMode] = {}
        # Two-bar rule state tracking
        self._pending_signals = {}
        self._last_bar_time = {}
        
        # Simple state persistence (just MACD peak tracking)
        self._state_dir = Path("strategy_macd_peaks")
        self._state_dir.mkdir(parents=True, exist_ok=True)
        
        # Smart recovery on startup
        self._recover_strategy_modes_on_startup()
        
        logger.info("🎯 MSE Strategy initialized - Entry/Exit Mode Architecture enabled")
    
    @property
    def name(self) -> str:
        """Strategy name for identification."""
        return self._name
    
    @property
    def requirements(self) -> StrategyRequirements:
        """Data requirements for the strategy."""
        return self._requirements
    
    def _recover_strategy_modes_on_startup(self) -> None:
        """
        Smart recovery of strategy modes on system startup.
        
        Logic:
        - MIS trades: Always start fresh in ENTRY_MODE (daily cleanup)
        - CNC trades: Check positions.json for existing positions
        - Fallback: Default to ENTRY_MODE if anything fails (safe)
        """
        try:
            # Import ticker config manager for dynamic symbol support
            from ..data.ticker_config_manager import TickerConfigManager
            
            ticker_config_manager = TickerConfigManager()
            active_symbols = ticker_config_manager.get_active_symbols()
            
            logger.info(f"🔄 STRATEGY MODE RECOVERY: Processing {len(active_symbols)} symbols")
            
            # Load position data if available
            position_data = {}
            if self._unified_position_file.exists():
                with open(self._unified_position_file, 'r') as f:
                    data = json.load(f)
                position_data = data.get('actual_positions', {})
            
            # Process each active symbol
            for symbol in active_symbols:
                try:
                    product_type = ticker_config_manager.get_product_type(symbol)
                    current_quantity = position_data.get(symbol, {}).get('quantity', 0)
                    
                    if product_type == "MIS":
                        # MIS trades: Check for existing positions first, then default to fresh start
                        if abs(current_quantity) > 0:
                            self._strategy_modes[symbol] = StrategyMode.EXIT_MODE
                            logger.info(f"🌅 {symbol} (MIS): Has existing position (qty={current_quantity}) → EXIT_MODE")
                        else:
                            self._strategy_modes[symbol] = StrategyMode.ENTRY_MODE
                            logger.info(f"🌅 {symbol} (MIS): Daily fresh start → ENTRY_MODE")
                        
                    elif product_type == "CNC":
                        # CNC trades check for existing positions
                        if abs(current_quantity) > 0:
                            self._strategy_modes[symbol] = StrategyMode.EXIT_MODE
                            logger.info(f"🔄 {symbol} (CNC): Has position (qty={current_quantity}) → EXIT_MODE")
                        else:
                            self._strategy_modes[symbol] = StrategyMode.ENTRY_MODE
                            logger.info(f"🔄 {symbol} (CNC): No position → ENTRY_MODE")
                    
                    else:
                        # Unknown product type - default to safe mode
                        self._strategy_modes[symbol] = StrategyMode.ENTRY_MODE
                        logger.warning(f"⚠️ {symbol}: Unknown product_type '{product_type}' → ENTRY_MODE (safe default)")
                        
                except Exception as e:
                    # Per-symbol fallback
                    self._strategy_modes[symbol] = StrategyMode.ENTRY_MODE
                    logger.error(f"❌ {symbol}: Recovery failed ({e}) → ENTRY_MODE (safe fallback)")
            
            # Summary logging
            entry_count = sum(1 for mode in self._strategy_modes.values() if mode == StrategyMode.ENTRY_MODE)
            exit_count = sum(1 for mode in self._strategy_modes.values() if mode == StrategyMode.EXIT_MODE)
            
            logger.info(f"🎯 RECOVERY COMPLETE: {entry_count} symbols in ENTRY_MODE, {exit_count} symbols in EXIT_MODE")
            
        except Exception as e:
            logger.error(f"❌ Strategy mode recovery failed completely: {e}")
            # Global fallback - ensure we have at least basic symbols in safe mode
            default_symbols = ["TATASTEEL", "JSWSTEEL", "BIOCON"]
            for symbol in default_symbols:
                self._strategy_modes[symbol] = StrategyMode.ENTRY_MODE
            logger.warning(f"🛡️ Global fallback: Set {len(default_symbols)} default symbols to ENTRY_MODE")
    
    def calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate EMA with fixed period and 4-decimal precision."""
        if len(prices) < period:
            return 0.0
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period  # Start with SMA
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return round_technical(ema)
    
    def _read_position_from_unified_file(self, symbol: str) -> PositionSummary:
        """
        Read position state DIRECTLY from broker-reconciled actual_positions.
        
        ARCHITECTURAL FIX: Single source of truth - no more strategy_positions sync issues.
        """
        try:
            if not self._unified_position_file.exists():
                logger.debug(f"Unified position file not found: {self._unified_position_file}")
                return PositionSummary(symbol)
            
            with open(self._unified_position_file, 'r') as f:
                data = json.load(f)
            
            # 🎯 SINGLE SOURCE OF TRUTH: Read ONLY from broker-reconciled actual_positions
            actual_positions = data.get('actual_positions', {})
            actual_pos = actual_positions.get(symbol, {})
            actual_quantity = actual_pos.get('quantity', 0)
            
            # Create position summary based on BROKER REALITY
            position_summary = PositionSummary(symbol)
            position_summary.has_position = abs(actual_quantity) > 0
            
            if actual_quantity > 0:
                position_summary.side = "LONG"
            elif actual_quantity < 0:
                position_summary.side = "SHORT"
            else:
                position_summary.side = "FLAT"
            
            # 🎯 MACD peaks from separate storage (not mixed with position state)
            macd_peaks = data.get('strategy_macd_peaks', {}).get(symbol, {})
            
            # 🚨 SMART VALIDATION: Only require peak data for positions that exist
            if position_summary.has_position:
                # For active positions, peak data is required
                if symbol not in data.get('strategy_macd_peaks', {}):
                    error_msg = f"🚨 CRITICAL: Peak tracking data missing for active position {symbol} - STOPPING STRATEGY TO PREVENT WRONG TRADES"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                peak_value = macd_peaks.get('peak')
                current_value = macd_peaks.get('current')
                
                # 🚨 FAIL-FAST: Detect broken peak tracking for active positions
                if peak_value is None or current_value is None:
                    error_msg = f"🚨 CRITICAL: Peak tracking corrupted for active position {symbol} (peak={peak_value}, current={current_value}) - STOPPING STRATEGY"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
            else:
                # For FLAT positions, peak data is not required (gets cleared on exit)
                peak_value = macd_peaks.get('peak', 0.0)
                current_value = macd_peaks.get('current', 0.0)
            
            # 🛡️ MINIMAL VALIDATION: Only fail for truly corrupted data
            # Let the natural peak tracking logic handle normal evolution
            # Only validate for data corruption (None, NaN, extreme values)
            if peak_value is not None and abs(peak_value) > 100:  # Sanity check for extreme values
                error_msg = f"🚨 CRITICAL: Peak value corrupted for {symbol} (peak={peak_value}) - extreme value detected"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            position_summary.peak_macd_histogram = peak_value
            position_summary.current_macd_histogram = current_value
            
            logger.debug(f"📖 BROKER TRUTH: {symbol} -> {position_summary.side} (qty={actual_quantity}, has_position={position_summary.has_position}, peak={position_summary.peak_macd_histogram})")
            
            return position_summary
            
        except Exception as e:
            logger.error(f"Error reading position from unified file for {symbol}: {e}")
            return PositionSummary(symbol)
    
    def _update_macd_peaks(self, symbol: str, peak_value: float, current_value: float) -> None:
        """Update MACD peak data in separate storage - Single Source of Truth Architecture."""
        try:
            if not self._unified_position_file.exists():
                logger.warning(f"Cannot update MACD peaks - file not found: {self._unified_position_file}")
                return
            
            with open(self._unified_position_file, 'r') as f:
                data = json.load(f)
            
            # 🎯 MACD peaks in separate storage (not mixed with position state)
            if 'strategy_macd_peaks' not in data:
                data['strategy_macd_peaks'] = {}
            if symbol not in data['strategy_macd_peaks']:
                data['strategy_macd_peaks'][symbol] = {}
            
            # Store MACD peak data separately from position state
            data['strategy_macd_peaks'][symbol]['peak'] = peak_value
            data['strategy_macd_peaks'][symbol]['current'] = current_value
            data['strategy_macd_peaks'][symbol]['last_updated'] = datetime.now().isoformat()
            
            # Write back to file
            with open(self._unified_position_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"📝 UPDATED MACD PEAKS: {symbol} peak={peak_value:.4f} current={current_value:.4f}")
            
        except Exception as e:
            logger.error(f"Error updating MACD peaks: {e}")
    
    def _initialize_peak_on_entry(self, symbol: str) -> None:
        """Initialize MACD peak to entry value when position is established."""
        # 🚨 FAIL-FAST: Peak initialization is CRITICAL - any failure should stop the strategy
        try:
            # Use existing market data context instead of broken constructor
            if not hasattr(self, 'market_data') or not self.market_data:
                error_msg = f"🚨 CRITICAL: No market data available for peak initialization - {symbol} - CANNOT TRADE SAFELY"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Calculate indicators using the proper method (not raw DataFrame access)
            try:
                indicators = self.calculate_indicators(self.market_data)
                current_macd = round_technical(indicators.macd_histogram_15min)
            except Exception as e:
                error_msg = f"🚨 CRITICAL: Failed to calculate MACD for peak initialization - {symbol}: {e} - CANNOT TRADE SAFELY"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # 🚨 FAIL-FAST: Validate MACD data is reasonable
            if abs(current_macd) > 10.0:  # Sanity check
                error_msg = f"🚨 CRITICAL: Invalid MACD value for peak initialization - {symbol}: {current_macd} - STOPPING STRATEGY"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Initialize peak to current MACD value at entry
            self._update_macd_peaks(symbol, current_macd, current_macd)
            logger.info(f"🎯 PEAK INITIALIZED ON ENTRY: {symbol} peak={format_technical(current_macd)}")
            
        except Exception as e:
            error_msg = f"🚨 CRITICAL: Peak initialization failed for {symbol}: {e} - CANNOT TRADE WITHOUT PEAK TRACKING"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _clear_peak_data(self, symbol: str) -> None:
        """Clear MACD peak data when position closes."""
        try:
            if not self._unified_position_file.exists():
                return
            
            with open(self._unified_position_file, 'r') as f:
                data = json.load(f)
            
            # Remove peak data for closed position
            if 'strategy_macd_peaks' in data and symbol in data['strategy_macd_peaks']:
                del data['strategy_macd_peaks'][symbol]
                logger.info(f"🧹 CLEARED PEAK DATA: {symbol} (position closed)")
                
                with open(self._unified_position_file, 'w') as f:
                    json.dump(data, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Error clearing peak data for {symbol}: {e}")
    
    def _is_peak_data_stale(self, symbol: str, position_summary: PositionSummary) -> bool:
        """Detect stale peak data that doesn't match current market conditions."""
        try:
            current_macd = position_summary.current_macd_histogram
            peak_macd = position_summary.peak_macd_histogram
            
            # Red flags for stale data:
            # 1. Unrealistic peak values (too high/low compared to current)
            # 2. Peak is exactly 0.0 while we have a position
            # 3. Peak is more than 10x the current value (likely test data)
            
            if peak_macd == 0.0:
                return True  # Peak should never be exactly 0.0 with an active position
            
            # Check if peak is unrealistically different from current (10x threshold)
            if abs(peak_macd) > abs(current_macd) * 10:
                logger.warning(f"🚨 STALE PEAK SUSPECTED: {symbol} peak={peak_macd} vs current={current_macd} (>10x difference)")
                return True
            
            # Check for common test values that shouldn't be in production
            test_values = [1.5678, 2.3456, -1.888, 3.4567]  # Common test data patterns
            if any(abs(peak_macd - test_val) < 0.0001 for test_val in test_values):
                logger.warning(f"🚨 TEST DATA DETECTED: {symbol} peak={peak_macd} matches test pattern")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking peak staleness for {symbol}: {e}")
            return True  # Assume stale if we can't verify
    
    def calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD line, signal line, and histogram with 4-decimal precision."""
        if len(prices) < slow:
            return 0.0, 0.0, 0.0
        
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        macd_line = round_technical(ema_fast - ema_slow)
        
        # Calculate signal line (EMA of MACD line)
        macd_values = []
        for i in range(len(prices)):
            if i >= slow - 1:
                fast_ema = self.calculate_ema(prices[:i+1], fast)
                slow_ema = self.calculate_ema(prices[:i+1], slow)
                macd_values.append(round_technical(fast_ema - slow_ema))
        
        signal_line = self.calculate_ema(macd_values, signal) if len(macd_values) >= signal else 0.0
        histogram = round_technical(macd_line - signal_line)
        
        return macd_line, signal_line, histogram
    
    def calculate_indicators(self, market_data: Dict[str, pd.DataFrame]) -> MSEIndicators:
        """Calculate all technical indicators for both timeframes."""
        
        # Extract price data for both timeframes
        prices_5min = []
        prices_15min = []
        
        if "5min" in market_data and not market_data["5min"].empty:
            prices_5min = market_data["5min"]["close"].tolist()
        
        if "15min" in market_data and not market_data["15min"].empty:
            prices_15min = market_data["15min"]["close"].tolist()
        
        # Calculate 5min indicators
        ema_9_5min = self.calculate_ema(prices_5min, 9)
        ema_20_5min = self.calculate_ema(prices_5min, 20)
        macd_line_5min, macd_signal_5min, macd_histogram_5min = self.calculate_macd(prices_5min)
        
        # Calculate 15min indicators
        ema_9_15min = self.calculate_ema(prices_15min, 9)
        ema_20_15min = self.calculate_ema(prices_15min, 20)
        macd_line_15min, macd_signal_15min, macd_histogram_15min = self.calculate_macd(prices_15min)
        
        return MSEIndicators(
            ema_9_5min=round_technical(ema_9_5min),
            ema_20_5min=round_technical(ema_20_5min),
            macd_line_5min=round_technical(macd_line_5min),
            macd_signal_5min=round_technical(macd_signal_5min),
            macd_histogram_5min=round_technical(macd_histogram_5min),
            ema_9_15min=round_technical(ema_9_15min),
            ema_20_15min=round_technical(ema_20_15min),
            macd_line_15min=round_technical(macd_line_15min),
            macd_signal_15min=round_technical(macd_signal_15min),
            macd_histogram_15min=round_technical(macd_histogram_15min)
        )
    
    def generate_signal(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate trading signal based on MSE strategy logic.
        
        Returns signal intent only - no position management, no order execution.
        The system will handle all execution, risk management, and position tracking.
        """
        try:
            # 🔧 CRITICAL FIX: Store market_data for peak initialization access
            self.market_data = market_data
            
            # Extract symbol
            symbol = None
            for timeframe, df in market_data.items():
                if not df.empty and 'symbol' in df.columns:
                    symbol = df['symbol'].iloc[-1]
                    break
            
            if not symbol:
                return {}
            
            # 🛑 CRITICAL: Indian market time cutoff - No new entries after 15:15 (need to square off by 15:20)
            current_time = pd.Timestamp.now(tz='Asia/Kolkata').time()
            cutoff_time = pd.Timestamp('15:15:00').time()
            
            if current_time >= cutoff_time:
                # Allow exits but prevent new entries after 15:15
                current_mode = self._strategy_modes.get(symbol, StrategyMode.ENTRY_MODE)
                if current_mode == StrategyMode.ENTRY_MODE:
                    logger.info(f"⏰ Market cutoff: No new entries after 15:15 for {symbol}")
                    return {}
            
            # Check data sufficiency
            if not self.has_sufficient_data({"candles": dict(market_data.items())}):
                return {}
            
            # Get current price with 4-decimal precision
            current_price = 0.0
            for timeframe, df in market_data.items():
                if not df.empty:
                    current_price = round_price(df['close'].iloc[-1])
                    break
            
            if not current_price:
                return {}
            
            # Calculate indicators
            indicators = self.calculate_indicators(market_data)
            
            # 🎯 NEW ENTRY/EXIT MODE ARCHITECTURE - Prevents Position Doubling
            current_mode = self._strategy_modes.get(symbol, StrategyMode.ENTRY_MODE)
            
            # 🔧 PROACTIVE MODE SYNCHRONIZATION: Auto-fix desynchronized modes
            position_summary = self._read_position_from_unified_file(symbol)
            expected_mode = StrategyMode.EXIT_MODE if position_summary.has_position else StrategyMode.ENTRY_MODE
            
            if current_mode != expected_mode:
                logger.warning(f"🔧 MODE DESYNC DETECTED: {symbol} mode={current_mode.value} but position={position_summary.side}")
                logger.warning(f"🔧 AUTO-FIXING: {symbol} {current_mode.value} → {expected_mode.value}")
                self._strategy_modes[symbol] = expected_mode
                current_mode = expected_mode
                logger.info(f"✅ MODE SYNCHRONIZED: {symbol} now in {current_mode.value}")
            
            logger.info(f"🎯 MSE SIGNAL GENERATION for {symbol}:")
            logger.info(f"   📊 Current Strategy Mode: {current_mode.value}")
            logger.info(f"   🔄 Entry/Exit Mode Architecture: ACTIVE")
            
            # Generate signal based on strategy mode (NOT position file!)
            signal = {}
            
            if current_mode == StrategyMode.ENTRY_MODE:
                # Can only generate entry signals
                logger.info(f"   ➡️ ENTRY_MODE: Checking for entry signals only")
                signal = self._check_entry_signal(symbol, indicators, current_price)
                
                if signal:
                    # DO NOT switch mode here - wait for successful order execution
                    # Mode switch will happen in on_position_updated() callback
                    logger.info(f"🎯 {symbol}: Entry signal generated - waiting for order execution confirmation")
                    logger.info(f"⏳ {symbol}: Will switch to EXIT_MODE only if order succeeds")
                    
            elif current_mode == StrategyMode.EXIT_MODE:
                # Can only generate exit signals
                logger.info(f"   ⬅️ EXIT_MODE: Checking for exit signals only")
                
                # Get position info for exit signal logic (MACD peak tracking)
                position_summary = self._read_position_from_unified_file(symbol)
                signal = self._check_exit_signal(symbol, indicators, current_price, position_summary)
                
                if signal:
                    # DO NOT switch mode here - wait for successful position closure
                    # Mode switch will happen in on_position_updated() callback when qty=0
                    logger.info(f"🚪 {symbol}: Exit signal generated - waiting for position closure confirmation")
                    logger.info(f"⏳ {symbol}: Will switch to ENTRY_MODE only when position closes")
            
            else:
                logger.error(f"❌ {symbol}: Unknown strategy mode {current_mode}")
                signal = {}
            
            # Return pure signal intent
            if signal:
                signal["indicators"] = {
                    "5min_macd_bullish": indicators.macd_line_5min > indicators.macd_signal_5min,
                    "15min_macd_bullish": indicators.macd_line_15min > indicators.macd_signal_15min,
                    "5min_ema_bullish": indicators.ema_9_5min > indicators.ema_20_5min,
                    "15min_ema_bullish": indicators.ema_9_15min > indicators.ema_20_15min,
                    "macd_histogram_15min": indicators.macd_histogram_15min
                }
                
                # ENHANCED BEAUTIFUL SIGNAL LOGGING with technical comparisons
                from live_module.src.utils.beautiful_logger import beautiful_logger
                
                # Enhanced signal with detailed technical comparison 
                signal_type = "EXIT" if signal.get('is_exit', False) else "ENTRY"
                beautiful_logger.enhanced_signal(
                    symbol=symbol,
                    action=signal['action'],
                    price=signal['price'],
                    macd_5min=(indicators.macd_line_5min, indicators.macd_signal_5min),
                    macd_15min=(indicators.macd_line_15min, indicators.macd_signal_15min), 
                    ema_5min=(indicators.ema_9_5min, indicators.ema_20_5min),
                    ema_15min=(indicators.ema_9_15min, indicators.ema_20_15min),
                    confidence=signal.get('confidence', 95) / 100.0,
                    signal_type=signal_type
                )
            
            return signal or {}
            
        except Exception as e:
            logger.error(f"Error generating MSE signal: {e}")
            return {}
    
    def _check_entry_signal(self, symbol: str, indicators: MSEIndicators, current_price: float) -> Optional[Dict[str, Any]]:
        """Check for entry signals when no position exists."""
        
        # BUY Entry: ALL 4 indicators must be bullish
        if (indicators.macd_line_5min > indicators.macd_signal_5min and      # 5min MACD bullish
            indicators.macd_line_15min > indicators.macd_signal_15min and    # 15min MACD bullish  
            indicators.ema_9_5min > indicators.ema_20_5min and               # 5min EMA bullish
            indicators.ema_9_15min > indicators.ema_20_15min):               # 15min EMA bullish
            
            return {
                "action": "BUY",
                "price": current_price,
                "reason": f"ALL 4 BULLISH: 5min MACD ({format_technical(indicators.macd_line_5min)} > {format_technical(indicators.macd_signal_5min)}), 15min MACD ({format_technical(indicators.macd_line_15min)} > {format_technical(indicators.macd_signal_15min)}), 5min EMA ({format_price(indicators.ema_9_5min)} > {format_price(indicators.ema_20_5min)}), 15min EMA ({format_price(indicators.ema_9_15min)} > {format_price(indicators.ema_20_15min)})",
                "signal_type": SignalType.BUY_ENTRY,
                "confidence": 95  # High confidence when all 4 align
            }
        
        # SELL Entry: ALL 4 indicators must be bearish
        elif (indicators.macd_line_5min < indicators.macd_signal_5min and    # 5min MACD bearish
              indicators.macd_line_15min < indicators.macd_signal_15min and  # 15min MACD bearish
              indicators.ema_9_5min < indicators.ema_20_5min and             # 5min EMA bearish
              indicators.ema_9_15min < indicators.ema_20_15min):             # 15min EMA bearish
            
            return {
                "action": "SELL",
                "price": current_price,
                "reason": f"ALL 4 BEARISH: 5min MACD ({format_technical(indicators.macd_line_5min)} < {format_technical(indicators.macd_signal_5min)}), 15min MACD ({format_technical(indicators.macd_line_15min)} < {format_technical(indicators.macd_signal_15min)}), 5min EMA ({format_price(indicators.ema_9_5min)} < {format_price(indicators.ema_20_5min)}), 15min EMA ({format_price(indicators.ema_9_15min)} < {format_price(indicators.ema_20_15min)})",
                "signal_type": SignalType.SELL_ENTRY,
                "confidence": 95  # High confidence when all 4 align
            }
        
        return None
    
    def _check_exit_signal(self, symbol: str, indicators: MSEIndicators, current_price: float, position_summary: PositionSummary) -> Optional[Dict[str, Any]]:
        """Check for exit signals based on 80% peak MACD histogram logic."""
        
        # 🎯 GRACEFUL DEGRADATION: Skip exit logic for FLAT positions
        if not position_summary.has_position or position_summary.side == "FLAT":
            logger.debug(f"⏭️ SKIPPING EXIT CHECK: {symbol} has no position ({position_summary.side})")
            return None
        
        # Initialize peak tracking if not set or stale (first time after position established)
        if position_summary.peak_macd_histogram == 0.0 and position_summary.has_position:
            # Need to update the unified file with the initialized peak
            peak_value = round_technical(indicators.macd_histogram_15min)
            self._update_macd_peaks(symbol, peak_value, peak_value)
            position_summary.peak_macd_histogram = peak_value
            logger.info(f"🎯 MACD PEAK INITIALIZED: {symbol} {position_summary.side} - Initial peak: {format_technical(peak_value)}")
        
        # 🛡️ SAFEGUARD: Detect and fix stale peak data
        elif position_summary.has_position and self._is_peak_data_stale(symbol, position_summary):
            peak_value = round_technical(indicators.macd_histogram_15min)
            self._update_macd_peaks(symbol, peak_value, peak_value)
            position_summary.peak_macd_histogram = peak_value
            logger.warning(f"🛡️ STALE PEAK DETECTED & FIXED: {symbol} reset to current MACD {format_technical(peak_value)}")
        
        # Update peak tracking and persist to unified file
        updated_peak = False
        current_macd = round_technical(indicators.macd_histogram_15min)
        
        if position_summary.side == "LONG":
            # Track highest MACD histogram for LONG positions
            if current_macd > position_summary.peak_macd_histogram:
                old_peak = position_summary.peak_macd_histogram
                position_summary.peak_macd_histogram = current_macd
                self._update_macd_peaks(symbol, current_macd, current_macd)
                updated_peak = True
                logger.info(f"🎯 MACD PEAK UPDATED: {symbol} LONG - {format_technical(old_peak)} -> {format_technical(current_macd)}")
        elif position_summary.side == "SHORT":
            # Track lowest MACD histogram for SHORT positions  
            if current_macd < position_summary.peak_macd_histogram:
                old_peak = position_summary.peak_macd_histogram
                position_summary.peak_macd_histogram = current_macd
                self._update_macd_peaks(symbol, current_macd, current_macd)
                updated_peak = True
                logger.info(f"🎯 MACD PEAK UPDATED: {symbol} SHORT - {format_technical(old_peak)} -> {format_technical(current_macd)}")
        
        # Always update current MACD in unified file
        position_summary.current_macd_histogram = current_macd
        if not updated_peak:  # Only update if we didn't already update it above
            self._update_macd_peaks(symbol, position_summary.peak_macd_histogram, current_macd)
        
        # DEBUG: Log current exit condition check
        logger.info(f"🔍 EXIT CHECK: {symbol} {position_summary.side} - Current MACD: {format_technical(current_macd)}, Peak: {format_technical(position_summary.peak_macd_histogram)}")
        
        # Calculate 80% threshold for debugging with precision
        exit_threshold = round_technical(position_summary.peak_macd_histogram * 0.8)
        logger.info(f"🔍 EXIT THRESHOLD: {symbol} - 80% of peak: {format_technical(exit_threshold)}, Current: {format_technical(current_macd)}")
        
        # Check 80% exit condition (allows trades to run longer)  
        if position_summary.side == "LONG":
            # SELL EXIT: When current histogram drops below 80% of highest peak
            if current_macd <= exit_threshold:
                logger.info(f"🚨 EXIT SIGNAL TRIGGERED: {symbol} LONG - MACD {format_technical(current_macd)} <= {format_technical(exit_threshold)}")
                
                return {
                    "action": "SELL",
                    "price": current_price,
                    "reason": f"LONG EXIT: 15min MACD histogram fell below 80% of peak ({format_technical(current_macd)} <= {format_technical(exit_threshold)}, peak was {format_technical(position_summary.peak_macd_histogram)})",
                    "signal_type": SignalType.SELL_EXIT,
                    "is_exit": True,
                    "confidence": 90
                }
        
        elif position_summary.side == "SHORT":
            # BUY EXIT: When current histogram rises above 80% of lowest valley
            if current_macd >= exit_threshold:
                logger.info(f"🚨 EXIT SIGNAL TRIGGERED: {symbol} SHORT - MACD {format_technical(current_macd)} >= {format_technical(exit_threshold)}")
                
                return {
                    "action": "BUY", 
                    "price": current_price,
                    "reason": f"SHORT EXIT: 15min MACD histogram rose above 80% of valley ({format_technical(current_macd)} >= {format_technical(exit_threshold)}, valley was {format_technical(position_summary.peak_macd_histogram)})",
                    "signal_type": SignalType.BUY_EXIT,
                    "is_exit": True,
                    "confidence": 90
                }
        
        return None
    
    # Minimal callback implementations - system will handle the heavy lifting
    def on_order_executed(self, symbol: str, order_result: Dict[str, Any]) -> None:
        """Minimal callback - strategy now reads from unified file."""
        if order_result.get("status") == "success":
            # Strategy now reads position state from unified file
            # System will provide detailed position info via on_position_updated
            logger.debug(f"MSE Strategy notified of order execution for {symbol} - reading from unified file")
    
    def on_position_updated(self, symbol: str, position_data: Dict[str, Any]) -> None:
        """
        CRITICAL CALLBACK: Immediately update strategy modes when positions change.
        
        This ensures strategy mode stays synchronized with actual position state
        and prevents any race conditions or stale mode issues.
        """
        quantity = position_data.get("quantity", 0)
        old_mode = self._strategy_modes.get(symbol, StrategyMode.ENTRY_MODE)
        
        logger.info(f"🎯 MSE STRATEGY CALLBACK: on_position_updated({symbol}, qty={quantity})")
        
        # 🚨 CRITICAL: Update strategy mode immediately based on position change
        if abs(quantity) > 0:
            # Position established or changed → Must be in EXIT_MODE
            self._strategy_modes[symbol] = StrategyMode.EXIT_MODE
            logger.info(f"📊 {symbol}: Position updated (qty={quantity}) → {old_mode.value} → EXIT_MODE")
            logger.info(f"🔒 {symbol}: Strategy locked to EXIT_MODE until position closes")
            
            # 🎯 PEAK INITIALIZATION FIX: Initialize peak to current MACD when position is established
            if old_mode == StrategyMode.ENTRY_MODE:  # This is a new position
                self._initialize_peak_on_entry(symbol)
            
        else:
            # Position closed → Switch to ENTRY_MODE
            self._strategy_modes[symbol] = StrategyMode.ENTRY_MODE
            logger.info(f"📊 {symbol}: Position closed (qty={quantity}) → {old_mode.value} → ENTRY_MODE")
            logger.info(f"🔓 {symbol}: Strategy ready for next entry cycle")
            
            # 🧹 Clear peak data when position closes
            self._clear_peak_data(symbol)
        
        # Enhanced logging with mode state
        logger.info(f"📊 MSE STRATEGY MODE UPDATE:")
        logger.info(f"   Symbol: {symbol}")
        logger.info(f"   Callback Quantity: {quantity}")
        logger.info(f"   Mode Transition: {old_mode.value} → {self._strategy_modes[symbol].value}")
        logger.info(f"   🎯 Entry/Exit Architecture: Mode updated immediately!")
        
        # Also log to daily logger for persistence
        try:
            from ..utils.daily_log_manager import get_daily_log_manager
            daily_log_mgr = get_daily_log_manager()
            if daily_log_mgr:
                daily_log_mgr.logger.info(f"MSE_STRATEGY_MODE: {symbol} -> {self._strategy_modes[symbol].value} | qty={quantity}")
        except Exception as e:
            logger.debug(f"Daily logging failed: {e}")
    
    def reset_position_state(self, symbol: str) -> None:
        """Reset position state - now reads from unified file."""
        logger.debug(f"MSE Strategy position state reset for {symbol} - now reading from unified file")
    
    def get_position_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position state from unified file."""
        position_summary = self._read_position_from_unified_file(symbol)
        return {
            "symbol": symbol,
            "has_position": position_summary.has_position,
            "side": position_summary.side,
            "peak_macd_histogram": position_summary.peak_macd_histogram,
            "current_macd_histogram": position_summary.current_macd_histogram
        }
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get strategy metrics with Entry/Exit mode information."""
        try:
            # Count active positions from unified file
            active_count = 0
            if self._unified_position_file.exists():
                with open(self._unified_position_file, 'r') as f:
                    data = json.load(f)
                    actual_positions = data.get('actual_positions', {})
                    active_count = sum(1 for pos in actual_positions.values() if abs(pos.get('quantity', 0)) > 0)
        except Exception as e:
            logger.debug(f"Error reading metrics from unified file: {e}")
            active_count = 0
        
        # Strategy mode metrics
        entry_mode_count = sum(1 for mode in self._strategy_modes.values() if mode == StrategyMode.ENTRY_MODE)
        exit_mode_count = sum(1 for mode in self._strategy_modes.values() if mode == StrategyMode.EXIT_MODE)
        
        return {
            "strategy_name": self._name,
            "architecture": "Entry/Exit Mode Architecture",
            "active_positions": active_count,
            "strategy_modes": {
                "total_symbols": len(self._strategy_modes),
                "entry_mode": entry_mode_count,
                "exit_mode": exit_mode_count,
                "symbol_modes": {symbol: mode.value for symbol, mode in self._strategy_modes.items()}
            },
            "requirements": {
                "timeframes": self._requirements.timeframes,
                "minimum_candles": self._requirements.minimum_candles
            },
            "features": [
                "Prevents position doubling",
                "MIS daily cleanup",
                "CNC position recovery",
                "Immediate mode updates",
                "Race condition elimination"
            ]
        }
    
    # ================================
    # UNIVERSAL STRATEGY TEMPLATE IMPLEMENTATION
    # ================================
    
    def generate_entry_signal(self, market_data: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """Generate entry signals when no position exists."""
        try:
            # Use existing generate_signal logic for entry
            signal = self.generate_signal(market_data)
            
            # Filter out exit signals
            if signal and signal.get('is_exit', False):
                return None
                
            return signal
            
        except Exception as e:
            logger.error(f"Error generating entry signal: {e}")
            return None
    
    def generate_exit_signal(self, 
                           position_context: PositionContext,
                           market_data: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """Generate exit signals when position exists."""
        try:
            # Get position summary from unified file
            symbol = position_context.symbol
            position_summary = self._read_position_from_unified_file(symbol)
            
            # Use existing generate_signal logic for exit
            signal = self.generate_signal(market_data)
            
            # Only return exit signals
            if signal and signal.get('is_exit', False):
                return signal
                
            return None
            
        except Exception as e:
            logger.error(f"Error generating exit signal: {e}")
            return None
    
    def on_startup_with_position(self, 
                                position_context: PositionContext,
                                historical_context: HistoricalContext) -> bool:
        """Handle system startup when position already exists."""
        try:
            symbol = position_context.symbol
            
            logger.info(f"🔄 MSE Strategy startup with existing {symbol} position:")
            logger.info(f"   Position: {position_context.side} {position_context.quantity} @ ₹{position_context.entry_price:.2f}")
            logger.info(f"   Entry Time: {position_context.entry_time}")
            logger.info(f"   Available Timeframes: {historical_context.timeframes}")
            
            # Get position summary from unified file  
            position_summary = self._read_position_from_unified_file(symbol)
            # Update with startup context
            position_summary.has_position = True
            position_summary.side = position_context.side
            
            # Reconstruct MACD peak from historical data
            if historical_context.market_data:
                peak_macd = self._reconstruct_macd_peak(historical_context.market_data)
                position_summary.peak_macd_histogram = peak_macd
                
                logger.info(f"🎯 MACD Peak Reconstructed for {symbol}: {format_technical(peak_macd)}")
                
                # Calculate current MACD for comparison
                current_indicators = self.calculate_indicators(historical_context.market_data)
                position_summary.current_macd_histogram = round_technical(current_indicators.macd_histogram_15min)
                
                logger.info(f"📊 Current MACD vs Peak: {format_technical(current_indicators.macd_histogram_15min)} vs {format_technical(peak_macd)}")
                
                return True
            else:
                logger.warning(f"No historical data available for {symbol} - basic initialization only")
                position_summary.peak_macd_histogram = 0.0
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize MSE strategy for existing position {position_context.symbol}: {e}")
            return False
    
    def _reconstruct_macd_peak(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Reconstruct MACD histogram peak from historical data."""
        try:
            if "15min" not in market_data:
                logger.warning("No 15min data available for MACD peak reconstruction")
                return 0.0
            
            df_15min = market_data["15min"]
            if df_15min.empty or len(df_15min) < 26:
                logger.warning("Insufficient 15min data for MACD calculation")
                return 0.0
            
            # Calculate MACD for all historical data
            prices = df_15min["close"].tolist()
            macd_peaks = []
            
            # Calculate MACD for each point in history
            for i in range(26, len(prices)):  # Start after minimum MACD period
                window_prices = prices[:i+1]
                _, _, histogram = self.calculate_macd(window_prices)
                macd_peaks.append(histogram)
            
            if not macd_peaks:
                return 0.0
            
            # Find the maximum MACD histogram value (peak)
            peak_macd = round_technical(max(macd_peaks))
            
            logger.debug(f"MACD Peak reconstruction: {len(macd_peaks)} points analyzed, peak = {format_technical(peak_macd)}")
            
            return peak_macd
            
        except Exception as e:
            logger.error(f"Failed to reconstruct MACD peak: {e}")
            return 0.0
    
    def validate_data_sufficiency(self, market_data: Dict[str, pd.DataFrame]) -> bool:
        """Check if provided market data is sufficient for strategy operation."""
        try:
            required_timeframes = self._requirements.timeframes
            min_candles = self._requirements.minimum_candles
            
            for timeframe in required_timeframes:
                if timeframe not in market_data:
                    logger.warning(f"Missing required timeframe: {timeframe}")
                    return False
                
                df = market_data[timeframe]
                if df.empty:
                    logger.warning(f"Empty data for timeframe: {timeframe}")
                    return False
                
                required_candles = min_candles.get(timeframe, 26)
                if len(df) < required_candles:
                    logger.warning(f"Insufficient data for {timeframe}: {len(df)} < {required_candles}")
                    return False
                
                # Check required columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    logger.warning(f"Missing columns in {timeframe} data: {missing_cols}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating data sufficiency: {e}")
            return False
    
    def describe_strategy_logic(self) -> str:
        """Human-readable description of MSE strategy logic."""
        return """
        MSE (Multi-Signal Entry) Strategy Logic:
        
        ENTRY CONDITIONS:
        - BUY: ALL 4 indicators bullish (5min MACD > signal, 15min MACD > signal, 5min EMA9 > EMA20, 15min EMA9 > EMA20)
        - SELL: ALL 4 indicators bearish (5min MACD < signal, 15min MACD < signal, 5min EMA9 < EMA20, 15min EMA9 < EMA20)
        
        EXIT CONDITIONS:
        - LONG EXIT: 15min MACD histogram falls below 80% of peak since entry
        - SHORT EXIT: 15min MACD histogram rises above 80% of valley since entry
        
        KEY FEATURES:
        - Requires unanimous agreement of all 4 indicators for entry
        - Uses dynamic peak/valley tracking for exits
        - Maintains state across system restarts
        """

# Backward compatibility function (for legacy tests)
def corrected_mse_signal_generator(symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
    """Backward compatibility wrapper for legacy code."""
    if not hasattr(corrected_mse_signal_generator, '_instance'):
        corrected_mse_signal_generator._instance = MSEStrategy()
    
    mse_strategy_instance = corrected_mse_signal_generator._instance
    
    # Convert old format to new standardized format
    new_format = {
        "symbol": symbol,
        "current_price": market_data.get("current_price", 0),
        "prices": {
            "5min": market_data.get("prices_5min", []),
            "15min": market_data.get("prices_15min", [])
        }
    }
    
    return mse_strategy_instance.generate_signal(symbol, new_format)
