"""
Clean MSE Strategy - Signal Generation Only

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
from ..utils.precision_handler import round_technical, format_technical

logger = logging.getLogger(__name__)


class StrategyMode(Enum):
    """Strategy operating modes to prevent position doubling."""
    ENTRY_MODE = "ENTRY_MODE"  # Can only generate entry signals
    EXIT_MODE = "EXIT_MODE"    # Can only generate exit signals


class SignalType(Enum):
    """Types of signals that can be generated."""
    BUY_ENTRY = "BUY_ENTRY"
    SELL_ENTRY = "SELL_ENTRY"
    BUY_EXIT = "BUY_EXIT"  # Exit from SHORT
    SELL_EXIT = "SELL_EXIT"  # Exit from LONG
    NO_SIGNAL = "NO_SIGNAL"


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
    
    def __init__(self, position_manager=None):
        self._name = "MSE_Strategy"
        self._requirements = StrategyRequirements(
            timeframes=["5min", "15min"],
            minimum_candles={
                "5min": 40,  # Increased for better EMA/MACD accuracy
                "15min": 40  # Increased for better EMA/MACD accuracy
            },
            warmup_minutes=525  # CRITICAL: Must match backtester for MACD stability (35 * 15min periods)
        )
        
        # Path to unified position file (single source of truth)
        # CRITICAL FIX: Use same path as UnifiedPositionManager writes to
        # Position Manager: live_module/data/positions/positions.json
        base_dir = Path(__file__).parent.parent.parent.parent  # Go up to project root
        self._unified_position_file = base_dir / "live_module" / "data" / "positions" / "positions.json"
        
        # 🔧 ARCHITECTURE FIX: Optional position_manager injection (preferred over file reads)
        # If provided, use position_manager.get_position() instead of reading files
        self._position_manager = position_manager  # Will be None if not injected
        
        # Entry/Exit Mode Architecture - Prevents position doubling
        self._strategy_modes: Dict[str, StrategyMode] = {}
        
        # Minimal tracking for signal generation
        self._position_summaries: Dict[str, PositionSummary] = {}
        
        # 🚫 CASCADING PREVENTION: Track first trade direction per day per symbol
        # Format: {"SYMBOL_YYYY-MM-DD": "LONG" or "SHORT"}
        # Blocks: LONG after LONG, SHORT after SHORT on same day
        self._daily_direction_tracker: Dict[str, str] = {}
        
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
        - Check positions.json for existing positions
        - If position exists → Set EXIT_MODE (can only exit)
        - If no position → Set ENTRY_MODE (can enter new)
        - Fallback: Default to ENTRY_MODE if anything fails (safe)
        """
        try:
            logger.info("🔄 STRATEGY MODE RECOVERY: Starting recovery process")
            
            # Load position data if available
            position_data = {}
            if self._unified_position_file.exists():
                try:
                    with open(self._unified_position_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    position_data = data.get('actual_positions', {})
                    logger.info(f"📖 Loaded {len(position_data)} positions from unified file")
                except Exception as e:
                    logger.error(f"Error loading unified position file: {e}")
            else:
                logger.warning(f"Unified position file not found: {self._unified_position_file}")
            
            # If we have position data, process each symbol
            if position_data:
                for symbol, pos_data in position_data.items():
                    try:
                        current_quantity = pos_data.get('quantity', 0)
                        
                        if abs(current_quantity) > 0:
                            self._strategy_modes[symbol] = StrategyMode.EXIT_MODE
                            logger.info(f"🔄 {symbol}: Has position (qty={current_quantity}) → EXIT_MODE")
                        else:
                            self._strategy_modes[symbol] = StrategyMode.ENTRY_MODE
                            logger.debug(f"✅ {symbol}: No position → ENTRY_MODE")
                            
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
            logger.warning("🛡️ Global fallback: Will default to ENTRY_MODE for new symbols")
    
    def _read_position_from_unified_file(self, symbol: str) -> PositionSummary:
        """
        Read position state DIRECTLY from broker-reconciled actual_positions.
        Single source of truth - no more sync issues.
        """
        try:
            if not self._unified_position_file.exists():
                logger.debug(f"Unified position file not found: {self._unified_position_file}")
                return PositionSummary(symbol)
            
            with open(self._unified_position_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # DEBUG: Log what we're reading
            logger.info(f"🔍 DEBUG READ: File timestamp: {data.get('timestamp', 'N/A')}")
            
            # FIXED: Read from strategy_positions for complete strategy data
            strategy_positions = data.get('strategy_positions', {})
            strategy_data = strategy_positions.get(self._name, {})  # Get this strategy's positions
            position_data = strategy_data.get(symbol, {})

            logger.info(f"🔍 DEBUG READ: strategy_positions keys: {list(strategy_positions.keys())}")
            logger.info(f"🔍 DEBUG READ: {self._name} positions: {list(strategy_data.keys())}")
            logger.info(f"🔍 DEBUG READ: {symbol} position_data = {position_data}")

            # Get quantity from strategy position (already synced with broker via reconciliation)
            quantity = position_data.get('quantity', 0)
            side = position_data.get('side', 'FLAT')

            logger.info(f"🔍 DEBUG READ: {symbol} quantity = {quantity}, side = {side}")

            # Create position summary based on strategy position (which includes broker-reconciled data)
            position_summary = PositionSummary(symbol)
            position_summary.has_position = abs(quantity) > 0
            position_summary.side = side

            # FIXED: Read peak_macd from strategy_positions (not the non-existent 'strategy_macd_peaks')
            position_summary.peak_macd_histogram = position_data.get('peak_macd_histogram', 0.0)
            position_summary.current_macd_histogram = position_data.get('current_macd_histogram', 0.0)

            # FIXED: Load entry indicators from strategy position
            position_summary.entry_signal_indicators = position_data.get('entry_signal_indicators', None)

            logger.info(f"📖 STRATEGY DATA: {symbol} → {position_summary.side} (qty={quantity}, peak_macd={position_summary.peak_macd_histogram:.4f}, has_position={position_summary.has_position})")
            
            return position_summary
            
        except Exception as e:
            logger.error(f"Error reading position from unified file for {symbol}: {e}")
            return PositionSummary(symbol)
    
    def _get_or_create_strategy_mode(self, symbol: str) -> StrategyMode:
        """
        Get current mode for symbol or create default ENTRY_MODE.
        Safe fallback ensures we never have undefined state.
        """
        if symbol not in self._strategy_modes:
            # Default to ENTRY_MODE for new symbols (safe default)
            self._strategy_modes[symbol] = StrategyMode.ENTRY_MODE
            logger.debug(f"🆕 {symbol}: Creating new mode → ENTRY_MODE (default)")
        
        return self._strategy_modes[symbol]
    
    def _switch_to_exit_mode(self, symbol: str) -> None:
        """
        Switch symbol to EXIT_MODE after successful entry execution.
        Prevents strategy from generating another entry signal.
        """
        self._strategy_modes[symbol] = StrategyMode.EXIT_MODE
        logger.info(f"🔄 {symbol}: Switched to EXIT_MODE (position entered)")
    
    def _switch_to_entry_mode(self, symbol: str) -> None:
        """
        Switch symbol to ENTRY_MODE after successful exit execution.
        Allows strategy to generate new entry signals.
        """
        self._strategy_modes[symbol] = StrategyMode.ENTRY_MODE
        logger.info(f"🔄 {symbol}: Switched to ENTRY_MODE (position closed)")
    
    def calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate EMA with fixed period and 4-decimal precision."""
        if len(prices) < period:
            return 0.0
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period  # Start with SMA
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return round_technical(ema)
    
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
        """
        Calculate all technical indicators for both timeframes.
        
        CRITICAL: Uses PREVIOUS bar data to prevent look-ahead bias.
        Signal decisions based on bar N-1, execution on bar N (matches backtester).
        """
        
        # Extract price data for both timeframes (use all but last candle for previous bar data)
        prices_5min = []
        prices_15min = []
        
        if "5min" in market_data and not market_data["5min"].empty:
            # Use all data for indicator calculation, but will use -2 index for signal generation
            prices_5min = market_data["5min"]["close"].tolist()
        
        if "15min" in market_data and not market_data["15min"].empty:
            # Use all data for indicator calculation, but will use -2 index for signal generation  
            prices_15min = market_data["15min"]["close"].tolist()
        
        # Calculate 5min indicators using ALL available data
        ema_9_5min = self.calculate_ema(prices_5min, 9)
        ema_20_5min = self.calculate_ema(prices_5min, 20)
        macd_line_5min, macd_signal_5min, macd_histogram_5min = self.calculate_macd(prices_5min)
        
        # Calculate 15min indicators using ALL available data
        ema_9_15min = self.calculate_ema(prices_15min, 9)
        ema_20_15min = self.calculate_ema(prices_15min, 20)
        macd_line_15min, macd_signal_15min, macd_histogram_15min = self.calculate_macd(prices_15min)
        
        # For live trading: Use PREVIOUS bar's indicator values for signal decisions
        # Calculate indicators for previous bar state (simulate backtester's .shift(1))
        if len(prices_5min) >= 2:
            prev_prices_5min = prices_5min[:-1]  # Remove last candle
            ema_9_5min = self.calculate_ema(prev_prices_5min, 9) 
            ema_20_5min = self.calculate_ema(prev_prices_5min, 20)
            macd_line_5min, macd_signal_5min, macd_histogram_5min = self.calculate_macd(prev_prices_5min)
            
        if len(prices_15min) >= 2:
            prev_prices_15min = prices_15min[:-1]  # Remove last candle
            ema_9_15min = self.calculate_ema(prev_prices_15min, 9)
            ema_20_15min = self.calculate_ema(prev_prices_15min, 20)
            macd_line_15min, macd_signal_15min, macd_histogram_15min = self.calculate_macd(prev_prices_15min)
        
        return MSEIndicators(
            ema_9_5min=ema_9_5min,
            ema_20_5min=ema_20_5min,
            macd_line_5min=macd_line_5min,
            macd_signal_5min=macd_signal_5min,
            macd_histogram_5min=macd_histogram_5min,
            ema_9_15min=ema_9_15min,
            ema_20_15min=ema_20_15min,
            macd_line_15min=macd_line_15min,
            macd_signal_15min=macd_signal_15min,
            macd_histogram_15min=macd_histogram_15min
        )
    
    def generate_signal(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate trading signal based on MSE strategy logic with mode-aware architecture.
        
        KEY FEATURE: Reads position state from unified positions.json file (single source of truth).
        Prevents position doubling by checking current mode before generating signals.
        
        Returns signal intent only - no position management, no order execution.
        The system will handle all execution, risk management, and position tracking.
        """
        try:
            # Extract symbol
            symbol = None
            for timeframe, df in market_data.items():
                if not df.empty and 'symbol' in df.columns:
                    symbol = df['symbol'].iloc[-1]
                    break
            
            if not symbol:
                return {}
            
            # Check data sufficiency
            if not self.has_sufficient_data({"candles": dict(market_data.items())}):
                return {}
            
            # Get current price
            current_price = 0
            for timeframe, df in market_data.items():
                if not df.empty:
                    current_price = df['close'].iloc[-1]
                    break
            
            if not current_price:
                return {}
            
            # Calculate indicators
            indicators = self.calculate_indicators(market_data)
            
            # 🎯 CRITICAL: Read position from unified file (single source of truth)
            position_summary = self._read_position_from_unified_file(symbol)
            
            # Update in-memory cache
            self._position_summaries[symbol] = position_summary
            
            # Get current strategy mode
            current_mode = self._get_or_create_strategy_mode(symbol)
            
            # CRITICAL LOGGING: Show what position state strategy is using
            logger.info(f"🎯 MSE SIGNAL GENERATION for {symbol}:")
            logger.info(f"   Current Mode: {current_mode.value}")
            logger.info(f"   Position from unified file: has_position={position_summary.has_position}, side={position_summary.side}")
            logger.info(f"   Peak MACD: {position_summary.peak_macd_histogram:.4f}")
            
            # 🛡️ MODE-AWARE SIGNAL GENERATION (Position Doubling Prevention)
            if current_mode == StrategyMode.EXIT_MODE:
                # In EXIT_MODE: Can ONLY generate exit signals
                if position_summary.has_position:
                    logger.info(f"   🔄 Checking EXIT signals (position exists: {position_summary.side})")
                    signal = self._check_exit_signal(symbol, indicators, current_price, position_summary)
                else:
                    # Position was closed but mode wasn't switched - auto-correct
                    logger.warning(f"⚠️ {symbol} in EXIT_MODE but no position found - switching to ENTRY_MODE")
                    self._switch_to_entry_mode(symbol)
                    signal = {}
            
            elif current_mode == StrategyMode.ENTRY_MODE:
                # In ENTRY_MODE: Can ONLY generate entry signals
                if not position_summary.has_position:
                    logger.info(f"   🎯 Checking ENTRY signals (no position)")
                    signal = self._check_entry_signal(symbol, indicators, current_price)
                else:
                    # Position exists but mode wasn't switched - auto-correct
                    logger.warning(f"⚠️ {symbol} in ENTRY_MODE but position found - switching to EXIT_MODE")
                    self._switch_to_exit_mode(symbol)
                    signal = {}
            else:
                # Unknown mode - default to safe behavior
                logger.error(f"❌ {symbol} in unknown mode: {current_mode}")
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
        
        # 🚫 CASCADING PREVENTION: Check if same direction already traded today
        today_key = f"{symbol}_{datetime.now().strftime('%Y-%m-%d')}"
        daily_direction = self._daily_direction_tracker.get(today_key, None)
        
        # BUY Entry: ALL 4 indicators must be bullish
        if (indicators.macd_line_5min > indicators.macd_signal_5min and      # 5min MACD bullish
            indicators.macd_line_15min > indicators.macd_signal_15min and    # 15min MACD bullish  
            indicators.ema_9_5min > indicators.ema_20_5min and               # 5min EMA bullish
            indicators.ema_9_15min > indicators.ema_20_15min):               # 15min EMA bullish
            
            # 🚫 Block if already went LONG today
            if daily_direction == "LONG":
                logger.warning(f"🚫 CASCADING BLOCKED: {symbol} - Already executed LONG today, cannot pyramid same direction")
                return None
            
            # Record this direction for today
            self._daily_direction_tracker[today_key] = "LONG"
            logger.info(f"✅ DIRECTION LOGGED: {symbol} - First LONG entry today at {datetime.now().strftime('%H:%M:%S')}")
            
            return {
                "action": "BUY",
                "price": current_price,
                "reason": f"ALL 4 BULLISH: 5min MACD ({indicators.macd_line_5min:.4f} > {indicators.macd_signal_5min:.4f}), 15min MACD ({indicators.macd_line_15min:.4f} > {indicators.macd_signal_15min:.4f}), 5min EMA ({indicators.ema_9_5min:.2f} > {indicators.ema_20_5min:.2f}), 15min EMA ({indicators.ema_9_15min:.2f} > {indicators.ema_20_15min:.2f})",
                "signal_type": SignalType.BUY_ENTRY,
                "confidence": 95,  # High confidence when all 4 align
                # CRITICAL FIX: Store entry indicators snapshot for position initialization
                "entry_indicators": {
                    "macd_5min": indicators.macd_line_5min,
                    "macd_signal_5min": indicators.macd_signal_5min,
                    "macd_histogram_5min": indicators.macd_histogram_5min,
                    "macd_15min": indicators.macd_line_15min,
                    "macd_signal_15min": indicators.macd_signal_15min,
                    "macd_histogram_15min": indicators.macd_histogram_15min,
                    "entry_peak_macd_histogram": indicators.macd_histogram_15min,  # Store entry peak for exit threshold
                    "ema_9_5min": indicators.ema_9_5min,
                    "ema_20_5min": indicators.ema_20_5min,
                    "ema_9_15min": indicators.ema_9_15min,
                    "ema_20_15min": indicators.ema_20_15min
                }
            }
        
        # SELL Entry: ALL 4 indicators must be bearish
        elif (indicators.macd_line_5min < indicators.macd_signal_5min and    # 5min MACD bearish
              indicators.macd_line_15min < indicators.macd_signal_15min and  # 15min MACD bearish
              indicators.ema_9_5min < indicators.ema_20_5min and             # 5min EMA bearish
              indicators.ema_9_15min < indicators.ema_20_15min):             # 15min EMA bearish
            
            # 🚫 Block if already went SHORT today
            if daily_direction == "SHORT":
                logger.warning(f"🚫 CASCADING BLOCKED: {symbol} - Already executed SHORT today, cannot pyramid same direction")
                return None
            
            # Record this direction for today
            self._daily_direction_tracker[today_key] = "SHORT"
            logger.info(f"✅ DIRECTION LOGGED: {symbol} - First SHORT entry today at {datetime.now().strftime('%H:%M:%S')}")
            
            return {
                "action": "SELL",
                "price": current_price,
                "reason": f"ALL 4 BEARISH: 5min MACD ({indicators.macd_line_5min:.4f} < {indicators.macd_signal_5min:.4f}), 15min MACD ({indicators.macd_line_15min:.4f} < {indicators.macd_signal_15min:.4f}), 5min EMA ({indicators.ema_9_5min:.2f} < {indicators.ema_20_5min:.2f}), 15min EMA ({indicators.ema_9_15min:.2f} < {indicators.ema_20_15min:.2f})",
                "signal_type": SignalType.SELL_ENTRY,
                "confidence": 95,  # High confidence when all 4 align
                # CRITICAL FIX: Store entry indicators snapshot for position initialization
                "entry_indicators": {
                    "macd_5min": indicators.macd_line_5min,
                    "macd_signal_5min": indicators.macd_signal_5min,
                    "macd_histogram_5min": indicators.macd_histogram_5min,
                    "macd_15min": indicators.macd_line_15min,
                    "macd_signal_15min": indicators.macd_signal_15min,
                    "macd_histogram_15min": indicators.macd_histogram_15min,
                    "entry_peak_macd_histogram": indicators.macd_histogram_15min,  # Store raw negative for SHORT (peak tracking expects negative)
                    "ema_9_5min": indicators.ema_9_5min,
                    "ema_20_5min": indicators.ema_20_5min,
                    "ema_9_15min": indicators.ema_9_15min,
                    "ema_20_15min": indicators.ema_20_15min
                }
            }
        
        return None
    
    def _check_exit_signal(self, symbol: str, indicators: MSEIndicators, current_price: float, position_summary: PositionSummary) -> Optional[Dict[str, Any]]:
        """Check for exit signals based on 80% peak MACD histogram logic."""
        
        # Initialize peak tracking if not set (first time after position established)
        if position_summary.peak_macd_histogram == 0.0 and position_summary.has_position:
            # CRITICAL FIX: Try to recover from entry_signal_indicators first
            entry_indicators = position_summary.entry_signal_indicators or {}
            entry_peak = entry_indicators.get('entry_peak_macd_histogram', 0.0)
            
            if entry_peak != 0.0:
                # SUCCESS: Recovered entry peak from position state
                position_summary.peak_macd_histogram = entry_peak
                logger.info(f"🎯 MACD PEAK RECOVERED from ENTRY: {symbol} - Peak: {entry_peak:.4f}")
            else:
                # FALLBACK: Use current MACD only if no entry peak available
                position_summary.peak_macd_histogram = indicators.macd_histogram_15min
                logger.warning(f"⚠️ MACD PEAK defaulted to CURRENT: {symbol} - {indicators.macd_histogram_15min:.4f} (NO ENTRY PEAK FOUND - this may cause incorrect exits)")
        
        # Update peak tracking
        if position_summary.side == "LONG":
            # Track highest MACD histogram for LONG positions
            if indicators.macd_histogram_15min > position_summary.peak_macd_histogram:
                old_peak = position_summary.peak_macd_histogram
                position_summary.peak_macd_histogram = indicators.macd_histogram_15min
                logger.info(f"🎯 MACD PEAK UPDATED: {symbol} LONG - {old_peak:.4f} -> {indicators.macd_histogram_15min:.4f}")
        elif position_summary.side == "SHORT":
            # Track lowest MACD histogram for SHORT positions  
            if indicators.macd_histogram_15min < position_summary.peak_macd_histogram:
                old_peak = position_summary.peak_macd_histogram
                position_summary.peak_macd_histogram = indicators.macd_histogram_15min
                logger.info(f"🎯 MACD PEAK UPDATED: {symbol} SHORT - {old_peak:.4f} -> {indicators.macd_histogram_15min:.4f}")
        
        position_summary.current_macd_histogram = indicators.macd_histogram_15min
        
        # DEBUG: Log current exit condition check
        logger.info(f"🔍 EXIT CHECK: {symbol} {position_summary.side} - Current MACD: {indicators.macd_histogram_15min:.4f}, Peak: {position_summary.peak_macd_histogram:.4f}")

        # Calculate 95% recovery threshold (5% remaining) for debugging
        exit_threshold = position_summary.peak_macd_histogram * 0.05
        logger.info(f"🔍 EXIT THRESHOLD: {symbol} - 95% recovery (5% remaining): {exit_threshold:.4f}, Current: {indicators.macd_histogram_15min:.4f}")

        # CRITICAL: Check for zero-cross (complete trend reversal) BEFORE normal exit logic
        # This catches extreme reversals where MACD crosses zero line
        if position_summary.side == "LONG":
            # LONG position: Exit immediately if MACD crosses below zero (bearish reversal)
            if indicators.macd_histogram_15min <= 0 and position_summary.peak_macd_histogram > 0:
                logger.warning(f"🚨 ZERO-CROSS REVERSAL: {symbol} LONG - MACD crossed below zero! ({indicators.macd_histogram_15min:.4f} <= 0, peak was {position_summary.peak_macd_histogram:.4f})")
                return {
                    "action": "SELL",
                    "price": current_price,
                    "reason": f"LONG EXIT: MACD zero-cross reversal (MACD: {indicators.macd_histogram_15min:.4f} <= 0, complete trend reversal from peak {position_summary.peak_macd_histogram:.4f})",
                    "signal_type": SignalType.SELL_EXIT,
                    "is_exit": True,
                    "exit_type": "ZERO_CROSS",
                    "exit_reason": f"MACD zero-cross: {indicators.macd_histogram_15min:.4f} crossed below 0",
                    "exit_indicators": {
                        "peak_macd": position_summary.peak_macd_histogram,
                        "current_macd": indicators.macd_histogram_15min,
                        "zero_cross": True
                    },
                    "confidence": 95
                }

        elif position_summary.side == "SHORT":
            # SHORT position: Exit immediately if MACD crosses above zero (bullish reversal)
            if indicators.macd_histogram_15min >= 0 and position_summary.peak_macd_histogram < 0:
                logger.warning(f"🚨 ZERO-CROSS REVERSAL: {symbol} SHORT - MACD crossed above zero! ({indicators.macd_histogram_15min:.4f} >= 0, valley was {position_summary.peak_macd_histogram:.4f})")
                return {
                    "action": "BUY",
                    "price": current_price,
                    "reason": f"SHORT EXIT: MACD zero-cross reversal (MACD: {indicators.macd_histogram_15min:.4f} >= 0, complete trend reversal from valley {position_summary.peak_macd_histogram:.4f})",
                    "signal_type": SignalType.BUY_EXIT,
                    "is_exit": True,
                    "exit_type": "ZERO_CROSS",
                    "exit_reason": f"MACD zero-cross: {indicators.macd_histogram_15min:.4f} crossed above 0",
                    "exit_indicators": {
                        "peak_macd": position_summary.peak_macd_histogram,
                        "current_macd": indicators.macd_histogram_15min,
                        "zero_cross": True
                    },
                    "confidence": 95
                }

        # Normal exit logic: Check 95% recovery exit condition (let winners run longer)
        if position_summary.side == "LONG":
            # SELL EXIT: When current histogram drops to 5% of highest peak (95% recovery toward zero)
            exit_threshold = position_summary.peak_macd_histogram * 0.05
            if indicators.macd_histogram_15min <= exit_threshold:
                logger.info(f"� EXIT_TYPE: STRATEGY | Symbol: {symbol} | MACD Peak Exit (LONG)")
                logger.info(f"   Peak MACD: {position_summary.peak_macd_histogram:.4f} | Current MACD: {indicators.macd_histogram_15min:.4f} | Threshold: {exit_threshold:.4f}")
                
                return {
                    "action": "SELL",
                    "price": current_price,
                    "reason": f"LONG EXIT: 15min MACD histogram at 95% recovery ({indicators.macd_histogram_15min:.4f} <= {exit_threshold:.4f}, peak was {position_summary.peak_macd_histogram:.4f})",
                    "signal_type": SignalType.SELL_EXIT,
                    "is_exit": True,
                    "exit_type": "STRATEGY",  # CRITICAL FIX: Mark exit type for logging
                    "exit_reason": f"MACD peak exit: {indicators.macd_histogram_15min:.4f} <= {exit_threshold:.4f}",
                    "exit_indicators": {
                        "peak_macd": position_summary.peak_macd_histogram,
                        "current_macd": indicators.macd_histogram_15min,
                        "exit_threshold": exit_threshold
                    },
                    "confidence": 90
                }
        
        elif position_summary.side == "SHORT":
            # BUY EXIT: When current histogram rises to 5% of lowest valley (95% recovery toward zero)
            exit_threshold = position_summary.peak_macd_histogram * 0.05
            if indicators.macd_histogram_15min >= exit_threshold:
                logger.info(f"� EXIT_TYPE: STRATEGY | Symbol: {symbol} | MACD Peak Exit (SHORT)")
                logger.info(f"   Peak MACD: {position_summary.peak_macd_histogram:.4f} | Current MACD: {indicators.macd_histogram_15min:.4f} | Threshold: {exit_threshold:.4f}")
                
                return {
                    "action": "BUY", 
                    "price": current_price,
                    "reason": f"SHORT EXIT: 15min MACD histogram at 95% recovery ({indicators.macd_histogram_15min:.4f} >= {exit_threshold:.4f}, valley was {position_summary.peak_macd_histogram:.4f})",
                    "signal_type": SignalType.BUY_EXIT,
                    "is_exit": True,
                    "exit_type": "STRATEGY",  # CRITICAL FIX: Mark exit type for logging
                    "exit_reason": f"MACD peak exit: {indicators.macd_histogram_15min:.4f} >= {exit_threshold:.4f}",
                    "exit_indicators": {
                        "peak_macd": position_summary.peak_macd_histogram,
                        "current_macd": indicators.macd_histogram_15min,
                        "exit_threshold": exit_threshold
                    },
                    "confidence": 90
                }
        
        return None
    
    # Minimal callback implementations - system will handle the heavy lifting
    def on_order_executed(self, symbol: str, order_result: Dict[str, Any]) -> None:
        """
        Minimal callback - just update position summary for signal generation.
        Also handles mode switching based on order type.
        """
        if order_result.get("status") == "success":
            # Update minimal position info needed for signal generation
            if symbol not in self._position_summaries:
                self._position_summaries[symbol] = PositionSummary(symbol)
            
            # 🔄 MODE SWITCHING: Switch mode based on order action
            action = order_result.get("action", "").upper()
            
            if action in ["BUY", "SELL"]:
                # Determine if this is entry or exit
                is_exit = order_result.get("is_exit", False)
                
                if is_exit:
                    # Exit order executed → Switch to ENTRY_MODE
                    self._switch_to_entry_mode(symbol)
                    logger.info(f"✅ {symbol}: Exit order executed → ENTRY_MODE")
                else:
                    # Entry order executed → Switch to EXIT_MODE
                    self._switch_to_exit_mode(symbol)
                    logger.info(f"✅ {symbol}: Entry order executed → EXIT_MODE")
            
            # System will provide more detailed position info via on_position_updated
            logger.debug(f"MSE Strategy notified of order execution for {symbol}")
    
    def notify_entry_executed(self, symbol: str) -> None:
        """
        Public method: Called by trading system after successful entry execution.
        Switches symbol to EXIT_MODE to prevent position doubling.
        """
        self._switch_to_exit_mode(symbol)
        logger.info(f"📢 SYSTEM NOTIFICATION: {symbol} entry executed → EXIT_MODE")
    
    def notify_exit_executed(self, symbol: str) -> None:
        """
        Public method: Called by trading system after successful exit execution.
        Switches symbol to ENTRY_MODE to allow new entries.
        """
        self._switch_to_entry_mode(symbol)
        logger.info(f"📢 SYSTEM NOTIFICATION: {symbol} exit executed → ENTRY_MODE")
    
    def get_current_mode(self, symbol: str) -> str:
        """
        Public method: Get current strategy mode for symbol.
        Returns 'ENTRY_MODE' or 'EXIT_MODE'.
        """
        mode = self._get_or_create_strategy_mode(symbol)
        return mode.value
    
    def on_position_updated(self, symbol: str, position_data: Dict[str, Any]) -> None:
        """Update minimal position summary for signal generation."""
        logger.info(f"🎯 MSE STRATEGY CALLBACK: on_position_updated({symbol}, {position_data})")
        
        if symbol not in self._position_summaries:
            self._position_summaries[symbol] = PositionSummary(symbol)
            logger.info(f"📝 CREATED new position summary for {symbol}")
        
        position_summary = self._position_summaries[symbol]
        old_side = position_summary.side
        old_has_position = position_summary.has_position
        
        # CRITICAL SAFETY: Clear recent entry signals when position is established
        if hasattr(self, '_recent_entry_signals') and symbol in self._recent_entry_signals:
            del self._recent_entry_signals[symbol]
            logger.info(f"🔒 SAFETY: Cleared recent entry signal for {symbol} - position established")
        
        # Update minimal info needed for signal generation
        quantity = position_data.get("quantity", 0)
        position_summary.has_position = abs(quantity) > 0
        
        if quantity > 0:
            position_summary.side = "LONG"
        elif quantity < 0:
            position_summary.side = "SHORT"
        else:
            position_summary.side = "FLAT"
        
        # Reset peak tracking when position changes
        if not position_summary.has_position:
            position_summary.peak_macd_histogram = 0.0
            position_summary.current_macd_histogram = 0.0
        
        logger.info(f"📊 MSE STRATEGY POSITION STATE UPDATED:")
        logger.info(f"   Symbol: {symbol}")
        logger.info(f"   Quantity: {quantity}")
        logger.info(f"   Side: {old_side} -> {position_summary.side}")
        logger.info(f"   Has Position: {old_has_position} -> {position_summary.has_position}")
        logger.info(f"   Strategy now knows: {symbol} position = {position_summary.side} ({quantity})")
        
        # Also log to daily logger for persistence
        try:
            from ..utils.daily_log_manager import get_daily_log_manager
            daily_log_mgr = get_daily_log_manager()
            if daily_log_mgr:
                daily_log_mgr.logger.info(f"MSE_STRATEGY_STATE: {symbol} -> {position_summary.side} {quantity} | has_position={position_summary.has_position}")
        except Exception as e:
            logger.debug(f"Daily logging failed: {e}")
    
    def reset_position_state(self, symbol: str) -> None:
        """Reset minimal position state."""
        if symbol in self._position_summaries:
            del self._position_summaries[symbol]
        logger.debug(f"MSE Strategy position state reset for {symbol}")
    
    def get_position_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get minimal position state for monitoring."""
        if symbol in self._position_summaries:
            position_summary = self._position_summaries[symbol]
            return {
                "symbol": symbol,
                "has_position": position_summary.has_position,
                "side": position_summary.side,
                "peak_macd_histogram": position_summary.peak_macd_histogram,
                "current_macd_histogram": position_summary.current_macd_histogram
            }
        return None
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get strategy metrics focused on signal generation."""
        return {
            "strategy_name": self._name,
            "tracked_symbols": len(self._position_summaries),
            "active_positions": sum(1 for pos in self._position_summaries.values() if pos.has_position),
            "requirements": {
                "timeframes": self._requirements.timeframes,
                "minimum_candles": self._requirements.minimum_candles
            },
            "focus": "Signal generation only - execution handled by system"
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
            # Update position summary with context
            symbol = position_context.symbol
            if symbol not in self._position_summaries:
                self._position_summaries[symbol] = PositionSummary(symbol)
            
            position_summary = self._position_summaries[symbol]
            position_summary.has_position = True
            position_summary.side = position_context.side
            
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
            
            # Create position summary
            if symbol not in self._position_summaries:
                self._position_summaries[symbol] = PositionSummary(symbol)
            
            position_summary = self._position_summaries[symbol]
            position_summary.has_position = True
            position_summary.side = position_context.side
            
            # Reconstruct MACD peak from historical data
            if historical_context.market_data:
                peak_macd = self._reconstruct_macd_peak(historical_context.market_data)
                position_summary.peak_macd_histogram = peak_macd
                
                logger.info(f"🎯 MACD Peak Reconstructed for {symbol}: {peak_macd:.4f}")
                
                # Calculate current MACD for comparison
                current_indicators = self.calculate_indicators(historical_context.market_data)
                position_summary.current_macd_histogram = current_indicators.macd_histogram_15min
                
                logger.info(f"📊 Current MACD vs Peak: {current_indicators.macd_histogram_15min:.4f} vs {peak_macd:.4f}")
                
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
            peak_macd = max(macd_peaks)
            
            logger.debug(f"MACD Peak reconstruction: {len(macd_peaks)} points analyzed, peak = {peak_macd:.4f}")
            
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
        - LONG EXIT: 15min MACD histogram falls below 40% of peak since entry
        - SHORT EXIT: 15min MACD histogram rises above 40% of valley since entry
        
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
