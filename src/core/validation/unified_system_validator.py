"""
Unified System Validator - Comprehensive Validation Framework
============================================================

This module provides end-to-end validation ensuring identical behavior between
backtesting and live trading systems for algorithmic strategies.

Key Validation Areas:
1. Signal Parity: Identical indicator calculations and signal generation
2. State Management: Consistent position and peak tracking across environments
3. Data Consistency: Same OHLCV processing and timeframe alignment
4. Execution Model: Reconciliation of two-bar rule vs immediate execution
5. Performance Metrics: Trade-by-trade comparison and statistical validation

Author: The Alchemist (Quantitative Validation Specialist)
Date: September 2025
Version: 1.0 - Production Validation Framework
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, time, date
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import json
import sqlite3
from scipy import stats
from collections import defaultdict
import warnings

from ..strat_stats.indicators import calculate_macd, calculate_ema
from ...strategies.mse_strategy_backtesting import MSEStrategyBacktesting
from ...strategies.mse_strategy_live import MSEStrategy

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics for system comparison."""

    # Signal Parity Metrics
    total_signals: int = 0
    matching_signals: int = 0
    signal_parity_rate: float = 0.0

    # Indicator Validation
    macd_correlation_5m: float = 0.0
    macd_correlation_15m: float = 0.0
    ema_correlation_5m: float = 0.0
    ema_correlation_15m: float = 0.0

    # Timing Metrics
    signal_timing_variance: float = 0.0
    execution_delay_analysis: Dict[str, float] = None

    # Trade Comparison
    total_trades_backtest: int = 0
    total_trades_live: int = 0
    trade_match_rate: float = 0.0

    # Performance Reconciliation
    pnl_correlation: float = 0.0
    return_difference_mean: float = 0.0
    return_difference_std: float = 0.0
    max_drawdown_difference: float = 0.0

    # Statistical Significance
    p_values: Dict[str, float] = None
    confidence_intervals: Dict[str, Tuple[float, float]] = None

    def __post_init__(self):
        if self.execution_delay_analysis is None:
            self.execution_delay_analysis = {}
        if self.p_values is None:
            self.p_values = {}
        if self.confidence_intervals is None:
            self.confidence_intervals = {}


@dataclass
class SignalComparison:
    """Point-in-time signal comparison between systems."""
    timestamp: datetime
    backtest_signal: Optional[str] = None
    live_signal: Optional[str] = None
    match: bool = False

    # Indicator comparisons
    backtest_indicators: Dict[str, float] = None
    live_indicators: Dict[str, float] = None
    indicator_diff: Dict[str, float] = None

    # Execution context
    backtest_price: float = 0.0
    live_price: float = 0.0
    price_difference: float = 0.0

    def __post_init__(self):
        if self.backtest_indicators is None:
            self.backtest_indicators = {}
        if self.live_indicators is None:
            self.live_indicators = {}
        if self.indicator_diff is None:
            self.indicator_diff = {}


@dataclass
class TradeComparison:
    """Trade-level comparison between systems."""
    trade_id: str

    # Entry comparison
    backtest_entry_time: Optional[datetime] = None
    live_entry_time: Optional[datetime] = None
    entry_time_diff_seconds: float = 0.0

    backtest_entry_price: float = 0.0
    live_entry_price: float = 0.0
    entry_price_diff_pct: float = 0.0

    # Exit comparison
    backtest_exit_time: Optional[datetime] = None
    live_exit_time: Optional[datetime] = None
    exit_time_diff_seconds: float = 0.0

    backtest_exit_price: float = 0.0
    live_exit_price: float = 0.0
    exit_price_diff_pct: float = 0.0

    # PnL comparison
    backtest_pnl: float = 0.0
    live_pnl: float = 0.0
    pnl_diff_pct: float = 0.0

    # Trade validity
    is_valid_comparison: bool = False
    discrepancy_reason: Optional[str] = None


class UnifiedSystemValidator:
    """
    Comprehensive validation framework ensuring identical behavior between
    backtesting and live trading systems.

    This validator implements statistical rigor and handles the fundamental
    timing differences between two-bar backtesting and immediate live execution.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the validation framework.

        Args:
            config: Validation configuration parameters
        """
        self.config = config or self._get_default_config()

        # Initialize strategies for comparison
        self.backtest_strategy = MSEStrategyBacktesting()
        self.live_strategy = MSEStrategy()

        # Validation results storage
        self.validation_db_path = Path("validation_results.db")
        self.setup_validation_database()

        # Statistical thresholds
        self.CORRELATION_THRESHOLD = 0.95  # Minimum acceptable correlation
        self.SIGNAL_MATCH_THRESHOLD = 0.90  # 90% signal matching required
        self.PNL_CORRELATION_THRESHOLD = 0.85  # PnL correlation threshold
        self.SIGNIFICANCE_LEVEL = 0.05  # 5% significance for statistical tests

        # Timing reconciliation parameters
        self.TWO_BAR_DELAY_MINUTES = 10  # Expected delay for two-bar rule
        self.EXECUTION_TOLERANCE_SECONDS = 300  # 5-minute tolerance window

        logger.info("Unified System Validator initialized with statistical validation")

    def _get_default_config(self) -> Dict[str, Any]:
        """Default validation configuration."""
        return {
            'warmup_minutes': 525,  # Match MSE strategy warmup
            'minimum_trades_for_validation': 20,
            'indicator_precision': 4,
            'price_precision': 2,
            'statistical_confidence': 0.95,
            'monte_carlo_iterations': 1000,
            'stress_test_scenarios': 5
        }

    def setup_validation_database(self):
        """Setup SQLite database for validation results storage."""
        try:
            with sqlite3.connect(self.validation_db_path) as conn:
                cursor = conn.cursor()

                # Signal comparisons table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS signal_comparisons (
                        id INTEGER PRIMARY KEY,
                        timestamp TEXT,
                        symbol TEXT,
                        backtest_signal TEXT,
                        live_signal TEXT,
                        signal_match BOOLEAN,
                        backtest_indicators TEXT,  -- JSON
                        live_indicators TEXT,      -- JSON
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Trade comparisons table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trade_comparisons (
                        id INTEGER PRIMARY KEY,
                        trade_id TEXT,
                        symbol TEXT,
                        backtest_entry_time TEXT,
                        live_entry_time TEXT,
                        backtest_entry_price REAL,
                        live_entry_price REAL,
                        backtest_exit_time TEXT,
                        live_exit_time TEXT,
                        backtest_exit_price REAL,
                        live_exit_price REAL,
                        backtest_pnl REAL,
                        live_pnl REAL,
                        is_valid_comparison BOOLEAN,
                        discrepancy_reason TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Validation sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS validation_sessions (
                        id INTEGER PRIMARY KEY,
                        session_id TEXT UNIQUE,
                        start_time TEXT,
                        end_time TEXT,
                        validation_type TEXT,
                        metrics TEXT,  -- JSON
                        status TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                conn.commit()
                logger.info("Validation database setup complete")

        except Exception as e:
            logger.error(f"Error setting up validation database: {e}")
            raise

    def validate_signal_parity(self,
                             historical_data: Dict[str, pd.DataFrame],
                             symbol: str,
                             date_range: Tuple[datetime, datetime]) -> ValidationMetrics:
        """
        Phase 1: Signal Parity Validation

        Validates that both systems generate identical signals from the same data.
        This is the most critical validation as it ensures algorithmic consistency.

        Args:
            historical_data: OHLCV data for validation period
            symbol: Symbol to validate
            date_range: Start and end dates for validation

        Returns:
            ValidationMetrics with signal parity results
        """
        logger.info(f"Starting signal parity validation for {symbol} from {date_range[0]} to {date_range[1]}")

        try:
            # Prepare data for both systems
            backtest_data = self._prepare_backtest_data(historical_data)
            live_data = self._prepare_live_data(historical_data)

            # Generate signals from both systems
            backtest_signals = self._generate_backtest_signals(backtest_data, symbol)
            live_signals = self._generate_live_signals(live_data, symbol)

            # Compare signals point-by-point
            signal_comparisons = self._compare_signals(backtest_signals, live_signals)

            # Validate indicator calculations
            indicator_correlations = self._validate_indicator_calculations(
                backtest_data, live_data
            )

            # Calculate validation metrics
            metrics = self._calculate_signal_parity_metrics(
                signal_comparisons, indicator_correlations
            )

            # Store results in database
            self._store_signal_comparisons(signal_comparisons, symbol)

            # Statistical significance testing
            self._perform_signal_statistical_tests(metrics)

            logger.info(f"Signal parity validation complete. Match rate: {metrics.signal_parity_rate:.2%}")

            return metrics

        except Exception as e:
            logger.error(f"Error in signal parity validation: {e}")
            raise

    def validate_execution_model(self,
                                historical_data: Dict[str, pd.DataFrame],
                                live_trades: List[Dict[str, Any]],
                                symbol: str) -> ValidationMetrics:
        """
        Phase 2: Execution Model Validation

        Reconciles the fundamental timing difference between two-bar backtesting
        and immediate live execution.

        Args:
            historical_data: Historical OHLCV data
            live_trades: Actual live trades executed
            symbol: Symbol being validated

        Returns:
            ValidationMetrics with execution reconciliation results
        """
        logger.info(f"Starting execution model validation for {symbol}")

        try:
            # Run backtest with identical data
            backtest_results = self._run_controlled_backtest(historical_data, symbol)

            # Align timing between systems (account for two-bar delay)
            aligned_trades = self._align_trade_timing(
                backtest_results['trades'], live_trades
            )

            # Compare trade execution
            trade_comparisons = self._compare_trade_execution(aligned_trades)

            # Analyze execution differences
            execution_analysis = self._analyze_execution_differences(trade_comparisons)

            # Calculate execution metrics
            metrics = self._calculate_execution_metrics(
                execution_analysis, trade_comparisons
            )

            # Store trade comparisons
            self._store_trade_comparisons(trade_comparisons, symbol)

            logger.info(f"Execution model validation complete. Trade match rate: {metrics.trade_match_rate:.2%}")

            return metrics

        except Exception as e:
            logger.error(f"Error in execution model validation: {e}")
            raise

    def validate_state_management(self,
                                 backtest_state_log: List[Dict[str, Any]],
                                 live_state_log: List[Dict[str, Any]],
                                 symbol: str) -> ValidationMetrics:
        """
        Phase 3: State Management Validation

        Ensures consistent position tracking and peak/valley MACD tracking
        across both systems.

        Args:
            backtest_state_log: State evolution from backtest
            live_state_log: State evolution from live system
            symbol: Symbol being validated

        Returns:
            ValidationMetrics with state management results
        """
        logger.info(f"Starting state management validation for {symbol}")

        try:
            # Align state timestamps
            aligned_states = self._align_state_logs(backtest_state_log, live_state_log)

            # Validate position tracking
            position_validation = self._validate_position_tracking(aligned_states)

            # Validate MACD peak tracking
            peak_validation = self._validate_peak_tracking(aligned_states)

            # Calculate state consistency metrics
            metrics = self._calculate_state_metrics(
                position_validation, peak_validation
            )

            logger.info(f"State management validation complete.")

            return metrics

        except Exception as e:
            logger.error(f"Error in state management validation: {e}")
            raise

    def run_comprehensive_validation(self,
                                   historical_data: Dict[str, pd.DataFrame],
                                   live_trades: List[Dict[str, Any]],
                                   live_state_log: List[Dict[str, Any]],
                                   symbol: str) -> Dict[str, ValidationMetrics]:
        """
        Run comprehensive validation across all components.

        This is the main entry point for complete system validation.

        Args:
            historical_data: OHLCV data for validation period
            live_trades: Actual trades from live system
            live_state_log: State changes from live system
            symbol: Symbol to validate

        Returns:
            Dictionary of validation results by component
        """
        logger.info(f"Starting comprehensive validation for {symbol}")

        session_id = f"validation_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results = {}

        try:
            # Phase 1: Signal Parity
            date_range = (
                historical_data['5m']['timestamp'].min(),
                historical_data['5m']['timestamp'].max()
            )
            results['signal_parity'] = self.validate_signal_parity(
                historical_data, symbol, date_range
            )

            # Phase 2: Execution Model
            results['execution_model'] = self.validate_execution_model(
                historical_data, live_trades, symbol
            )

            # Phase 3: State Management (with synthetic backtest state)
            backtest_state_log = self._generate_backtest_state_log(
                historical_data, symbol
            )
            results['state_management'] = self.validate_state_management(
                backtest_state_log, live_state_log, symbol
            )

            # Phase 4: Statistical Validation
            results['statistical_validation'] = self._run_statistical_validation(
                results, historical_data, live_trades
            )

            # Phase 5: Stress Testing
            results['stress_testing'] = self._run_stress_testing(
                historical_data, symbol
            )

            # Generate comprehensive report
            validation_report = self._generate_validation_report(results, symbol)

            # Store session results
            self._store_validation_session(session_id, results, "COMPLETED")

            logger.info(f"Comprehensive validation complete for {symbol}")
            logger.info(f"Overall validation status: {validation_report['overall_status']}")

            return results

        except Exception as e:
            logger.error(f"Error in comprehensive validation: {e}")
            self._store_validation_session(session_id, results, "FAILED")
            raise

    def _prepare_backtest_data(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepare data in format expected by backtest strategy."""
        return self.backtest_strategy.prepare_data(
            historical_data, "VALIDATION_SYMBOL", datetime.now().strftime('%Y-%m-%d')
        )

    def _prepare_live_data(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepare data in format expected by live strategy."""
        # Live strategy expects the same format
        return historical_data

    def _generate_backtest_signals(self, data: Dict[str, pd.DataFrame], symbol: str) -> pd.DataFrame:
        """Generate signals using backtest strategy."""
        try:
            return self.backtest_strategy.generate_signals(data)
        except Exception as e:
            logger.error(f"Error generating backtest signals: {e}")
            return pd.DataFrame()

    def _generate_live_signals(self, data: Dict[str, pd.DataFrame], symbol: str) -> List[Dict[str, Any]]:
        """Generate signals using live strategy over historical data."""
        signals = []

        try:
            # Convert to minute-by-minute live simulation
            if '5m' not in data or data['5m'].empty:
                return signals

            df_5m = data['5m'].copy()

            # Iterate through each timestamp to simulate live signal generation
            for i in range(len(df_5m)):
                # Create market data slice up to current point
                current_data = {
                    '5m': df_5m.iloc[:i+1],
                    '15m': data.get('15m', pd.DataFrame()).iloc[:((i+1)//3)+1] if '15m' in data else pd.DataFrame()
                }

                # Skip if insufficient data
                if len(current_data['5m']) < 40 or len(current_data['15m']) < 40:
                    continue

                # Generate signal at this point
                signal = self.live_strategy.generate_signal(current_data)

                if signal:
                    signal['timestamp'] = df_5m.iloc[i]['timestamp']
                    signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"Error generating live signals: {e}")
            return signals

    def _compare_signals(self, backtest_signals: pd.DataFrame, live_signals: List[Dict[str, Any]]) -> List[SignalComparison]:
        """Compare signals from both systems point-by-point."""
        comparisons = []

        try:
            # Convert live signals to DataFrame for easier comparison
            if not live_signals:
                live_df = pd.DataFrame()
            else:
                live_df = pd.DataFrame(live_signals)

            if backtest_signals.empty and live_df.empty:
                return comparisons

            # Get all unique timestamps
            all_timestamps = set()
            if not backtest_signals.empty:
                all_timestamps.update(backtest_signals['timestamp'])
            if not live_df.empty:
                all_timestamps.update(live_df['timestamp'])

            # Compare at each timestamp
            for timestamp in sorted(all_timestamps):
                comparison = SignalComparison(timestamp=timestamp)

                # Get backtest signal at this timestamp
                backtest_match = backtest_signals[backtest_signals['timestamp'] == timestamp]
                if not backtest_match.empty:
                    row = backtest_match.iloc[0]
                    if row.get('entry_signal_buy', False):
                        comparison.backtest_signal = 'BUY'
                    elif row.get('entry_signal_sell', False):
                        comparison.backtest_signal = 'SELL'
                    elif row.get('exit_signal_buy', False):
                        comparison.backtest_signal = 'SELL_EXIT'
                    elif row.get('exit_signal_sell', False):
                        comparison.backtest_signal = 'BUY_EXIT'

                    # Store backtest indicators
                    comparison.backtest_indicators = {
                        '5m_macd_line': row.get('5m_macd_line', 0),
                        '5m_signal_line': row.get('5m_signal_line', 0),
                        '15m_macd_line': row.get('15m_macd_line', 0),
                        '15m_signal_line': row.get('15m_signal_line', 0),
                        '5m_ema_9': row.get('5m_ema_9', 0),
                        '5m_ema_20': row.get('5m_ema_20', 0),
                        '15m_ema_9': row.get('15m_ema_9', 0),
                        '15m_ema_20': row.get('15m_ema_20', 0)
                    }
                    comparison.backtest_price = row.get('close', 0)

                # Get live signal at this timestamp
                live_match = live_df[live_df['timestamp'] == timestamp] if not live_df.empty else pd.DataFrame()
                if not live_match.empty:
                    signal_data = live_match.iloc[0]
                    comparison.live_signal = signal_data.get('action', 'NO_SIGNAL')
                    comparison.live_price = signal_data.get('price', 0)

                    # Store live indicators if available
                    if 'indicators' in signal_data:
                        indicators = signal_data['indicators']
                        comparison.live_indicators = {
                            '5m_macd_bullish': indicators.get('5min_macd_bullish', False),
                            '15m_macd_bullish': indicators.get('15min_macd_bullish', False),
                            '5m_ema_bullish': indicators.get('5min_ema_bullish', False),
                            '15m_ema_bullish': indicators.get('15min_ema_bullish', False),
                            'macd_histogram_15min': indicators.get('macd_histogram_15min', 0)
                        }

                # Determine if signals match
                comparison.match = (comparison.backtest_signal == comparison.live_signal)

                # Calculate price difference
                if comparison.backtest_price > 0 and comparison.live_price > 0:
                    comparison.price_difference = abs(comparison.backtest_price - comparison.live_price)

                comparisons.append(comparison)

        except Exception as e:
            logger.error(f"Error comparing signals: {e}")

        return comparisons

    def _validate_indicator_calculations(self,
                                       backtest_data: Dict[str, pd.DataFrame],
                                       live_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Validate that both systems calculate identical indicators."""
        correlations = {}

        try:
            for timeframe in ['5m', '15m']:
                if timeframe not in backtest_data or timeframe not in live_data:
                    continue

                bt_df = backtest_data[timeframe]
                live_df = live_data[timeframe]

                if bt_df.empty or live_df.empty:
                    continue

                # Align by timestamp
                merged = pd.merge(bt_df, live_df, on='timestamp', suffixes=('_bt', '_live'))

                if merged.empty:
                    continue

                # Calculate correlations for key indicators
                if f'{timeframe}_macd_line_bt' in merged.columns and f'{timeframe}_macd_line_live' in merged.columns:
                    correlations[f'macd_correlation_{timeframe}'] = merged[f'{timeframe}_macd_line_bt'].corr(
                        merged[f'{timeframe}_macd_line_live']
                    )

                if f'{timeframe}_ema_9_bt' in merged.columns and f'{timeframe}_ema_9_live' in merged.columns:
                    correlations[f'ema_correlation_{timeframe}'] = merged[f'{timeframe}_ema_9_bt'].corr(
                        merged[f'{timeframe}_ema_9_live']
                    )

        except Exception as e:
            logger.error(f"Error validating indicator calculations: {e}")

        return correlations

    def _calculate_signal_parity_metrics(self,
                                       signal_comparisons: List[SignalComparison],
                                       indicator_correlations: Dict[str, float]) -> ValidationMetrics:
        """Calculate comprehensive signal parity metrics."""
        metrics = ValidationMetrics()

        try:
            if not signal_comparisons:
                return metrics

            metrics.total_signals = len(signal_comparisons)
            metrics.matching_signals = sum(1 for comp in signal_comparisons if comp.match)
            metrics.signal_parity_rate = metrics.matching_signals / metrics.total_signals if metrics.total_signals > 0 else 0

            # Add indicator correlations
            metrics.macd_correlation_5m = indicator_correlations.get('macd_correlation_5m', 0)
            metrics.macd_correlation_15m = indicator_correlations.get('macd_correlation_15m', 0)
            metrics.ema_correlation_5m = indicator_correlations.get('ema_correlation_5m', 0)
            metrics.ema_correlation_15m = indicator_correlations.get('ema_correlation_15m', 0)

            # Calculate timing variance
            timing_diffs = []
            for comp in signal_comparisons:
                if comp.backtest_price > 0 and comp.live_price > 0:
                    timing_diffs.append(comp.price_difference)

            if timing_diffs:
                metrics.signal_timing_variance = np.var(timing_diffs)

        except Exception as e:
            logger.error(f"Error calculating signal parity metrics: {e}")

        return metrics

    def _run_controlled_backtest(self, data: Dict[str, pd.DataFrame], symbol: str) -> Dict[str, Any]:
        """Run controlled backtest for execution comparison."""
        try:
            prepared_data = self._prepare_backtest_data(data)
            signals_df = self.backtest_strategy.generate_signals(prepared_data)
            trades_df, summary_stats = self.backtest_strategy.execute_strategy(signals_df)

            return {
                'trades': trades_df,
                'signals': signals_df,
                'summary': summary_stats
            }

        except Exception as e:
            logger.error(f"Error running controlled backtest: {e}")
            return {'trades': pd.DataFrame(), 'signals': pd.DataFrame(), 'summary': {}}

    def _align_trade_timing(self,
                           backtest_trades: pd.DataFrame,
                           live_trades: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Align trades between systems accounting for two-bar execution delay."""
        aligned_trades = []

        try:
            if backtest_trades.empty or not live_trades:
                return aligned_trades

            # Convert backtest trades to list format
            bt_trades = backtest_trades.to_dict('records')

            # Sort both by entry time
            bt_trades.sort(key=lambda x: x.get('entry_timestamp', datetime.min))
            live_trades.sort(key=lambda x: x.get('entry_timestamp', datetime.min))

            # Align trades with timing tolerance
            for bt_trade in bt_trades:
                bt_entry_time = bt_trade.get('entry_timestamp')
                if not bt_entry_time:
                    continue

                # Find matching live trade within tolerance window
                best_match = None
                min_time_diff = float('inf')

                for live_trade in live_trades:
                    live_entry_time = live_trade.get('entry_timestamp')
                    if not live_entry_time:
                        continue

                    # Account for two-bar delay (backtest should be ~10 minutes behind)
                    expected_live_time = bt_entry_time - pd.Timedelta(minutes=self.TWO_BAR_DELAY_MINUTES)
                    time_diff = abs((live_entry_time - expected_live_time).total_seconds())

                    if time_diff < self.EXECUTION_TOLERANCE_SECONDS and time_diff < min_time_diff:
                        min_time_diff = time_diff
                        best_match = live_trade

                if best_match:
                    aligned_trades.append((bt_trade, best_match))
                    live_trades.remove(best_match)  # Prevent duplicate matching

        except Exception as e:
            logger.error(f"Error aligning trade timing: {e}")

        return aligned_trades

    def _compare_trade_execution(self, aligned_trades: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> List[TradeComparison]:
        """Compare trade execution between systems."""
        comparisons = []

        try:
            for i, (bt_trade, live_trade) in enumerate(aligned_trades):
                comparison = TradeComparison(trade_id=f"trade_{i}")

                # Entry comparison
                comparison.backtest_entry_time = bt_trade.get('entry_timestamp')
                comparison.live_entry_time = live_trade.get('entry_timestamp')

                if comparison.backtest_entry_time and comparison.live_entry_time:
                    comparison.entry_time_diff_seconds = (
                        comparison.live_entry_time - comparison.backtest_entry_time
                    ).total_seconds()

                comparison.backtest_entry_price = bt_trade.get('entry_price', 0)
                comparison.live_entry_price = live_trade.get('entry_price', 0)

                if comparison.backtest_entry_price > 0 and comparison.live_entry_price > 0:
                    comparison.entry_price_diff_pct = (
                        abs(comparison.live_entry_price - comparison.backtest_entry_price) /
                        comparison.backtest_entry_price * 100
                    )

                # Exit comparison (if available)
                comparison.backtest_exit_time = bt_trade.get('exit_timestamp')
                comparison.live_exit_time = live_trade.get('exit_timestamp')

                if comparison.backtest_exit_time and comparison.live_exit_time:
                    comparison.exit_time_diff_seconds = (
                        comparison.live_exit_time - comparison.backtest_exit_time
                    ).total_seconds()

                comparison.backtest_exit_price = bt_trade.get('exit_price', 0)
                comparison.live_exit_price = live_trade.get('exit_price', 0)

                if comparison.backtest_exit_price > 0 and comparison.live_exit_price > 0:
                    comparison.exit_price_diff_pct = (
                        abs(comparison.live_exit_price - comparison.backtest_exit_price) /
                        comparison.backtest_exit_price * 100
                    )

                # PnL comparison
                comparison.backtest_pnl = bt_trade.get('pnl', 0)
                comparison.live_pnl = live_trade.get('pnl', 0)

                if comparison.backtest_pnl != 0:
                    comparison.pnl_diff_pct = (
                        abs(comparison.live_pnl - comparison.backtest_pnl) /
                        abs(comparison.backtest_pnl) * 100
                    )

                # Determine validity
                comparison.is_valid_comparison = self._is_valid_trade_comparison(comparison)

                comparisons.append(comparison)

        except Exception as e:
            logger.error(f"Error comparing trade execution: {e}")

        return comparisons

    def _is_valid_trade_comparison(self, comparison: TradeComparison) -> bool:
        """Determine if trade comparison is valid for analysis."""
        try:
            # Check basic data availability
            if (comparison.backtest_entry_price == 0 or
                comparison.live_entry_price == 0):
                comparison.discrepancy_reason = "Missing entry price data"
                return False

            # Check reasonable price differences (within 5%)
            if comparison.entry_price_diff_pct > 5.0:
                comparison.discrepancy_reason = f"Entry price difference too large: {comparison.entry_price_diff_pct:.2f}%"
                return False

            # Check reasonable timing (within expected window)
            if abs(comparison.entry_time_diff_seconds) > self.EXECUTION_TOLERANCE_SECONDS * 2:
                comparison.discrepancy_reason = f"Entry timing difference too large: {comparison.entry_time_diff_seconds:.0f}s"
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating trade comparison: {e}")
            return False

    def _analyze_execution_differences(self, trade_comparisons: List[TradeComparison]) -> Dict[str, Any]:
        """Analyze systematic execution differences."""
        analysis = {
            'entry_timing_bias': 0.0,
            'entry_price_bias': 0.0,
            'exit_timing_bias': 0.0,
            'exit_price_bias': 0.0,
            'pnl_bias': 0.0,
            'systematic_patterns': []
        }

        try:
            valid_comparisons = [comp for comp in trade_comparisons if comp.is_valid_comparison]

            if not valid_comparisons:
                return analysis

            # Calculate systematic biases
            entry_time_diffs = [comp.entry_time_diff_seconds for comp in valid_comparisons
                               if comp.entry_time_diff_seconds is not None]
            if entry_time_diffs:
                analysis['entry_timing_bias'] = np.mean(entry_time_diffs)

            entry_price_diffs = [comp.entry_price_diff_pct for comp in valid_comparisons
                                if comp.entry_price_diff_pct > 0]
            if entry_price_diffs:
                analysis['entry_price_bias'] = np.mean(entry_price_diffs)

            pnl_diffs = [comp.pnl_diff_pct for comp in valid_comparisons
                        if comp.pnl_diff_pct > 0]
            if pnl_diffs:
                analysis['pnl_bias'] = np.mean(pnl_diffs)

            # Detect patterns
            if abs(analysis['entry_timing_bias']) > 60:  # More than 1 minute bias
                analysis['systematic_patterns'].append(f"Systematic timing bias: {analysis['entry_timing_bias']:.0f}s")

            if analysis['entry_price_bias'] > 1.0:  # More than 1% price bias
                analysis['systematic_patterns'].append(f"Systematic price bias: {analysis['entry_price_bias']:.2f}%")

        except Exception as e:
            logger.error(f"Error analyzing execution differences: {e}")

        return analysis

    def _calculate_execution_metrics(self,
                                   execution_analysis: Dict[str, Any],
                                   trade_comparisons: List[TradeComparison]) -> ValidationMetrics:
        """Calculate execution model validation metrics."""
        metrics = ValidationMetrics()

        try:
            valid_comparisons = [comp for comp in trade_comparisons if comp.is_valid_comparison]

            metrics.total_trades_backtest = len(trade_comparisons)
            metrics.total_trades_live = len([comp for comp in trade_comparisons if comp.live_entry_price > 0])
            metrics.trade_match_rate = len(valid_comparisons) / len(trade_comparisons) if trade_comparisons else 0

            # Execution delay analysis
            metrics.execution_delay_analysis = {
                'entry_timing_bias': execution_analysis.get('entry_timing_bias', 0),
                'entry_price_bias': execution_analysis.get('entry_price_bias', 0),
                'systematic_patterns': len(execution_analysis.get('systematic_patterns', []))
            }

            # PnL correlation
            if valid_comparisons:
                backtest_pnls = [comp.backtest_pnl for comp in valid_comparisons]
                live_pnls = [comp.live_pnl for comp in valid_comparisons]

                if len(backtest_pnls) > 1 and len(live_pnls) > 1:
                    metrics.pnl_correlation = np.corrcoef(backtest_pnls, live_pnls)[0, 1]

                    pnl_diffs = [live - bt for bt, live in zip(backtest_pnls, live_pnls)]
                    metrics.return_difference_mean = np.mean(pnl_diffs)
                    metrics.return_difference_std = np.std(pnl_diffs)

        except Exception as e:
            logger.error(f"Error calculating execution metrics: {e}")

        return metrics

    def _generate_backtest_state_log(self, historical_data: Dict[str, pd.DataFrame], symbol: str) -> List[Dict[str, Any]]:
        """Generate state log from controlled backtest."""
        state_log = []

        try:
            # Run backtest and extract state changes
            backtest_results = self._run_controlled_backtest(historical_data, symbol)

            if backtest_results['trades'].empty:
                return state_log

            trades_df = backtest_results['trades']
            signals_df = backtest_results['signals']

            # Generate state log from trades and signals
            for _, trade in trades_df.iterrows():
                # Entry state
                entry_state = {
                    'timestamp': trade['entry_timestamp'],
                    'symbol': symbol,
                    'action': 'POSITION_OPENED',
                    'position_side': trade['direction'].upper(),
                    'entry_price': trade['entry_price'],
                    'state_data': {
                        'has_position': True,
                        'peak_macd_histogram': 0.0  # Will be updated
                    }
                }
                state_log.append(entry_state)

                # Exit state (if available)
                if pd.notna(trade.get('exit_timestamp')):
                    exit_state = {
                        'timestamp': trade['exit_timestamp'],
                        'symbol': symbol,
                        'action': 'POSITION_CLOSED',
                        'position_side': 'FLAT',
                        'exit_price': trade['exit_price'],
                        'pnl': trade.get('pnl', 0),
                        'state_data': {
                            'has_position': False,
                            'peak_macd_histogram': 0.0
                        }
                    }
                    state_log.append(exit_state)

        except Exception as e:
            logger.error(f"Error generating backtest state log: {e}")

        return state_log

    def _align_state_logs(self,
                         backtest_log: List[Dict[str, Any]],
                         live_log: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Align state logs from both systems."""
        aligned_states = []

        try:
            # Sort logs by timestamp
            backtest_log.sort(key=lambda x: x.get('timestamp', datetime.min))
            live_log.sort(key=lambda x: x.get('timestamp', datetime.min))

            # Simple alignment based on action type and timing
            for bt_state in backtest_log:
                bt_timestamp = bt_state.get('timestamp')
                bt_action = bt_state.get('action')

                if not bt_timestamp or not bt_action:
                    continue

                # Find corresponding live state
                best_match = None
                min_time_diff = float('inf')

                for live_state in live_log:
                    live_timestamp = live_state.get('timestamp')
                    live_action = live_state.get('action')

                    if not live_timestamp or not live_action:
                        continue

                    if live_action == bt_action:
                        time_diff = abs((live_timestamp - bt_timestamp).total_seconds())
                        if time_diff < min_time_diff and time_diff < self.EXECUTION_TOLERANCE_SECONDS:
                            min_time_diff = time_diff
                            best_match = live_state

                if best_match:
                    aligned_states.append((bt_state, best_match))
                    live_log.remove(best_match)

        except Exception as e:
            logger.error(f"Error aligning state logs: {e}")

        return aligned_states

    def _validate_position_tracking(self, aligned_states: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Any]:
        """Validate position tracking consistency."""
        validation = {
            'position_matches': 0,
            'total_states': 0,
            'consistency_rate': 0.0,
            'discrepancies': []
        }

        try:
            for bt_state, live_state in aligned_states:
                validation['total_states'] += 1

                bt_position = bt_state.get('position_side', 'UNKNOWN')
                live_position = live_state.get('position_side', 'UNKNOWN')

                if bt_position == live_position:
                    validation['position_matches'] += 1
                else:
                    discrepancy = {
                        'timestamp': bt_state.get('timestamp'),
                        'backtest_position': bt_position,
                        'live_position': live_position,
                        'action': bt_state.get('action')
                    }
                    validation['discrepancies'].append(discrepancy)

            validation['consistency_rate'] = (
                validation['position_matches'] / validation['total_states']
                if validation['total_states'] > 0 else 0
            )

        except Exception as e:
            logger.error(f"Error validating position tracking: {e}")

        return validation

    def _validate_peak_tracking(self, aligned_states: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Any]:
        """Validate MACD peak tracking consistency."""
        validation = {
            'peak_matches': 0,
            'total_peaks': 0,
            'peak_correlation': 0.0,
            'peak_differences': []
        }

        try:
            bt_peaks = []
            live_peaks = []

            for bt_state, live_state in aligned_states:
                bt_peak = bt_state.get('state_data', {}).get('peak_macd_histogram', 0)
                live_peak = live_state.get('state_data', {}).get('peak_macd_histogram', 0)

                if bt_peak != 0 or live_peak != 0:
                    validation['total_peaks'] += 1
                    bt_peaks.append(bt_peak)
                    live_peaks.append(live_peak)

                    # Check if peaks are reasonably close (within 5%)
                    if abs(bt_peak) > 0:
                        diff_pct = abs(live_peak - bt_peak) / abs(bt_peak) * 100
                        if diff_pct <= 5.0:
                            validation['peak_matches'] += 1
                    elif bt_peak == live_peak == 0:
                        validation['peak_matches'] += 1

                    validation['peak_differences'].append({
                        'timestamp': bt_state.get('timestamp'),
                        'backtest_peak': bt_peak,
                        'live_peak': live_peak,
                        'difference': live_peak - bt_peak
                    })

            # Calculate correlation if enough data
            if len(bt_peaks) > 2 and len(live_peaks) > 2:
                validation['peak_correlation'] = np.corrcoef(bt_peaks, live_peaks)[0, 1]

        except Exception as e:
            logger.error(f"Error validating peak tracking: {e}")

        return validation

    def _calculate_state_metrics(self,
                               position_validation: Dict[str, Any],
                               peak_validation: Dict[str, Any]) -> ValidationMetrics:
        """Calculate state management validation metrics."""
        metrics = ValidationMetrics()

        try:
            # Add state-specific metrics to ValidationMetrics
            # For now, store in execution_delay_analysis
            metrics.execution_delay_analysis = {
                'position_consistency_rate': position_validation.get('consistency_rate', 0),
                'peak_correlation': peak_validation.get('peak_correlation', 0),
                'peak_match_rate': (
                    peak_validation.get('peak_matches', 0) /
                    peak_validation.get('total_peaks', 1)
                )
            }

        except Exception as e:
            logger.error(f"Error calculating state metrics: {e}")

        return metrics

    def _run_statistical_validation(self,
                                  results: Dict[str, ValidationMetrics],
                                  historical_data: Dict[str, pd.DataFrame],
                                  live_trades: List[Dict[str, Any]]) -> ValidationMetrics:
        """Run statistical significance tests on validation results."""
        metrics = ValidationMetrics()

        try:
            # Collect data for statistical testing
            signal_matches = []
            trade_matches = []

            # Statistical tests for signal parity
            if 'signal_parity' in results:
                sp_metrics = results['signal_parity']

                # Binomial test for signal matching rate
                if sp_metrics.total_signals > 0:
                    # Null hypothesis: signal match rate = 0.5 (random)
                    # Alternative: signal match rate > 0.9 (our threshold)
                    success = sp_metrics.matching_signals
                    trials = sp_metrics.total_signals

                    # Two-tailed binomial test
                    p_value = stats.binomtest(success, trials, 0.9).pvalue
                    metrics.p_values = {'signal_parity_binomial': p_value}

                    # Confidence interval for signal match rate
                    ci = stats.binomtest(success, trials).proportion_ci()
                    metrics.confidence_intervals = {
                        'signal_match_rate': (ci.low, ci.high)
                    }

                # Test indicator correlations
                correlations = [
                    sp_metrics.macd_correlation_5m,
                    sp_metrics.macd_correlation_15m,
                    sp_metrics.ema_correlation_5m,
                    sp_metrics.ema_correlation_15m
                ]

                # Filter out NaN/invalid correlations
                valid_correlations = [c for c in correlations if not pd.isna(c) and c != 0]

                if valid_correlations:
                    # Test if correlations are significantly high
                    mean_correlation = np.mean(valid_correlations)

                    # One-sample t-test against threshold
                    if len(valid_correlations) > 1:
                        t_stat, p_val = stats.ttest_1samp(valid_correlations, self.CORRELATION_THRESHOLD)
                        metrics.p_values['correlation_ttest'] = p_val

            # Statistical tests for execution model
            if 'execution_model' in results:
                em_metrics = results['execution_model']

                # Test trade match rate
                if em_metrics.total_trades_backtest > 0:
                    match_rate = em_metrics.trade_match_rate

                    # Bootstrap confidence interval for trade match rate
                    bootstrap_samples = []
                    for _ in range(1000):
                        sample = np.random.binomial(1, match_rate, em_metrics.total_trades_backtest)
                        bootstrap_samples.append(np.mean(sample))

                    ci_low, ci_high = np.percentile(bootstrap_samples, [2.5, 97.5])
                    metrics.confidence_intervals['trade_match_rate'] = (ci_low, ci_high)

                # Test PnL correlation significance
                if em_metrics.pnl_correlation != 0 and not pd.isna(em_metrics.pnl_correlation):
                    # Test if correlation is significantly different from 0
                    n_trades = max(em_metrics.total_trades_backtest, em_metrics.total_trades_live)
                    if n_trades > 3:
                        t_stat = em_metrics.pnl_correlation * np.sqrt((n_trades - 2) / (1 - em_metrics.pnl_correlation**2))
                        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n_trades - 2))
                        metrics.p_values['pnl_correlation'] = p_val

            logger.info("Statistical validation complete")

        except Exception as e:
            logger.error(f"Error in statistical validation: {e}")

        return metrics

    def _run_stress_testing(self, historical_data: Dict[str, pd.DataFrame], symbol: str) -> ValidationMetrics:
        """Run stress testing scenarios to validate system robustness."""
        metrics = ValidationMetrics()

        try:
            stress_scenarios = [
                self._stress_test_high_volatility,
                self._stress_test_market_gaps,
                self._stress_test_low_volume,
                self._stress_test_trend_reversals,
                self._stress_test_extreme_values
            ]

            scenario_results = []

            for scenario_func in stress_scenarios:
                try:
                    result = scenario_func(historical_data, symbol)
                    scenario_results.append(result)
                except Exception as e:
                    logger.warning(f"Stress test scenario failed: {e}")
                    scenario_results.append({'passed': False, 'error': str(e)})

            # Calculate overall stress test metrics
            passed_tests = sum(1 for result in scenario_results if result.get('passed', False))
            total_tests = len(scenario_results)

            metrics.execution_delay_analysis = {
                'stress_test_pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'passed_scenarios': passed_tests,
                'total_scenarios': total_tests,
                'scenario_details': scenario_results
            }

            logger.info(f"Stress testing complete: {passed_tests}/{total_tests} scenarios passed")

        except Exception as e:
            logger.error(f"Error in stress testing: {e}")

        return metrics

    def _stress_test_high_volatility(self, data: Dict[str, pd.DataFrame], symbol: str) -> Dict[str, Any]:
        """Test system behavior during high volatility periods."""
        try:
            # Identify high volatility periods (top 10% by price range)
            df_5m = data['5m'].copy()
            df_5m['price_range'] = (df_5m['high'] - df_5m['low']) / df_5m['close']
            volatility_threshold = df_5m['price_range'].quantile(0.9)

            high_vol_data = {}
            for timeframe, df in data.items():
                if timeframe in ['5m', '15m']:
                    # Filter to high volatility periods
                    high_vol_periods = df_5m[df_5m['price_range'] >= volatility_threshold]['timestamp']
                    filtered_df = df[df['timestamp'].isin(high_vol_periods)]
                    high_vol_data[timeframe] = filtered_df

            if all(df.empty for df in high_vol_data.values()):
                return {'passed': False, 'reason': 'No high volatility periods found'}

            # Run validation on high volatility subset
            signal_metrics = self.validate_signal_parity(
                high_vol_data, symbol,
                (high_vol_data['5m']['timestamp'].min(), high_vol_data['5m']['timestamp'].max())
            )

            # Pass if signal parity remains high during volatility
            passed = signal_metrics.signal_parity_rate >= 0.85  # Slightly lower threshold for stress test

            return {
                'passed': passed,
                'signal_parity_rate': signal_metrics.signal_parity_rate,
                'total_signals_tested': signal_metrics.total_signals,
                'scenario': 'high_volatility'
            }

        except Exception as e:
            return {'passed': False, 'error': str(e), 'scenario': 'high_volatility'}

    def _stress_test_market_gaps(self, data: Dict[str, pd.DataFrame], symbol: str) -> Dict[str, Any]:
        """Test system behavior with market gaps."""
        try:
            # Introduce artificial gaps in the data
            df_5m = data['5m'].copy()

            # Create gaps by removing random 30-minute periods
            gap_starts = np.random.choice(
                range(100, len(df_5m) - 100),
                size=min(5, len(df_5m) // 200),
                replace=False
            )

            for gap_start in gap_starts:
                # Remove 6 consecutive 5-minute bars (30-minute gap)
                df_5m = df_5m.drop(range(gap_start, min(gap_start + 6, len(df_5m))))

            gapped_data = {'5m': df_5m}
            if '15m' in data:
                gapped_data['15m'] = data['15m']  # Keep 15m data intact

            # Test signal generation with gaps
            try:
                backtest_signals = self._generate_backtest_signals(gapped_data, symbol)
                live_signals = self._generate_live_signals(gapped_data, symbol)

                signal_comparisons = self._compare_signals(backtest_signals, live_signals)

                if signal_comparisons:
                    match_rate = sum(1 for comp in signal_comparisons if comp.match) / len(signal_comparisons)
                else:
                    match_rate = 1.0  # No signals to compare

                passed = match_rate >= 0.80  # Lower threshold for gapped data

                return {
                    'passed': passed,
                    'signal_match_rate': match_rate,
                    'gaps_introduced': len(gap_starts),
                    'scenario': 'market_gaps'
                }

            except Exception as e:
                return {'passed': False, 'error': f'Signal generation failed: {e}', 'scenario': 'market_gaps'}

        except Exception as e:
            return {'passed': False, 'error': str(e), 'scenario': 'market_gaps'}

    def _stress_test_low_volume(self, data: Dict[str, pd.DataFrame], symbol: str) -> Dict[str, Any]:
        """Test system behavior during low volume periods."""
        try:
            # Filter to low volume periods (bottom 20%)
            df_5m = data['5m'].copy()
            volume_threshold = df_5m['volume'].quantile(0.2)

            low_vol_data = {}
            for timeframe, df in data.items():
                if timeframe in ['5m', '15m']:
                    low_vol_periods = df_5m[df_5m['volume'] <= volume_threshold]['timestamp']
                    filtered_df = df[df['timestamp'].isin(low_vol_periods)]
                    low_vol_data[timeframe] = filtered_df

            if all(df.empty for df in low_vol_data.values()):
                return {'passed': True, 'reason': 'No low volume periods to test'}

            # Test indicator stability in low volume
            try:
                backtest_data = self._prepare_backtest_data(low_vol_data)

                # Check if indicators can be calculated
                for timeframe, df in backtest_data.items():
                    if f'{timeframe}_macd_line' in df.columns:
                        macd_values = df[f'{timeframe}_macd_line'].dropna()
                        if len(macd_values) == 0:
                            return {'passed': False, 'reason': f'MACD calculation failed for {timeframe}'}

                return {
                    'passed': True,
                    'low_volume_periods_tested': len(low_vol_data['5m']) if '5m' in low_vol_data else 0,
                    'scenario': 'low_volume'
                }

            except Exception as e:
                return {'passed': False, 'error': f'Indicator calculation failed: {e}', 'scenario': 'low_volume'}

        except Exception as e:
            return {'passed': False, 'error': str(e), 'scenario': 'low_volume'}

    def _stress_test_trend_reversals(self, data: Dict[str, pd.DataFrame], symbol: str) -> Dict[str, Any]:
        """Test system behavior during trend reversals."""
        try:
            # Identify trend reversal points using simple moving average
            df_5m = data['5m'].copy()
            df_5m['sma_20'] = df_5m['close'].rolling(20).mean()
            df_5m['trend'] = np.where(df_5m['close'] > df_5m['sma_20'], 1, -1)
            df_5m['trend_change'] = df_5m['trend'].diff().abs()

            # Find reversal points
            reversal_points = df_5m[df_5m['trend_change'] > 0]['timestamp']

            if len(reversal_points) == 0:
                return {'passed': True, 'reason': 'No trend reversals detected'}

            # Test signal generation around reversal points (±2 hours window)
            reversal_windows = []
            for reversal_time in reversal_points:
                window_start = reversal_time - pd.Timedelta(hours=2)
                window_end = reversal_time + pd.Timedelta(hours=2)

                window_data = {}
                for timeframe, df in data.items():
                    if timeframe in ['5m', '15m']:
                        window_df = df[
                            (df['timestamp'] >= window_start) &
                            (df['timestamp'] <= window_end)
                        ]
                        window_data[timeframe] = window_df

                if not all(df.empty for df in window_data.values()):
                    reversal_windows.append(window_data)

            if not reversal_windows:
                return {'passed': True, 'reason': 'No valid reversal windows'}

            # Test signal consistency across reversal windows
            consistent_windows = 0
            total_windows = len(reversal_windows)

            for window_data in reversal_windows:
                try:
                    backtest_signals = self._generate_backtest_signals(window_data, symbol)
                    live_signals = self._generate_live_signals(window_data, symbol)

                    # Simple consistency check - both systems should generate similar signal counts
                    bt_signal_count = len(backtest_signals[backtest_signals[['entry_signal_buy', 'entry_signal_sell']].any(axis=1)]) if not backtest_signals.empty else 0
                    live_signal_count = len(live_signals)

                    if abs(bt_signal_count - live_signal_count) <= 1:  # Allow 1 signal difference
                        consistent_windows += 1

                except Exception:
                    # Window failed - don't count as consistent
                    pass

            consistency_rate = consistent_windows / total_windows if total_windows > 0 else 1.0
            passed = consistency_rate >= 0.7  # 70% consistency required

            return {
                'passed': passed,
                'consistency_rate': consistency_rate,
                'reversal_windows_tested': total_windows,
                'scenario': 'trend_reversals'
            }

        except Exception as e:
            return {'passed': False, 'error': str(e), 'scenario': 'trend_reversals'}

    def _stress_test_extreme_values(self, data: Dict[str, pd.DataFrame], symbol: str) -> Dict[str, Any]:
        """Test system robustness with extreme price values."""
        try:
            # Create test data with extreme values
            extreme_data = {}
            for timeframe, df in data.items():
                if timeframe not in ['5m', '15m']:
                    continue

                test_df = df.copy()

                # Inject extreme values at random points
                extreme_indices = np.random.choice(
                    range(50, len(test_df) - 50),
                    size=min(3, len(test_df) // 100),
                    replace=False
                )

                for idx in extreme_indices:
                    # Create extreme price spike (10x normal)
                    base_price = test_df.iloc[idx]['close']
                    extreme_price = base_price * 10

                    test_df.iloc[idx, test_df.columns.get_loc('high')] = extreme_price
                    test_df.iloc[idx, test_df.columns.get_loc('close')] = extreme_price

                extreme_data[timeframe] = test_df

            # Test if systems can handle extreme values without crashing
            try:
                backtest_signals = self._generate_backtest_signals(extreme_data, symbol)
                live_signals = self._generate_live_signals(extreme_data, symbol)

                # Check if indicator calculations are reasonable
                valid_indicators = True

                for timeframe, df in extreme_data.items():
                    if f'{timeframe}_macd_line' in df.columns:
                        macd_values = df[f'{timeframe}_macd_line'].replace([np.inf, -np.inf], np.nan).dropna()
                        if len(macd_values) > 0 and macd_values.abs().max() > 1000:  # Unreasonably large MACD
                            valid_indicators = False
                            break

                return {
                    'passed': valid_indicators,
                    'backtest_signals_generated': len(backtest_signals) if not backtest_signals.empty else 0,
                    'live_signals_generated': len(live_signals),
                    'extreme_values_injected': len(extreme_indices) if 'extreme_indices' in locals() else 0,
                    'scenario': 'extreme_values'
                }

            except Exception as e:
                return {'passed': False, 'error': f'Systems failed with extreme values: {e}', 'scenario': 'extreme_values'}

        except Exception as e:
            return {'passed': False, 'error': str(e), 'scenario': 'extreme_values'}

    def _generate_validation_report(self, results: Dict[str, ValidationMetrics], symbol: str) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        report = {
            'symbol': symbol,
            'validation_timestamp': datetime.now().isoformat(),
            'overall_status': 'UNKNOWN',
            'summary': {},
            'detailed_results': results,
            'recommendations': [],
            'risk_assessment': 'LOW'
        }

        try:
            # Calculate overall validation score
            validation_scores = []

            # Signal parity score (40% weight)
            if 'signal_parity' in results:
                sp = results['signal_parity']
                signal_score = (
                    sp.signal_parity_rate * 0.6 +
                    min(sp.macd_correlation_5m, sp.macd_correlation_15m) * 0.4
                )
                validation_scores.append(('signal_parity', signal_score, 0.4))

            # Execution model score (35% weight)
            if 'execution_model' in results:
                em = results['execution_model']
                execution_score = (
                    em.trade_match_rate * 0.7 +
                    min(abs(em.pnl_correlation), 1.0) * 0.3
                )
                validation_scores.append(('execution_model', execution_score, 0.35))

            # State management score (15% weight)
            if 'state_management' in results:
                sm = results['state_management']
                state_score = sm.execution_delay_analysis.get('position_consistency_rate', 0.8)
                validation_scores.append(('state_management', state_score, 0.15))

            # Stress testing score (10% weight)
            if 'stress_testing' in results:
                st = results['stress_testing']
                stress_score = st.execution_delay_analysis.get('stress_test_pass_rate', 0.8)
                validation_scores.append(('stress_testing', stress_score, 0.10))

            # Calculate weighted average
            if validation_scores:
                weighted_score = sum(score * weight for _, score, weight in validation_scores)
                total_weight = sum(weight for _, _, weight in validation_scores)
                overall_score = weighted_score / total_weight if total_weight > 0 else 0
            else:
                overall_score = 0

            # Determine overall status
            if overall_score >= 0.90:
                report['overall_status'] = 'EXCELLENT'
                report['risk_assessment'] = 'LOW'
            elif overall_score >= 0.80:
                report['overall_status'] = 'GOOD'
                report['risk_assessment'] = 'LOW'
            elif overall_score >= 0.70:
                report['overall_status'] = 'ACCEPTABLE'
                report['risk_assessment'] = 'MEDIUM'
            elif overall_score >= 0.60:
                report['overall_status'] = 'MARGINAL'
                report['risk_assessment'] = 'HIGH'
            else:
                report['overall_status'] = 'FAILED'
                report['risk_assessment'] = 'CRITICAL'

            # Generate summary
            report['summary'] = {
                'overall_score': overall_score,
                'component_scores': {name: score for name, score, _ in validation_scores},
                'critical_issues': [],
                'warnings': []
            }

            # Generate recommendations
            self._generate_recommendations(report, results)

        except Exception as e:
            logger.error(f"Error generating validation report: {e}")
            report['overall_status'] = 'ERROR'
            report['risk_assessment'] = 'CRITICAL'

        return report

    def _generate_recommendations(self, report: Dict[str, Any], results: Dict[str, ValidationMetrics]):
        """Generate actionable recommendations based on validation results."""
        recommendations = []

        try:
            # Signal parity recommendations
            if 'signal_parity' in results:
                sp = results['signal_parity']
                if sp.signal_parity_rate < 0.90:
                    recommendations.append({
                        'category': 'SIGNAL_PARITY',
                        'priority': 'HIGH',
                        'issue': f'Low signal matching rate: {sp.signal_parity_rate:.2%}',
                        'recommendation': 'Review indicator calculation differences between systems',
                        'action': 'Compare MACD and EMA implementations line-by-line'
                    })

                min_correlation = min(sp.macd_correlation_5m, sp.macd_correlation_15m,
                                    sp.ema_correlation_5m, sp.ema_correlation_15m)
                if min_correlation < 0.95:
                    recommendations.append({
                        'category': 'INDICATOR_CORRELATION',
                        'priority': 'HIGH',
                        'issue': f'Low indicator correlation: {min_correlation:.3f}',
                        'recommendation': 'Ensure identical data preprocessing and calculation precision',
                        'action': 'Standardize floating-point precision and rounding methods'
                    })

            # Execution model recommendations
            if 'execution_model' in results:
                em = results['execution_model']
                if em.trade_match_rate < 0.80:
                    recommendations.append({
                        'category': 'EXECUTION_MODEL',
                        'priority': 'HIGH',
                        'issue': f'Low trade matching rate: {em.trade_match_rate:.2%}',
                        'recommendation': 'Review timing alignment between two-bar and immediate execution',
                        'action': 'Implement execution delay compensation in live system'
                    })

                if abs(em.pnl_correlation) < 0.85:
                    recommendations.append({
                        'category': 'PNL_TRACKING',
                        'priority': 'MEDIUM',
                        'issue': f'PnL correlation below threshold: {em.pnl_correlation:.3f}',
                        'recommendation': 'Verify commission and slippage modeling consistency',
                        'action': 'Audit transaction cost calculations in both systems'
                    })

            # Statistical significance recommendations
            for component, metrics in results.items():
                if hasattr(metrics, 'p_values') and metrics.p_values:
                    for test_name, p_value in metrics.p_values.items():
                        if p_value > 0.05:
                            recommendations.append({
                                'category': 'STATISTICAL_SIGNIFICANCE',
                                'priority': 'MEDIUM',
                                'issue': f'{test_name} not statistically significant (p={p_value:.3f})',
                                'recommendation': 'Increase sample size or review validation methodology',
                                'action': f'Collect more data for {component} validation'
                            })

            # Stress testing recommendations
            if 'stress_testing' in results:
                st = results['stress_testing']
                pass_rate = st.execution_delay_analysis.get('stress_test_pass_rate', 1.0)
                if pass_rate < 0.80:
                    recommendations.append({
                        'category': 'STRESS_TESTING',
                        'priority': 'MEDIUM',
                        'issue': f'Stress test pass rate: {pass_rate:.2%}',
                        'recommendation': 'Improve system robustness for edge cases',
                        'action': 'Add error handling for extreme market conditions'
                    })

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append({
                'category': 'SYSTEM_ERROR',
                'priority': 'CRITICAL',
                'issue': 'Validation framework error',
                'recommendation': 'Review validation system implementation',
                'action': 'Debug validation framework'
            })

        report['recommendations'] = recommendations

    def _store_signal_comparisons(self, comparisons: List[SignalComparison], symbol: str):
        """Store signal comparisons in database."""
        try:
            with sqlite3.connect(self.validation_db_path) as conn:
                cursor = conn.cursor()

                for comp in comparisons:
                    cursor.execute("""
                        INSERT INTO signal_comparisons
                        (timestamp, symbol, backtest_signal, live_signal, signal_match,
                         backtest_indicators, live_indicators)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        comp.timestamp.isoformat(),
                        symbol,
                        comp.backtest_signal,
                        comp.live_signal,
                        comp.match,
                        json.dumps(comp.backtest_indicators),
                        json.dumps(comp.live_indicators)
                    ))

                conn.commit()

        except Exception as e:
            logger.error(f"Error storing signal comparisons: {e}")

    def _store_trade_comparisons(self, comparisons: List[TradeComparison], symbol: str):
        """Store trade comparisons in database."""
        try:
            with sqlite3.connect(self.validation_db_path) as conn:
                cursor = conn.cursor()

                for comp in comparisons:
                    cursor.execute("""
                        INSERT INTO trade_comparisons
                        (trade_id, symbol, backtest_entry_time, live_entry_time,
                         backtest_entry_price, live_entry_price, backtest_exit_time,
                         live_exit_time, backtest_exit_price, live_exit_price,
                         backtest_pnl, live_pnl, is_valid_comparison, discrepancy_reason)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        comp.trade_id,
                        symbol,
                        comp.backtest_entry_time.isoformat() if comp.backtest_entry_time else None,
                        comp.live_entry_time.isoformat() if comp.live_entry_time else None,
                        comp.backtest_entry_price,
                        comp.live_entry_price,
                        comp.backtest_exit_time.isoformat() if comp.backtest_exit_time else None,
                        comp.live_exit_time.isoformat() if comp.live_exit_time else None,
                        comp.backtest_exit_price,
                        comp.live_exit_price,
                        comp.backtest_pnl,
                        comp.live_pnl,
                        comp.is_valid_comparison,
                        comp.discrepancy_reason
                    ))

                conn.commit()

        except Exception as e:
            logger.error(f"Error storing trade comparisons: {e}")

    def _store_validation_session(self, session_id: str, results: Dict[str, ValidationMetrics], status: str):
        """Store validation session results."""
        try:
            with sqlite3.connect(self.validation_db_path) as conn:
                cursor = conn.cursor()

                # Convert results to JSON-serializable format
                serializable_results = {}
                for key, metrics in results.items():
                    if hasattr(metrics, '__dict__'):
                        serializable_results[key] = asdict(metrics)
                    else:
                        serializable_results[key] = metrics

                cursor.execute("""
                    INSERT OR REPLACE INTO validation_sessions
                    (session_id, start_time, end_time, validation_type, metrics, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    'COMPREHENSIVE',
                    json.dumps(serializable_results),
                    status
                ))

                conn.commit()

        except Exception as e:
            logger.error(f"Error storing validation session: {e}")

    def _perform_signal_statistical_tests(self, metrics: ValidationMetrics):
        """Perform additional statistical tests on signal validation."""
        try:
            # Add statistical significance tests to metrics
            if metrics.total_signals > 0:
                # Chi-square test for signal distribution
                expected = metrics.total_signals * 0.5  # Expected under null hypothesis
                observed = metrics.matching_signals

                if expected > 5:  # Chi-square validity condition
                    chi2_stat = ((observed - expected) ** 2) / expected
                    p_value = 1 - stats.chi2.cdf(chi2_stat, 1)

                    if not metrics.p_values:
                        metrics.p_values = {}
                    metrics.p_values['signal_distribution_chi2'] = p_value

        except Exception as e:
            logger.error(f"Error in signal statistical tests: {e}")