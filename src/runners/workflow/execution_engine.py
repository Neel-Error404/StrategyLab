#!/usr/bin/env python3
"""
Execution Engine - Core backtest task processing

Handles individual backtest tasks including:
- Strategy execution with optimization
- Bias detection and validation  
- Risk management application
- Transaction cost modeling
- Comprehensive metrics calculation
- Result persistence with enhanced structure
"""

import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from config.unified_config import BacktestConfig
from src.core.etl.loader import load_base_data
from src.core.strat_stats.strategy_executor import extract_trades
from src.core.strat_stats.statistics import (
    calculate_metrics,
    calculate_advanced_metrics,
    calculate_returns_series
)
from src.strategies.strategy_factory import StrategyFactory


class ExecutionEngine:
    """Handles individual backtest task execution with comprehensive feature set."""
    
    def __init__(self, config: BacktestConfig, 
                 risk_manager=None, transaction_costs=None, 
                 bias_detector=None, options_engine=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Component dependencies injected for testability
        self.risk_manager = risk_manager
        self.transaction_costs = transaction_costs
        self.bias_detector = bias_detector
        self.options_engine = options_engine
        
    def run_backtest_task(self, args_tuple) -> Dict[str, Any]:
        """
        Enhanced backtest task that combines features from both runners.
        Supports bias detection, risk management, transaction costs, and optimization.
        """
        ticker, date_range, strategy_name, optimization_params = args_tuple
        
        self.logger.info(f"Processing {ticker} with {strategy_name} for {date_range}")
        try:
            # Load and validate base data
            base_df = load_base_data(date_range, ticker)
            if base_df is None or base_df.empty:
                self.logger.warning(f"No data found for {ticker} in {date_range}")
                return {}
            
            # Parse date range for metadata
            date_range_meta = self._parse_date_range_metadata(date_range)
            
            # Get and prepare strategy instance
            strategy = StrategyFactory.get_strategy(strategy_name)
            if strategy is None:
                self.logger.error(f"Strategy '{strategy_name}' not found")
                return {}
            
            # Strategy optimization if requested
            if optimization_params:
                self._apply_strategy_optimization(strategy, base_df, ticker, date_range, optimization_params)
            
            # Execute strategy with bias detection
            bias_report, final_df = self._execute_strategy_with_bias_check(
                strategy, base_df, ticker, date_range
            )
            
            if final_df is None or final_df.empty:
                self.logger.warning(f"Strategy returned no data for {ticker} in {date_range}")
                return {}
                
            # Extract and process trades
            trades = extract_trades(final_df)
            strategy_trades = trades.copy() if trades else []  # Store all strategy-generated trades
            
            # Apply transaction costs and risk management
            processed_trades, risk_report = self._process_trades(
                trades, strategy_trades, ticker, base_df
            )
            
            # Calculate comprehensive metrics
            metrics = self._calculate_comprehensive_metrics(
                processed_trades, final_df, ticker, strategy_name, date_range_meta, risk_report
            )
            
            # Save results with enhanced structure
            self._save_enhanced_results(
                strategy_name, date_range, ticker, final_df, 
                strategy_trades, processed_trades, metrics, bias_report, risk_report
            )
            
            self.logger.info(f"Backtest completed for {ticker} with {strategy_name} on {date_range}. "
                           f"Trades: {len(processed_trades)}, Metrics calculated: {len(metrics)}")
                           
            return {
                'ticker': ticker,
                'strategy': strategy_name,
                'date_range': date_range,
                'trades': processed_trades,  # Risk-approved trades
                'strategy_trades': strategy_trades,  # All strategy-generated trades
                'metrics': metrics,
                'bias_report': bias_report,
                'risk_report': risk_report,
                'base_data': final_df  # Include base data for visualization
            }
            
        except Exception as e:
            self.logger.error(f"Error in backtest task for {ticker} with {strategy_name} on {date_range}: {e}", 
                            exc_info=True)
            return {}
    
    def _parse_date_range_metadata(self, date_range: str) -> Dict[str, Any]:
        """Parse date range string and extract metadata."""
        try:
            if "_to_" in date_range:
                start_str, end_str = date_range.split("_to_")
                start_date = datetime.strptime(start_str, "%Y-%m-%d")
                end_date = datetime.strptime(end_str, "%Y-%m-%d")
                return {
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "days": (end_date - start_date).days + 1
                }
            else:
                return {"date_range": date_range}
        except ValueError:
            self.logger.warning(f"Invalid date range format: {date_range}")
            return {"date_range": date_range}
    
    def _apply_strategy_optimization(self, strategy, base_df: pd.DataFrame, 
                                   ticker: str, date_range: str, optimization_params: Dict):
        """Apply parameter optimization to strategy."""
        try:
            optimal_params = strategy.optimize_parameters(base_df, ticker, date_range, optimization_params)
            self.logger.info(f"Optimized parameters for {strategy.name}: {optimal_params}")
            strategy.update_parameters(optimal_params)
        except Exception as e:
            self.logger.warning(f"Parameter optimization failed: {e}")
    
    def _execute_strategy_with_bias_check(self, strategy, base_df: pd.DataFrame, 
                                        ticker: str, date_range: str) -> Tuple[Dict, pd.DataFrame]:
        """Execute strategy with optional bias detection."""
        bias_report = {}
        
        if self.config.validation.enabled and self.bias_detector:
            is_valid, final_df, violations = self._validate_strategy_bias(
                strategy, base_df, ticker, date_range
            )
            bias_report = {
                'look_ahead_valid': is_valid,
                'violations': violations[:10]  # Keep first 10 violations
            }
            if not is_valid:
                self.logger.warning(f"Strategy failed bias validation for {ticker}. Continuing with warnings.")
        else:
            # Execute strategy without bias check
            final_df = strategy.execute(base_df, ticker, date_range)
            
        return bias_report, final_df
    
    def _validate_strategy_bias(self, strategy, base_df: pd.DataFrame, ticker: str, 
                               date_range: str) -> Tuple[bool, pd.DataFrame, List[Dict]]:
        """Validate strategy for potential biases."""
        if not self.bias_detector:
            return True, strategy.execute(base_df, ticker, date_range), []
        
        # Execute strategy to get signals
        final_df = strategy.execute(base_df, ticker, date_range)
        
        if final_df is None or final_df.empty:
            return False, pd.DataFrame(), [{'error': 'Strategy returned empty DataFrame'}]
        
        # Define signal and indicator columns
        signal_columns = [col for col in final_df.columns if 'signal' in col.lower()]
        indicator_columns = [col for col in final_df.columns 
                           if any(ind in col.lower() for ind in ['macd', 'ema', 'rsi', 'bb', 'sma'])]
        
        # Check for look-ahead bias
        is_valid, violations = self.bias_detector.validate_no_lookahead(
            final_df, signal_columns, indicator_columns
        )
        
        if not is_valid:
            self.logger.warning(f"Look-ahead bias detected in {strategy.name} for {ticker}")
            for violation in violations[:5]:
                self.logger.warning(f"  {violation}")
        
        return is_valid, final_df, violations
    
    def _process_trades(self, trades: List[Dict], strategy_trades: List[Dict], 
                       ticker: str, base_df: pd.DataFrame) -> Tuple[List[Dict], Dict]:
        """Apply transaction costs and risk management to trades."""
        risk_report = {}
        
        if not trades:
            self.logger.warning(f"No trades generated for {ticker}")
            # Get risk attribution for zero trades
            if self.risk_manager and self.config.risk.enabled:
                attribution = self.risk_manager.get_zero_trades_attribution(0)
                risk_report = {'attribution': attribution}
            return [], risk_report
        
        # Apply transaction costs
        processed_trades = trades
        if self.transaction_costs and self.config.transaction.enabled:
            processed_trades = self._apply_transaction_costs(processed_trades, base_df)
            # Also apply to strategy_trades for comparison
            strategy_trades = self._apply_transaction_costs(strategy_trades.copy(), base_df)
        
        # Apply risk management with enhanced reporting
        if self.risk_manager and self.config.risk:
            processed_trades, risk_report = self._apply_risk_management(processed_trades, ticker, base_df)
        else:
            risk_report = {'risk_management_disabled': True}
            
        return processed_trades, risk_report
    
    def _apply_transaction_costs(self, trades: List[Dict], market_data: pd.DataFrame) -> List[Dict]:
        """Apply advanced transaction cost models."""
        if not self.transaction_costs:
            return trades
        
        try:
            enhanced_trades = []
            for trade in trades:
                # Prepare trade data for cost calculation
                entry_trade = {
                    'size': trade.get('size', 100),
                    'price': trade['Entry Price'],
                    'timestamp': pd.to_datetime(trade['Entry Time']),
                    'ticker': trade.get('ticker', 'UNKNOWN'),
                    'side': 'BUY'
                }
                
                exit_trade = {
                    'size': trade.get('size', 100),
                    'price': trade['Exit Price'],
                    'timestamp': pd.to_datetime(trade['Exit Time']),
                    'ticker': trade.get('ticker', 'UNKNOWN'),
                    'side': 'SELL'
                }
                
                # Create market state from market data
                market_state = self._create_market_state(trade, market_data)
                
                # Calculate transaction costs
                entry_costs = self.transaction_costs.calculate_costs(entry_trade, market_state)
                exit_costs = self.transaction_costs.calculate_costs(exit_trade, market_state)
                
                # Apply costs to trade
                trade_copy = trade.copy()
                trade_copy['Entry Transaction Cost'] = entry_costs['total_cost']
                trade_copy['Exit Transaction Cost'] = exit_costs['total_cost']
                trade_copy['Total Transaction Cost'] = entry_costs['total_cost'] + exit_costs['total_cost']
                
                # Adjust PnL for costs
                original_pnl = trade.get('PnL', 0)
                adjusted_pnl = original_pnl - trade_copy['Total Transaction Cost']
                trade_copy['PnL'] = adjusted_pnl
                trade_copy['Cost Impact %'] = (trade_copy['Total Transaction Cost'] / abs(original_pnl) * 100) if original_pnl != 0 else 0
                
                enhanced_trades.append(trade_copy)
                
            return enhanced_trades
            
        except Exception as e:
            self.logger.error(f"Error applying transaction costs: {e}")
            return trades
    
    def _create_market_state(self, trade: Dict, market_data: pd.DataFrame) -> Dict:
        """Create market state for transaction cost calculation."""
        if isinstance(market_data, pd.DataFrame) and len(market_data) > 0:
            # Safely access volume column
            if 'volume' in market_data.columns:
                volume = market_data['volume'].iloc[0] if len(market_data) > 0 else 1000000
            else:
                volume = 1000000
            
            return {
                'close': trade['Entry Price'],
                'volume': volume,
                'bid': trade['Entry Price'] * 0.999,
                'ask': trade['Entry Price'] * 1.001,
                'spread': trade['Entry Price'] * 0.002
            }
        else:
            # Fallback market state for testing
            return {
                'close': trade['Entry Price'],
                'volume': 1000000,
                'bid': trade['Entry Price'] * 0.999,
                'ask': trade['Entry Price'] * 1.001,
                'spread': trade['Entry Price'] * 0.002
            }
    
    def _apply_risk_management(self, trades: List[Dict], ticker: str, 
                             market_data: pd.DataFrame) -> Tuple[List[Dict], Dict]:
        """Apply risk management with detailed reporting."""
        if not self.risk_manager:
            return trades, {'risk_management_disabled': True}
        
        try:
            original_count = len(trades)
            
            # Apply risk management
            approved_trades = []
            rejected_trades = []
            
            for trade in trades:
                if self.risk_manager.validate_trade(trade, ticker, market_data):
                    approved_trades.append(trade)
                else:
                    rejected_trades.append(trade)
            
            # Create comprehensive risk report
            risk_report = {
                'original_trade_count': original_count,
                'approved_trade_count': len(approved_trades),
                'rejected_trade_count': len(rejected_trades),
                'rejection_rate': (len(rejected_trades) / original_count * 100) if original_count > 0 else 0,
                'risk_management_enabled': True,
                'risk_management_bypassed': False
            }
            
            if rejected_trades:
                self.logger.info(f"Risk management rejected {len(rejected_trades)}/{original_count} trades for {ticker}")
                risk_report['sample_rejections'] = [
                    {
                        'entry_time': trade.get('Entry Time', 'Unknown'),
                        'entry_price': trade.get('Entry Price', 0),
                        'reason': 'Risk validation failed'
                    }
                    for trade in rejected_trades[:3]  # Sample first 3 rejections
                ]
            
            return approved_trades, risk_report
            
        except Exception as e:
            self.logger.error(f"Error in risk management: {e}")
            return trades, {'risk_management_error': str(e)}
    
    def _calculate_comprehensive_metrics(self, trades: List[Dict], final_df: pd.DataFrame,
                                       ticker: str, strategy_name: str, 
                                       date_range_meta: Dict, risk_report: Dict) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        if not trades:
            metrics = calculate_advanced_metrics([])
            # Add zero trades attribution from risk report
            if 'attribution' in risk_report:
                attribution = risk_report['attribution']
                metrics.update({
                    "zero_trades_reason": attribution.get('primary_reason', 'unknown'),
                    "zero_trades_category": attribution.get('category', 'unknown'),
                    "zero_trades_explanation": attribution.get('explanation', 'No explanation available')
                })
        else:
            # Calculate basic and advanced metrics
            basic_metrics = calculate_metrics(trades, final_df)
            advanced_metrics = calculate_advanced_metrics(trades)
            metrics = {**basic_metrics, **advanced_metrics}
            
            # Add risk metrics if available
            if self.risk_manager and trades:
                returns = calculate_returns_series(pd.DataFrame(trades), self.config.strategy.initial_capital)
                portfolio = {'total_value': self.config.strategy.initial_capital, 'positions': {}}
                risk_metrics = self.risk_manager.calculate_portfolio_risk_metrics(portfolio, returns)
                metrics.update(risk_metrics)
            
            # Options calculations if enabled
            if self.options_engine and self.config.options.enabled:
                options_metrics = self.options_engine.calculate_options_metrics(
                    trades, final_df, self.config.options
                )
                metrics.update(options_metrics)
        
        # Update metrics with metadata
        strategy = StrategyFactory.get_strategy(strategy_name)
        metrics.update(date_range_meta)
        metrics["Parameters"] = str(strategy.parameters) if strategy else "Unknown"
        metrics["ticker"] = ticker
        metrics["strategy"] = strategy_name
        
        # Add risk management metrics
        if risk_report:
            metrics.update({
                "risk_original_trades": risk_report.get('original_trade_count', len(trades)),
                "risk_approved_trades": risk_report.get('approved_trade_count', len(trades)),
                "risk_rejected_trades": risk_report.get('rejected_trade_count', 0),
                "risk_management_enabled": not risk_report.get('risk_management_disabled', False),
                "risk_management_bypassed": risk_report.get('risk_management_bypassed', False)
            })
        
        return metrics
    
    def _save_enhanced_results(self, strategy_name: str, date_range: str, ticker: str,
                             final_df: pd.DataFrame, strategy_trades: List[Dict],
                             processed_trades: List[Dict], metrics: Dict,
                             bias_report: Dict, risk_report: Dict):
        """Save results with enhanced structure using optimized output system."""
        try:
            # This will be implemented by the caller (orchestrator) that has access to output systems
            # For now, just log that results would be saved
            self.logger.info(f"Results ready for saving: {strategy_name} - {ticker} - {date_range}")
            
        except Exception as e:
            self.logger.error(f"Error saving enhanced results: {e}")
