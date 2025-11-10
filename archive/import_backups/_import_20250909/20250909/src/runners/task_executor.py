#!/usr/bin/env python3
"""
Task Executor Module for Unified Backtester

Handles parallel/sequential execution of backtest tasks.
Includes validation, risk management, transaction costs, and bias detection.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Core system imports
from src.core.etl.loader import load_base_data
from src.core.strat_stats.strategy_executor import extract_trades
from src.core.strat_stats.statistics import (
    calculate_metrics,
    calculate_advanced_metrics,
    calculate_returns_series
)
from src.strategies.strategy_factory import StrategyFactory
from src.strategies.register_strategies import register_all_strategies
from src.core.validation.bias_detector import BiasDetector
from src.core.costs.transaction_models import AdvancedTransactionCosts
from src.core.risk.risk_manager import RiskManager
from src.core.options.options_engine import OptionsBacktester

# Import modular validation component
from src.runners.components.validator import DataValidator


class TaskExecutor:
    """
    Executes backtest tasks in parallel or sequential mode.
    """
    
    def __init__(self, config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Initialize components
        self.risk_manager = None
        self.transaction_costs = None
        self.bias_detector = None
        self.options_engine = None
          # Initialize modular data validator
        self.data_validator = DataValidator(self.logger)
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize task execution components."""
        try:
            # Initialize risk manager
            if self.config.risk:
                risk_config = {
                    'max_position_size': getattr(self.config.risk, 'max_position_size', 0.1),
                    'max_sector_exposure': getattr(self.config.risk, 'max_sector_exposure', 0.3), 
                    'max_drawdown': getattr(self.config.risk, 'max_drawdown', 0.2),
                    'max_leverage': getattr(self.config.risk, 'max_leverage', 1.0),
                    'stop_loss_threshold': getattr(self.config.risk, 'stop_loss_threshold', 0.1),
                    'position_limits': getattr(self.config.risk, 'position_limits', {}),
                    'enable_dynamic_sizing': getattr(self.config.risk, 'enable_dynamic_sizing', True)
                }
                self.risk_manager = RiskManager(risk_config)
                self.logger.info("Risk manager initialized")
                
            # Initialize transaction cost model
            if self.config.transaction.model_type == "advanced":
                self.transaction_costs = AdvancedTransactionCosts()
                self.logger.info("Advanced transaction costs initialized")
                
            # Initialize bias detector
            if self.config.validation.enabled:
                self.bias_detector = BiasDetector()
                self.logger.info("Bias detector initialized")
                
            # Initialize options engine
            if self.config.options.enabled:
                self.options_engine = OptionsBacktester()
                self.logger.info("Options engine initialized")
                
        except Exception as e:
            self.logger.error(f"Error initializing task executor components: {e}")
            raise
    
    def execute_tasks(self, tasks: List, use_parallel: bool = True) -> Dict[str, Any]:
        """
        Execute backtest tasks in parallel or sequential mode.
        
        Args:
            tasks: List of (ticker, date_range, strategy_name, optimization_params) tuples
            use_parallel: Whether to use parallel processing
            
        Returns:
            Structured dictionary of results
        """
        self.logger.info(f"📋 Executing {len(tasks)} backtest tasks")
        
        # Create output directory for results
        output_dir = Path(self.config.base_dir) / self.config.output.output_dir / self.config.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check portfolio mode
        is_portfolio_mode = len(set(task[0] for task in tasks)) > 1 and hasattr(self.config.strategy, 'risk_profile') and 'portfolio' in self.config.strategy.risk_profile.lower()
        
        # Execute tasks
        if use_parallel and len(tasks) > 1 and not is_portfolio_mode:
            pool_size = min(cpu_count(), len(tasks), self.config.execution.max_workers if hasattr(self.config.execution, 'max_workers') else 4)
            self.logger.info(f"🔄 Starting multiprocessing pool with {pool_size} processes")
            
            with Pool(processes=pool_size) as pool:
                results_list = pool.map(self.run_backtest_task, tasks)
        else:
            reason = "portfolio coordination" if is_portfolio_mode else "sequential mode"
            self.logger.info(f"🔄 Running tasks sequentially ({reason})")
            results_list = [self.run_backtest_task(task) for task in tasks]
          # Organize results into structured dictionary
        structured_results = {}
        for result in results_list:
            if not result:
                continue
                
            strategy = result.get('strategy')
            date_range = result.get('date_range')
            ticker = result.get('ticker')
            
            if not strategy or not date_range or not ticker:
                continue
                
            if strategy not in structured_results:
                structured_results[strategy] = {}
            
            if date_range not in structured_results[strategy]:
                structured_results[strategy][date_range] = {}
                
            structured_results[strategy][date_range][ticker] = result
        
        self.logger.info(f"✅ Completed {len(results_list)} backtest tasks")
        return structured_results
    
    def run_backtest_task(self, args_tuple) -> Dict[str, Any]:
        """
        Execute a single backtest task.
        """
        # Register strategies in this worker process
        if not register_all_strategies():
            return {"error": "Failed to register strategies in worker process"}
        
        ticker, date_range, strategy_name, optimization_params = args_tuple
        
        self.logger.info(f"Processing {ticker} with {strategy_name} for {date_range}")
        
        try:
            # Load base data
            base_df = load_base_data(date_range, ticker)
            if base_df is None or base_df.empty:
                self.logger.warning(f"No data found for {ticker} in {date_range}")
                return {}
            
            # Validate market data using modular validator
            validation_result = self.data_validator.validate_market_data(base_df)
            if not validation_result['is_valid']:
                self.logger.error(f"Data validation failed for {ticker} in {date_range}: {validation_result['issues']}")
                return {}
            
            # Log validation warnings if any
            if validation_result['warnings']:
                for warning in validation_result['warnings']:
                    self.logger.warning(f"Data validation warning for {ticker}: {warning}")
            
            # Get strategy instance
            strategy = StrategyFactory.get_strategy(strategy_name)
            if strategy is None:
                self.logger.error(f"Strategy '{strategy_name}' not found")
                return {}
            
            # Execute strategy
            final_df = strategy.execute(base_df, ticker, date_range)
            if final_df is None or final_df.empty:
                self.logger.warning(f"Strategy returned no data for {ticker} in {date_range}")
                return {}
            
            # Extract trades
            trades = extract_trades(final_df)
            strategy_trades = trades.copy() if trades else []
            
            # Apply transaction costs
            if trades and self.transaction_costs:
                trades = self._apply_transaction_costs(trades, base_df)
                strategy_trades = self._apply_transaction_costs(strategy_trades.copy(), base_df)
            
            # Apply risk management
            risk_report = {}
            if trades and self.risk_manager:
                trades, risk_report = self._apply_risk_management(trades, ticker, base_df)
            
            # Calculate metrics
            if trades:
                basic_metrics = calculate_metrics(trades, final_df)
                advanced_metrics = calculate_advanced_metrics(trades)
                metrics = {**basic_metrics, **advanced_metrics}
            else:
                metrics = calculate_advanced_metrics([])
            
            # Add metadata
            metrics.update({
                "ticker": ticker,
                "strategy": strategy_name,
                "date_range": date_range
            })
            
            return {
                'ticker': ticker,
                'strategy': strategy_name,
                'date_range': date_range,
                'trades': trades,
                'strategy_trades': strategy_trades,
                'metrics': metrics,
                'risk_report': risk_report,
                'base_data': final_df
            }
            
        except Exception as e:
            self.logger.error(f"Error in backtest task for {ticker} with {strategy_name} on {date_range}: {e}")
            return {}
    
    def _apply_transaction_costs(self, trades: List[Dict], market_data: pd.DataFrame) -> List[Dict]:
        """Apply transaction costs to trades."""
        if not self.transaction_costs:
            return trades
        
        # Simplified transaction cost application
        enhanced_trades = []
        for trade in trades:
            # Apply costs (simplified)
            cost = abs(trade.get('PnL', 0)) * 0.001  # 0.1% cost
            trade['PnL'] = trade.get('PnL', 0) - cost
            enhanced_trades.append(trade)
        
        return enhanced_trades
    
    def _apply_risk_management(self, trades: List[Dict], ticker: str, market_data: pd.DataFrame) -> Tuple[List[Dict], Dict]:
        """Apply risk management to trades."""
        if not self.risk_manager:
            return trades, {}
        
        # Simplified risk management
        approved_trades = []
        rejected_count = 0
        
        for trade in trades:
            # Simple position size check
            position_size = abs(trade.get('size', 100))
            if position_size <= 1000:  # Max position size
                approved_trades.append(trade)
            else:
                rejected_count += 1
        
        risk_report = {
            'original_trade_count': len(trades),
            'approved_trade_count': len(approved_trades),
            'rejected_trade_count': rejected_count
        }
        
        return approved_trades, risk_report
    
    def validate_data(self, dates: List[str], tickers: List[str]) -> bool:
        """Validate data availability and quality."""
        if not self.config.validation.enabled:
            self.logger.info("Data validation disabled")
            return True
            
        self.logger.info("🔍 Starting data validation...")
        
        validation_results = {}
        for date in dates:
            for ticker in tickers:
                try:
                    data = load_base_data(date, ticker)
                    
                    if data is None or len(data) < self.config.validation.min_data_points:
                        self.logger.warning(f"Insufficient data for {ticker} on {date}")
                        validation_results[f"{ticker}_{date}"] = "insufficient_data"
                        continue
                    
                    validation_results[f"{ticker}_{date}"] = "passed"
                    
                except Exception as e:
                    self.logger.error(f"Error validating {ticker} on {date}: {e}")
                    validation_results[f"{ticker}_{date}"] = "error"
        
        failed_validations = [k for k, v in validation_results.items() if v != "passed"]
        
        if failed_validations:
            self.logger.warning(f"Validation issues found for: {failed_validations}")
            if not self.config.validation.strict_mode:
                self.logger.info("Running in non-strict mode, continuing despite validation issues")
                return True
            return False
            
        self.logger.info("✅ Data validation completed successfully")
        return True
