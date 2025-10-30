#!/usr/bin/env python3
"""
Task Executor Module for Unified Backtester

Thin orchestration layer that now delegates execution to the rich ExecutionEngine
while maintaining parallelism and legacy interfaces.
"""

import logging
from multiprocessing import Pool, cpu_count
import signal
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.core.etl.loader import load_base_data
from src.strategies.register_strategies import register_all_strategies
from src.core.validation.bias_detector import BiasDetector
from src.core.costs.transaction_models import AdvancedTransactionCosts
from src.core.risk.risk_manager import RiskManager
from src.core.options.options_engine import OptionsBacktester
from src.runners.components.validator import DataValidator
from src.runners.workflow.execution_engine import ExecutionEngine


def init_worker_process():
    """Initialize worker process with proper signal handling and strategy registration."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ignore SIGINT in workers
    signal.signal(signal.SIGTERM, signal.SIG_DFL)  # Default SIGTERM handling
    
    # Register strategies once per worker process (not per task)
    register_all_strategies()


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
        self.execution_engine = ExecutionEngine(
            config,
            risk_manager=self.risk_manager,
            transaction_costs=self.transaction_costs,
            bias_detector=self.bias_detector,
            options_engine=self.options_engine,
            data_validator=self.data_validator
        )
    
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
            
            try:
                with Pool(processes=pool_size, initializer=init_worker_process) as pool:
                    results_list = pool.map(self.run_backtest_task, tasks)
            except KeyboardInterrupt:
                self.logger.warning("🛑 KeyboardInterrupt received - terminating all worker processes")
                pool.terminate()
                pool.join()
                raise KeyboardInterrupt("Backtesting interrupted by user")
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
        Execute a single backtest task via the consolidated execution engine.
        """
        return self.execution_engine.run_backtest_task(args_tuple)
    
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
