#!/usr/bin/env python3
"""
Unified Backtester Runner - Clean Modular Version

A production-ready backtesting system with fully modular architecture.
All legacy monolithic code has been removed and replaced with modular components.

ARCHITECTURE:
- CLI Handler: Command-line parsing and configuration management
- Workflow Manager: Mode-specific workflow orchestration  
- Task Executor: Parallel/sequential task execution and validation
- Analysis Engine: Portfolio-level analysis and metrics
- Visualization Engine: Comprehensive visualization generation

FEATURES:
- Comprehensive command-line argument parsing
- YAML-based configuration management
- Modern class-based architecture
- Signal-based graceful shutdown
- Streamlined error handling and logging
- Built-in data validation and quality checks
- Progress tracking with completion percentage

Usage:
    # Modern YAML-based approach
    python unified_runner.py --mode backtest --template conservative --dates 2024-01-01 2024-01-02
    
    # Traditional CLI approach (backward compatible)
    python unified_runner.py --mode backtest --strategy mse --date-ranges 2024-01-01_to_2024-01-02
"""

import signal
import sys
import traceback
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Fix import paths
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import new modular components
try:
    from src.runners.cli_handler import CLIHandler
    from src.runners.workflow_manager import WorkflowManager
    from src.runners.task_executor import TaskExecutor
    from src.runners.analysis_engine import AnalysisEngine
    from src.runners.visualization_engine import VisualizationEngine
    from src.runners.utils.naming import create_deterministic_name
    from src.runners.utils.helpers import configure_logging
    from src.strategies.register_strategies import register_all_strategies
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modular components are available")
    sys.exit(1)

# Core configuration imports
from config.unified_config import BacktestConfig


class UnifiedBacktesterRunner:
    """
    Unified backtester runner with modular architecture.
    
    This class orchestrates the entire backtesting workflow by delegating
    responsibilities to specialized modules:
    - CLI Handler: Command-line parsing and configuration    - Workflow Manager: Mode-specific workflow orchestration
    - Task Executor: Parallel/sequential task execution
    - Analysis Engine: Portfolio-level analysis and metrics
    - Visualization Engine: Comprehensive visualization generation
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.start_time = datetime.now()
        
        # Initialize CLI handler and setup logging
        self.cli_handler = CLIHandler()
        self.logger = configure_logging()
        
        # Register all available strategies
        if not register_all_strategies():
            self.logger.error("Failed to register strategies")
            raise RuntimeError("Strategy registration failed")
          # Set clean directory structure following monolith pattern
        # Format: {timestamp}/{strategy}/{date_range}
        from src.runners.utils.naming import create_monolith_directory_structure
        
        strategies = self.config.strategy.names if hasattr(self.config.strategy, 'names') else ['mse']
        date_ranges = self.config.strategy.date_ranges if hasattr(self.config.strategy, 'date_ranges') else ['2025-06-06_to_2025-06-07']
        
        # Use first strategy and first date range for directory structure
        strategy_name = strategies[0] if strategies else 'mse'
        date_range = date_ranges[0] if date_ranges else '2025-06-06_to_2025-06-07'
        
        # Create clean directory structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_run_dir = create_monolith_directory_structure(
            str(Path(self.config.base_dir) / self.config.output.output_dir),
            strategy_name,
            date_range,
            timestamp
        )
        
        # Set the clean directory as our run_id for consistency
        self.config.run_id = f"{timestamp}/{strategy_name}/{date_range}"
        self.config.strategy_run_dir = strategy_run_dir  # Store full path for components to use
        
        self.logger.info(f"Using clean output directory structure: {strategy_run_dir}")
        
        # Initialize modular components
        self.workflow_manager = WorkflowManager(config, self.logger)
        self.task_executor = TaskExecutor(config, self.logger)
        self.analysis_engine = AnalysisEngine(config, self.logger)
        self.visualization_engine = VisualizationEngine(config, self.logger)
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info(f"UnifiedBacktesterRunner initialized with {config.strategy.risk_profile} profile")
        self.logger.info("All modular components loaded successfully")    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}. Shutting down gracefully...")
        sys.exit(0)
    
    def _auto_discover_tickers(self, date_ranges: List[str]) -> List[str]:
        """
        Auto-discover available tickers from data pools for given date ranges.
        
        Args:
            date_ranges: List of date ranges to check
            
        Returns:
            List of discovered ticker symbols
        """
        discovered_tickers = set()
        
        for date_range in date_ranges:
            data_pool_path = Path(f"data/pools/{date_range}/1minute")
            if data_pool_path.exists():
                # Look for CSV files with ticker names
                for csv_file in data_pool_path.glob("*.csv"):
                    # Extract ticker from filename (format: TICKER_DATERANGE.csv)
                    filename = csv_file.stem  # Remove .csv extension
                    if '_' in filename:
                        ticker = filename.split('_')[0]  # Take everything before first underscore
                        discovered_tickers.add(ticker)
        
        return list(discovered_tickers)

    def _generate_tasks_from_config(self) -> List:
        """
        Generate tasks from configuration.
        
        Returns:
            List of (ticker, date_range, strategy_name, optimization_params) tuples
        """
        tasks = []
        strategies = self.config.strategy.names if hasattr(self.config.strategy, 'names') else ['mse']
        date_ranges = self.config.strategy.date_ranges if hasattr(self.config.strategy, 'date_ranges') else []
        tickers = self.config.strategy.tickers if hasattr(self.config.strategy, 'tickers') else []        # Auto-discovery for all modes when no tickers provided
        mode = getattr(self.config, 'mode', 'backtest')
        if not tickers and date_ranges:
            tickers = self._auto_discover_tickers(date_ranges)
            if tickers:
                self.logger.info(f"Auto-discovered {len(tickers)} tickers for {mode} mode: {tickers}")
                # Update the config with discovered tickers
                self.config.strategy.tickers = tickers
            else:
                self.logger.warning(f"No tickers found for {mode} mode in date ranges: {date_ranges}")
                return tasks
        
        # Generate tasks for all combinations
        for strategy in strategies:
            for date_range in date_ranges:
                for ticker in tickers:
                    optimization_params = {}  # TODO: Extract from config if needed
                    tasks.append((ticker, date_range, strategy, optimization_params))
        
        self.logger.info(f"Generated {len(tasks)} tasks from configuration")
        return tasks
    
    def run(self) -> Dict[str, Any]:
        """
        Main entry point for the unified backtester.
        
        Returns:
            Dict containing all results and outputs
        """
        
        try:
            self.logger.info("Starting unified backtester execution...")
            
            # Generate tasks from configuration
            tasks = self._generate_tasks_from_config()
            
            if not tasks:
                raise RuntimeError("No tasks generated from configuration")
              # Execute the workflow based on mode
            mode = getattr(self.config, 'mode', 'backtest')
            
            if mode == 'backtest':
                # Full workflow
                results = self.workflow_manager.execute_full_workflow(
                    tasks=tasks,
                    use_parallel=True,
                    skip_visualization=False
                )
            elif mode == 'analyze':
                # Analysis workflow
                results = self.workflow_manager.execute_analysis_workflow(
                    tasks=tasks,
                    use_parallel=True
                )
            elif mode == 'visualize':
                # Visualization workflow
                results = self.workflow_manager.execute_visualization_workflow(
                    tasks=tasks,
                    use_parallel=True
                )           
            elif mode == 'validate':
                # Validation only
                date_ranges = self.config.strategy.date_ranges if hasattr(self.config.strategy, 'date_ranges') else []
                tickers = self.config.strategy.tickers if hasattr(self.config.strategy, 'tickers') else []
                validation_passed = self.validate_data(date_ranges, tickers)
                results = {
                    'status': 'success' if validation_passed else 'error',
                    'validation_passed': validation_passed
                }
            elif mode == 'fetch':
                # Data fetching mode - can run interactively with zero arguments
                date_ranges = self.config.strategy.date_ranges if hasattr(self.config.strategy, 'date_ranges') else []
                tickers = self.config.strategy.tickers if hasattr(self.config.strategy, 'tickers') else []
                
                # If no arguments provided, launch interactive mode
                if not date_ranges and not tickers:
                    self.logger.info("No arguments provided for fetch mode - launching interactive interface")
                    from src.core.etl.data_fetcher import DataFetcher
                    fetcher = DataFetcher()
                    results = fetcher.interactive_fetch()
                else:
                    # Use existing workflow with provided arguments
                    results = self.workflow_manager.execute_fetch_workflow(date_ranges, tickers)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            # Validate execution was successful
            if not results or 'status' not in results:
                raise RuntimeError("Workflow execution failed - no results returned")
                
            if results['status'] != 'success':
                raise RuntimeError(f"Workflow execution failed: {results.get('error', 'Unknown error')}")
            
            # Log execution summary
            execution_time = datetime.now() - self.start_time
            self.logger.info(f"Unified backtester completed successfully in {execution_time}")
            self.logger.info(f"Output directory: {results.get('output_dir', 'Not specified')}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Unified backtester execution failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'status': 'error',
                'error': str(e),
                'execution_time': datetime.now() - self.start_time
            }
    
    def run_backtest(self, dates: List[str], tickers: List[str], strategies: List[str], 
                     optimization_params: Optional[Dict] = None, use_parallel: bool = True,
                     skip_visualization: bool = False) -> Dict[str, Any]:
        """
        Run comprehensive backtesting with analysis and visualization.
        This maintains API compatibility with the original monolithic runner.
        
        Args:
            dates: List of dates or date ranges
            tickers: List of ticker symbols
            strategies: List of strategy names
            optimization_params: Optional optimization parameters
            use_parallel: Whether to use parallel processing
            skip_visualization: Skip visualization generation
        """
        self.logger.info(f"Running backtest: {len(dates)} dates, {len(tickers)} tickers, {len(strategies)} strategies")
        
        # Convert dates to date ranges if needed
        date_ranges = []
        for date_str in dates:
            if '_to_' in date_str:
                date_ranges.append(date_str)
            else:
                date_ranges.append(f"{date_str}_to_{date_str}")
        
        # Validate data 
        if not self.validate_data(date_ranges, tickers):
            if getattr(self.config.validation, 'strict_mode', False):
                raise RuntimeError("Data validation failed in strict mode")
            else:
                self.logger.warning("Data validation failed, continuing in non-strict mode")
        
        # Create tasks for all strategy-ticker combinations
        tasks = []
        for date_range in date_ranges:
            for strategy_name in strategies:
                for ticker in tickers:
                    tasks.append((ticker, date_range, strategy_name, optimization_params))
        
        # Execute workflow based on current mode
        mode = getattr(self.config, 'mode', 'backtest')
        
        if mode == 'backtest':
            # Full workflow
            return self.workflow_manager.execute_full_workflow(
                tasks=tasks, 
                use_parallel=use_parallel, 
                skip_visualization=skip_visualization
            )
        elif mode == 'analyze':
            # Analysis workflow
            return self.workflow_manager.execute_analysis_workflow(
                tasks=tasks,
                use_parallel=use_parallel
            )
        elif mode == 'visualize':
            # Visualization workflow  
            return self.workflow_manager.execute_visualization_workflow(
                tasks=tasks,
                use_parallel=use_parallel
            )
        elif mode == 'validate':
            # Validation only
            return {"validation_passed": self.validate_data(date_ranges, tickers)}
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def validate_data(self, dates: List[str], tickers: List[str]) -> bool:
        """
        Comprehensive data validation with bias detection.
        This maintains API compatibility with the original monolithic runner.
        """
        return self.task_executor.data_validator.validate_data(dates, tickers)
    
    def run_backtest_task(self, args_tuple) -> Dict[str, Any]:
        """
        Execute individual backtest task.
        This maintains API compatibility with the original monolithic runner.
        """
        return self.task_executor.run_backtest_task(args_tuple)


def create_config_from_args() -> BacktestConfig:
    """Create configuration from command line arguments."""
    # Use CLI handler to parse arguments and create config
    cli_handler = CLIHandler()
    args = cli_handler.parse_arguments()
    
    # Validate arguments
    if not cli_handler.validate_arguments(args):
        sys.exit(1)
    
    return cli_handler.load_config(args)


def main():
    """Main entry point for the unified backtester."""
    try:
        # Parse command line arguments and create config
        config = create_config_from_args()
        
        # Create and run the unified backtester
        runner = UnifiedBacktesterRunner(config)
        results = runner.run()
        
        # Exit with appropriate code
        if results['status'] == 'success':
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"Failed to start unified backtester: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
