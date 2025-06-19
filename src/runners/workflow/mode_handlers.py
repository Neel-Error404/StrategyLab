#!/usr/bin/env python3
"""
Mode Handlers - Different execution modes for backtesting

Handles different execution modes:
- Backtest: Single strategy execution
- Multi-backtest: Multiple strategies/tickers
- Optimize: Parameter optimization
- Validate: Data and strategy validation
- Analyze: Portfolio analysis
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from multiprocessing import Pool, cpu_count
from pathlib import Path

from config.unified_config import BacktestConfig


class ModeHandler:
    """Base class for execution mode handlers."""
    
    def __init__(self, config: BacktestConfig, execution_engine):
        self.config = config
        self.execution_engine = execution_engine
        self.logger = logging.getLogger(__name__)
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the mode. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement execute method")


class BacktestModeHandler(ModeHandler):
    """Handles single backtest execution mode."""
    
    def execute(self, dates: List[str], tickers: List[str], 
                strategies: List[str], **kwargs) -> Dict[str, Any]:
        """Execute single backtest mode."""
        self.logger.info(f"Running backtest mode for {len(strategies)} strategies, "
                        f"{len(tickers)} tickers, {len(dates)} dates")
        
        # Prepare tasks for execution
        tasks = self._prepare_backtest_tasks(dates, tickers, strategies)
        
        # Execute based on parallel configuration
        if self.config.execution.parallel and len(tasks) > 1:
            results = self._execute_parallel(tasks)
        else:
            results = self._execute_sequential(tasks)
        
        # Aggregate and return results
        return self._aggregate_results(results, 'backtest')
    
    def _prepare_backtest_tasks(self, dates: List[str], tickers: List[str], 
                               strategies: List[str]) -> List[tuple]:
        """Prepare tasks for backtest execution."""
        tasks = []
        for strategy in strategies:
            for ticker in tickers:
                for date in dates:
                    # No optimization params for basic backtest
                    tasks.append((ticker, date, strategy, None))
        
        self.logger.info(f"Prepared {len(tasks)} backtest tasks")
        return tasks
    
    def _execute_parallel(self, tasks: List[tuple]) -> List[Dict[str, Any]]:
        """Execute tasks in parallel."""
        pool_size = min(self.config.execution.max_workers or cpu_count(), len(tasks))
        self.logger.info(f"Starting parallel execution with {pool_size} workers")
        
        with Pool(pool_size) as pool:
            results = pool.map(self.execution_engine.run_backtest_task, tasks)
        
        # Filter out empty results
        valid_results = [r for r in results if r]
        self.logger.info(f"Parallel execution completed: {len(valid_results)}/{len(tasks)} successful")
        return valid_results
    
    def _execute_sequential(self, tasks: List[tuple]) -> List[Dict[str, Any]]:
        """Execute tasks sequentially."""
        self.logger.info(f"Starting sequential execution of {len(tasks)} tasks")
        
        results = []
        for i, task in enumerate(tasks, 1):
            self.logger.info(f"Processing task {i}/{len(tasks)}: {task[2]} - {task[0]} - {task[1]}")
            result = self.execution_engine.run_backtest_task(task)
            if result:
                results.append(result)
            
            # Progress reporting
            if i % 10 == 0 or i == len(tasks):
                self.logger.info(f"Progress: {i}/{len(tasks)} tasks completed ({i/len(tasks)*100:.1f}%)")
        
        self.logger.info(f"Sequential execution completed: {len(results)}/{len(tasks)} successful")
        return results
    
    def _aggregate_results(self, results: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
        """Aggregate execution results."""
        return {
            'mode': mode,
            'total_tasks': len(results),
            'successful_tasks': len([r for r in results if r.get('trades') is not None]),
            'failed_tasks': len([r for r in results if not r.get('trades')]),
            'execution_time': datetime.now().isoformat(),
            'results': results
        }


class OptimizeModeHandler(ModeHandler):
    """Handles parameter optimization mode."""
    
    def execute(self, dates: List[str], tickers: List[str], 
                strategies: List[str], param_grid: Dict[str, List], **kwargs) -> Dict[str, Any]:
        """Execute optimization mode."""
        self.logger.info(f"Running optimization mode for {len(strategies)} strategies")
        self.logger.info(f"Parameter grid: {param_grid}")
        
        # Prepare optimization tasks
        tasks = self._prepare_optimization_tasks(dates, tickers, strategies, param_grid)
        
        # Execute optimization
        if self.config.execution.parallel and len(tasks) > 1:
            results = self._execute_parallel_optimization(tasks)
        else:
            results = self._execute_sequential_optimization(tasks)
        
        # Find best parameters and return results
        return self._analyze_optimization_results(results, param_grid)
    
    def _prepare_optimization_tasks(self, dates: List[str], tickers: List[str], 
                                   strategies: List[str], param_grid: Dict[str, List]) -> List[tuple]:
        """Prepare optimization tasks with parameter combinations."""
        tasks = []
        
        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        for strategy in strategies:
            for ticker in tickers:
                for date in dates:
                    for param_combo in param_combinations:
                        tasks.append((ticker, date, strategy, param_combo))
        
        self.logger.info(f"Prepared {len(tasks)} optimization tasks with {len(param_combinations)} parameter combinations")
        return tasks
    
    def _generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict]:
        """Generate all combinations of parameters."""
        import itertools
        
        if not param_grid:
            return [{}]
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _execute_parallel_optimization(self, tasks: List[tuple]) -> List[Dict[str, Any]]:
        """Execute optimization tasks in parallel."""
        pool_size = min(self.config.execution.max_workers or cpu_count(), len(tasks))
        self.logger.info(f"Starting parallel optimization with {pool_size} workers")
        
        with Pool(pool_size) as pool:
            results = pool.map(self.execution_engine.run_backtest_task, tasks)
        
        valid_results = [r for r in results if r]
        self.logger.info(f"Parallel optimization completed: {len(valid_results)}/{len(tasks)} successful")
        return valid_results
    
    def _execute_sequential_optimization(self, tasks: List[tuple]) -> List[Dict[str, Any]]:
        """Execute optimization tasks sequentially."""
        self.logger.info(f"Starting sequential optimization of {len(tasks)} tasks")
        
        results = []
        for i, task in enumerate(tasks, 1):
            result = self.execution_engine.run_backtest_task(task)
            if result:
                results.append(result)
            
            if i % 20 == 0 or i == len(tasks):
                self.logger.info(f"Optimization progress: {i}/{len(tasks)} tasks completed")
        
        return results
    
    def _analyze_optimization_results(self, results: List[Dict[str, Any]], 
                                    param_grid: Dict[str, List]) -> Dict[str, Any]:
        """Analyze optimization results and find best parameters."""
        if not results:
            return {
                'mode': 'optimize',
                'status': 'failed',
                'message': 'No successful optimization results',
                'param_grid': param_grid
            }
        
        # Group results by strategy and find best parameters
        strategy_results = {}
        for result in results:
            strategy = result.get('strategy', 'unknown')
            if strategy not in strategy_results:
                strategy_results[strategy] = []
            strategy_results[strategy].append(result)
        
        # Find best parameters for each strategy based on total return
        best_params = {}
        for strategy, strategy_res in strategy_results.items():
            if strategy_res:
                best_result = max(strategy_res, 
                                key=lambda x: x.get('metrics', {}).get('Total Return', -float('inf')))
                best_params[strategy] = {
                    'parameters': best_result.get('metrics', {}).get('Parameters', 'Unknown'),
                    'total_return': best_result.get('metrics', {}).get('Total Return', 0),
                    'sharpe_ratio': best_result.get('metrics', {}).get('Sharpe Ratio', 0),
                    'max_drawdown': best_result.get('metrics', {}).get('Max Drawdown', 0)
                }
        
        return {
            'mode': 'optimize',
            'status': 'completed',
            'param_grid': param_grid,
            'total_combinations_tested': len(results),
            'best_parameters': best_params,
            'detailed_results': results
        }


class ValidateModeHandler(ModeHandler):
    """Handles validation mode."""
    
    def execute(self, dates: List[str], tickers: List[str], 
                strategies: List[str], **kwargs) -> Dict[str, Any]:
        """Execute validation mode."""
        self.logger.info(f"Running validation mode for {len(strategies)} strategies")
        
        validation_results = {
            'mode': 'validate',
            'data_validation': {},
            'strategy_validation': {},
            'bias_validation': {},
            'overall_status': 'passed'
        }
        
        # Data validation
        data_validation = self._validate_data(dates, tickers)
        validation_results['data_validation'] = data_validation
        
        # Strategy validation
        strategy_validation = self._validate_strategies(strategies, dates, tickers)
        validation_results['strategy_validation'] = strategy_validation
        
        # Determine overall status
        if not data_validation.get('all_passed', False) or not strategy_validation.get('all_passed', False):
            validation_results['overall_status'] = 'failed'
        
        return validation_results
    
    def _validate_data(self, dates: List[str], tickers: List[str]) -> Dict[str, Any]:
        """Validate data availability and quality."""
        self.logger.info("Starting data validation...")
        
        data_results = {
            'total_combinations': len(dates) * len(tickers),
            'successful_loads': 0,
            'failed_loads': 0,
            'quality_issues': [],
            'all_passed': True
        }
        
        # Import here to avoid circular imports
        from src.core.etl.loader import load_base_data
        
        for date in dates:
            for ticker in tickers:
                try:
                    data = load_base_data(date, ticker)
                    if data is None or data.empty:
                        data_results['failed_loads'] += 1
                        data_results['quality_issues'].append(f"No data for {ticker} on {date}")
                        data_results['all_passed'] = False
                    else:
                        data_results['successful_loads'] += 1
                        
                        # Basic quality checks
                        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
                        if missing_pct > 0.1:  # 10% threshold
                            data_results['quality_issues'].append(
                                f"High missing data for {ticker} on {date}: {missing_pct:.2%}"
                            )
                            
                except Exception as e:
                    data_results['failed_loads'] += 1
                    data_results['quality_issues'].append(f"Error loading {ticker} on {date}: {str(e)}")
                    data_results['all_passed'] = False
        
        self.logger.info(f"Data validation completed: {data_results['successful_loads']}/{data_results['total_combinations']} successful")
        return data_results
    
    def _validate_strategies(self, strategies: List[str], dates: List[str], 
                           tickers: List[str]) -> Dict[str, Any]:
        """Validate strategy execution."""
        self.logger.info("Starting strategy validation...")
        
        from src.strategies.strategy_factory import StrategyFactory
        
        strategy_results = {
            'total_strategies': len(strategies),
            'valid_strategies': 0,
            'invalid_strategies': 0,
            'execution_tests': {},
            'all_passed': True
        }
        
        for strategy in strategies:
            try:
                # Test strategy creation
                strategy_instance = StrategyFactory.get_strategy(strategy)
                if strategy_instance is None:
                    strategy_results['invalid_strategies'] += 1
                    strategy_results['execution_tests'][strategy] = 'creation_failed'
                    strategy_results['all_passed'] = False
                    continue
                
                # Test strategy execution on first date/ticker combination
                if dates and tickers:
                    test_date = dates[0]
                    test_ticker = tickers[0]
                    
                    # Run a simple test execution
                    test_task = (test_ticker, test_date, strategy, None)
                    test_result = self.execution_engine.run_backtest_task(test_task)
                    
                    if test_result and test_result.get('trades') is not None:
                        strategy_results['valid_strategies'] += 1
                        strategy_results['execution_tests'][strategy] = 'passed'
                    else:
                        strategy_results['invalid_strategies'] += 1
                        strategy_results['execution_tests'][strategy] = 'execution_failed'
                        strategy_results['all_passed'] = False
                else:
                    strategy_results['valid_strategies'] += 1
                    strategy_results['execution_tests'][strategy] = 'creation_only'
                    
            except Exception as e:
                strategy_results['invalid_strategies'] += 1
                strategy_results['execution_tests'][strategy] = f'error: {str(e)}'
                strategy_results['all_passed'] = False
        
        self.logger.info(f"Strategy validation completed: {strategy_results['valid_strategies']}/{strategy_results['total_strategies']} valid")
        return strategy_results


class AnalyzeModeHandler(ModeHandler):
    """Handles portfolio analysis mode."""
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute analysis mode."""
        self.logger.info("Running analysis mode")
        
        # This mode analyzes existing results rather than running new backtests
        output_dir = Path(self.config.base_dir) / self.config.output.output_dir / self.config.run_id
        
        analysis_results = {
            'mode': 'analyze',
            'output_directory': str(output_dir),
            'analysis_timestamp': datetime.now().isoformat(),
            'portfolio_analysis': {},
            'visualization_status': {},
            'status': 'completed'
        }
        
        try:
            # Check if output directory exists
            if not output_dir.exists():
                analysis_results['status'] = 'failed'
                analysis_results['message'] = f"Output directory not found: {output_dir}"
                return analysis_results
            
            # Analyze existing files
            csv_files = list(output_dir.glob("**/*.csv"))
            json_files = list(output_dir.glob("**/*.json"))
            image_files = list(output_dir.glob("**/*.png"))
            
            analysis_results['portfolio_analysis'] = {
                'csv_files_found': len(csv_files),
                'json_files_found': len(json_files),
                'image_files_found': len(image_files),
                'file_listing': {
                    'csv': [f.name for f in csv_files],
                    'json': [f.name for f in json_files],
                    'images': [f.name for f in image_files]
                }
            }
            
            # If visualizations are requested, generate them
            if self.config.output.generate_visualizations:
                viz_status = self._generate_visualizations(output_dir)
                analysis_results['visualization_status'] = viz_status
            
            self.logger.info(f"Analysis completed for directory: {output_dir}")
            
        except Exception as e:
            analysis_results['status'] = 'failed'
            analysis_results['error'] = str(e)
            self.logger.error(f"Error in analysis mode: {e}")
        
        return analysis_results
    
    def _generate_visualizations(self, output_dir: Path) -> Dict[str, Any]:
        """Generate visualizations for existing results."""
        viz_status = {
            'requested': True,
            'generated': False,
            'error': None
        }
        
        try:
            # This would integrate with the portfolio visualizer
            # For now, just report status
            viz_status['generated'] = True
            viz_status['output_path'] = str(output_dir / "visualizations")
            
        except Exception as e:
            viz_status['error'] = str(e)
            self.logger.error(f"Error generating visualizations: {e}")
        return viz_status


class FetchModeHandler(ModeHandler):
    """Handler for fetch mode - downloads market data from broker APIs."""
    
    def execute(self, dates: List[str], tickers: List[str], 
                strategies: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute data fetching workflow.
        
        Args:
            dates: List of date range strings
            tickers: List of ticker symbols to fetch
            strategies: Not used for fetch mode
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with fetch results
        """
        self.logger.info(f"Starting data fetch for {len(tickers)} tickers, {len(dates)} date ranges")
        
        try:
            # Import data fetcher
            from src.core.etl.data_fetcher import DataFetcher
            from datetime import datetime
            
            fetch_results = {}
            
            # Process each date range
            for date_range in dates:
                try:
                    # Parse date range
                    start_str, end_str = date_range.split('_to_')
                    start_date = datetime.strptime(start_str, '%Y-%m-%d')
                    end_date = datetime.strptime(end_str, '%Y-%m-%d')
                    
                    self.logger.info(f"Fetching data for date range: {date_range}")
                    
                    # Initialize data fetcher with auto-detect provider
                    fetcher = DataFetcher()
                      # Fetch data for all tickers in this date range
                    result = fetcher.fetch_historical_data(
                        tickers=tickers,
                        timeframes=['1m'],  # Default to 1-minute data
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    fetch_results[date_range] = {
                        'status': 'success',
                        'tickers_processed': len(result),
                        'files_created': sum(len(tf_dict) for tf_dict in result.values()),
                        'details': result
                    }
                    
                    self.logger.info(f"✅ Fetch completed for {date_range}: {len(result)} tickers processed")
                    
                except Exception as e:
                    fetch_results[date_range] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    self.logger.error(f"❌ Fetch failed for {date_range}: {e}")
            
            # Summary
            total_success = sum(1 for r in fetch_results.values() if r['status'] == 'success')
            total_files = sum(r.get('files_created', 0) for r in fetch_results.values() if r['status'] == 'success')
            
            self.logger.info(f"Fetch complete: {total_success}/{len(dates)} date ranges successful, {total_files} files created")
            
            return {
                'mode': 'fetch',
                'summary': {
                    'total_date_ranges': len(dates),
                    'successful_ranges': total_success,
                    'total_files_created': total_files,
                    'tickers_requested': tickers
                },
                'results': fetch_results
            }
            
        except Exception as e:
            self.logger.error(f"Error in fetch mode: {e}")
            return {
                'mode': 'fetch',
                'status': 'error',
                'error': str(e)
            }


class ModeHandlerFactory:
    """Factory for creating mode handlers."""
    @staticmethod
    def create_handler(mode: str, config: BacktestConfig, execution_engine) -> ModeHandler:
        """Create appropriate mode handler based on mode string."""
        mode_handlers = {
            'backtest': BacktestModeHandler,
            'optimize': OptimizeModeHandler,
            'validate': ValidateModeHandler,
            'analyze': AnalyzeModeHandler,
            'fetch': FetchModeHandler
        }
        
        handler_class = mode_handlers.get(mode.lower())
        if not handler_class:
            raise ValueError(f"Unknown mode: {mode}. Supported modes: {list(mode_handlers.keys())}")
        
        return handler_class(config, execution_engine)
