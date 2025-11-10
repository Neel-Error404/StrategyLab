#!/usr/bin/env python3
"""
Workflow Manager Module for Unified Backtester

Manages different workflow modes (full, analysis, visualization, validation).
Handles mode-specific execution logic and orchestration.
"""

import logging
from typing import Dict, Any, List
from pathlib import Path

from .task_executor import TaskExecutor
from .analysis_engine import AnalysisEngine
from .visualization_engine import VisualizationEngine


class WorkflowManager:
    """
    Manages workflow execution for different modes.
    """
    
    def __init__(self, config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Initialize engines
        self.task_executor = TaskExecutor(config, logger)
        self.analysis_engine = AnalysisEngine(config, logger)
        self.visualization_engine = VisualizationEngine(config, logger)
    
    def execute_full_workflow(self, tasks: List, use_parallel: bool = True, skip_visualization: bool = False) -> Dict[str, Any]:
        """
        Execute the full workflow: backtest + analysis + visualization.
        Backtest runs only once, data is reused for analysis and visualization.
        """
        self.logger.info("🔄 Executing full workflow (backtest + analysis + visualization)")
        
        # Execute backtest tasks
        results = self.task_executor.execute_tasks(tasks, use_parallel)
        
        if not results:
            self.logger.warning("No backtest results generated")
            return {
                'status': 'error',
                'error': 'No backtest results generated'
            }
        
        try:
            # Run analysis using backtest results
            self.analysis_engine.run_portfolio_analysis(results)
            
            # Generate visualizations unless skipped
            if not skip_visualization:
                try:
                    self.visualization_engine.generate_visualizations(results)
                except Exception as viz_error:
                    self.logger.warning(f"⚠️  Visualization generation failed: {viz_error}")
                    self.logger.info("Continuing without visualizations")
            else:
                self.logger.info("⏭️  Skipping visualization generation")
            
            self.logger.info("✅ Full workflow completed successfully")
            
            # Return results with success status
            workflow_results = {
                'status': 'success',
                'output_dir': f"{self.config.base_dir}/{self.config.output.output_dir}/{self.config.run_id}"
            }
            # Merge the actual backtest results
            workflow_results.update(results)
            return workflow_results
            
        except Exception as e:
            self.logger.error(f"❌ Error in full workflow: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def execute_analysis_workflow(self, tasks: List, use_parallel: bool = True) -> Dict[str, Any]:
        """
        Execute analysis workflow: backtest + analysis only.
        """
        self.logger.info("🔄 Executing analysis workflow (backtest + analysis)")
        
        # Execute backtest tasks
        results = self.task_executor.execute_tasks(tasks, use_parallel)
        
        if not results:
            self.logger.warning("No backtest results available for analysis")
            return {
                'status': 'error',
                'error': 'No backtest results available for analysis'
            }
        
        try:
            # Run analysis using backtest results
            self.analysis_engine.run_portfolio_analysis(results)
            
            self.logger.info("✅ Analysis workflow completed successfully")
            
            # Return results with success status
            workflow_results = {
                'status': 'success',
                'output_dir': f"{self.config.base_dir}/{self.config.output.output_dir}/{self.config.run_id}"
            }
            # Merge the actual backtest results
            workflow_results.update(results)
            return workflow_results
            
        except Exception as e:
            self.logger.error(f"❌ Error in analysis workflow: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def execute_visualization_workflow(self, tasks: List, use_parallel: bool = True) -> Dict[str, Any]:
        """
        Execute visualization workflow: backtest + visualization only.
        """
        self.logger.info("🔄 Executing visualization workflow (backtest + visualization)")
        
        # Execute backtest tasks
        results = self.task_executor.execute_tasks(tasks, use_parallel)
        
        if not results:
            self.logger.warning("No backtest results available for visualization")
            return {
                'status': 'error',
                'error': 'No backtest results available for visualization'
            }
        
        try:
            # Generate visualizations using backtest results
            self.visualization_engine.generate_visualizations(results)
            
            self.logger.info("✅ Visualization workflow completed successfully")
            
            # Return results with success status
            workflow_results = {
                'status': 'success',
                'output_dir': f"{self.config.base_dir}/{self.config.output.output_dir}/{self.config.run_id}"
            }
            # Merge the actual backtest results
            workflow_results.update(results)
            return workflow_results
            
        except Exception as e:
            self.logger.error(f"❌ Error in visualization workflow: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def execute_validation_workflow(self, dates: List[str], tickers: List[str]) -> Dict[str, Any]:
        """
        Execute validation workflow: data validation only.
        """
        self.logger.info("🔍 Executing validation workflow (data validation only)")
        
        try:
            # Use task executor's validation capabilities
            validation_passed = self.task_executor.validate_data(dates, tickers)
            
            result = {
                "status": "success",
                "validation_passed": validation_passed
            }
            
            if validation_passed:
                self.logger.info("✅ Data validation passed")
            else:
                self.logger.warning("⚠️  Data validation failed")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error in validation workflow: {e}")            
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def execute_fetch_workflow(self, dates: List[str], tickers: List[str]) -> Dict[str, Any]:
        """
        Execute fetch workflow: download market data from broker APIs.
        """
        self.logger.info("📥 Executing fetch workflow (market data download)")
        
        try:
            # Use the FetchModeHandler via the mode handler factory
            from src.runners.workflow.mode_handlers import ModeHandlerFactory
            
            fetch_handler = ModeHandlerFactory.create_handler(
                'fetch', 
                self.config, 
                self.task_executor
            )
            
            result = fetch_handler.execute(dates=dates, tickers=tickers)
            
            if result.get('status') == 'error':
                self.logger.error("❌ Data fetch failed")
            else:
                total_files = result.get('summary', {}).get('total_files_created', 0)
                self.logger.info(f"✅ Data fetch completed: {total_files} files created")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error in fetch workflow: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def create_task_list(self, date_ranges: List[str], tickers: List[str], strategies: List[str], optimization_params=None) -> List:
        """
        Create task list for execution.
        """
        tasks = []
        for date_range in date_ranges:
            for strategy_name in strategies:
                for ticker in tickers:
                    tasks.append((ticker, date_range, strategy_name, optimization_params))
        
        self.logger.info(f"📝 Created {len(tasks)} tasks for processing")
        return tasks
