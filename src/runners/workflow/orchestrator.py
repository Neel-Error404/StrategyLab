"""
Workflow orchestration for the unified backtester.
Coordinates execution between different components and modes.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from config.unified_config import BacktestConfig


class WorkflowOrchestrator:
    """
    Orchestrates the overall workflow for different execution modes.
    Coordinates between execution engine, analyzers, and visualizers.
    """
    
    def __init__(self, config: BacktestConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def orchestrate_workflow(self, mode: str, tasks: List, **kwargs) -> Dict[str, Any]:
        """
        Orchestrate the workflow based on the specified mode.
        
        Args:
            mode: Execution mode (backtest, analyze, visualize, validate)
            tasks: List of tasks to execute
            **kwargs: Additional workflow parameters
            
        Returns:
            Workflow results
        """
        self.logger.info(f"🔄 Orchestrating {mode} workflow with {len(tasks)} tasks")
        
        if mode == 'backtest':
            return self._orchestrate_full_workflow(tasks, **kwargs)
        elif mode == 'analyze':
            return self._orchestrate_analysis_workflow(tasks, **kwargs)
        elif mode == 'visualize':
            return self._orchestrate_visualization_workflow(tasks, **kwargs)
        elif mode == 'validate':
            return self._orchestrate_validation_workflow(tasks, **kwargs)
        else:
            raise ValueError(f"Unknown workflow mode: {mode}")
    
    def _orchestrate_full_workflow(self, tasks: List, **kwargs) -> Dict[str, Any]:
        """Orchestrate full workflow: backtest + analysis + visualization."""
        from .execution_engine import ExecutionEngine
        from .mode_handlers import ModeHandler
        
        # Initialize components
        execution_engine = ExecutionEngine(self.config, self.logger)
        mode_handler = ModeHandler(self.config, self.logger)
        
        # Execute backtest tasks
        results = execution_engine.execute_tasks(tasks, kwargs.get('use_parallel', False))
        
        # Run analysis and visualization if not skipped
        if not kwargs.get('skip_visualization', False):
            mode_handler.run_analysis_and_visualization(results)
        
        return results
    
    def _orchestrate_analysis_workflow(self, tasks: List, **kwargs) -> Dict[str, Any]:
        """Orchestrate analysis-only workflow."""
        from .execution_engine import ExecutionEngine
        from .mode_handlers import ModeHandler
        
        execution_engine = ExecutionEngine(self.config, self.logger)
        mode_handler = ModeHandler(self.config, self.logger)
        
        # Execute backtest tasks
        results = execution_engine.execute_tasks(tasks, kwargs.get('use_parallel', False))
        
        # Run analysis only
        mode_handler.run_analysis_only(results)
        
        return results
    
    def _orchestrate_visualization_workflow(self, tasks: List, **kwargs) -> Dict[str, Any]:
        """Orchestrate visualization-only workflow."""
        from .execution_engine import ExecutionEngine
        from .mode_handlers import ModeHandler
        
        execution_engine = ExecutionEngine(self.config, self.logger)
        mode_handler = ModeHandler(self.config, self.logger)
        
        # Execute backtest tasks
        results = execution_engine.execute_tasks(tasks, kwargs.get('use_parallel', False))
        
        # Run visualization only        mode_handler.run_visualization_only(results)
        
        return results
    
    def _orchestrate_validation_workflow(self, tasks: List, **kwargs) -> Dict[str, Any]:
        """Orchestrate validation-only workflow."""
        from ..components.validator import DataValidator
        
        validator = DataValidator(self.logger)
        
        # Extract dates and tickers from tasks
        dates = list(set(task[1] for task in tasks))  # task[1] is date_range
        tickers = list(set(task[0] for task in tasks))  # task[0] is ticker
        
        validation_passed = validator.validate_data(dates, tickers)
        
        return {"validation_passed": validation_passed}
