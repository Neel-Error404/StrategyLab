#!/usr/bin/env python3
"""
Analysis Engine Module for Unified Backtester

Handles portfolio-level analysis and metrics calculation.
Processes backtest results to generate comprehensive analysis reports.
Uses the EnhancedOutputOrchestrator for comprehensive output generation.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import json

# Import enhanced output system
from src.core.output.enhanced_output_orchestrator import EnhancedOutputOrchestrator
# Keep the simple analyzer as fallback
from src.runners.components.analyzer import PortfolioAnalyzer


class AnalysisEngine:
    """
    Manages portfolio analysis and metrics generation.
    Uses the EnhancedOutputOrchestrator for comprehensive output generation.
    """
    
    def __init__(self, config, logger: logging.Logger):
        self.config = config
        self.logger = logger        # Initialize enhanced output orchestrator
        base_output_dir = Path(config.base_dir) / config.output.output_dir
        self.enhanced_orchestrator = EnhancedOutputOrchestrator(base_output_dir)        
        # Keep simple analyzer as fallback for compatibility
        self.portfolio_analyzer = PortfolioAnalyzer(config)
    
    def run_portfolio_analysis(self, results: Dict[str, Any]) -> None:
        """
        Run comprehensive portfolio analysis using the enhanced output orchestrator.
        
        Args:
            results: Structured dictionary of backtest results
        """
        self.logger.info("📊 Running comprehensive portfolio analysis")
        
        if not results:
            self.logger.warning("No backtest results available for analysis")
            return
        
        try:            # Extract metadata from results
            from src.runners.utils.naming import extract_strategy_and_date_from_results
            strategy_name, date_range = extract_strategy_and_date_from_results(results)
            tickers = self._extract_tickers(results)
            
            self.logger.info(f"Processing results for {strategy_name} on {tickers} for {date_range}")            # Use enhanced orchestrator for comprehensive output
            self.logger.info("Using enhanced output orchestrator for comprehensive analysis")
            output_results = self.enhanced_orchestrator.process_complete_backtest_results(
                strategy_name=strategy_name,
                date_range=date_range, 
                tickers=tickers,
                results_data=results,
                run_id=self.config.run_id,
                strategy_run_dir=getattr(self.config, 'strategy_run_dir', None)
            )
            
            self.logger.info(f"Enhanced analysis completed. Output directory: {output_results['strategy_run_dir']}")
            
            self.logger.info("✅ Comprehensive portfolio analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in portfolio analysis: {e}")
            # Fallback to simple analysis if enhanced fails
            self.logger.info("Falling back to simple analysis...")
            self._run_simple_analysis(results)
    
    def _extract_strategy_name(self, results: Dict[str, Any]) -> str:
        """Extract strategy name from results."""
        if results:
            # Get first strategy name from results keys
            return list(results.keys())[0]
        return "unknown_strategy"
    
    def _extract_date_range(self, results: Dict[str, Any]) -> str:
        """Extract date range from results."""
        if results:
            strategy_results = list(results.values())[0]
            if strategy_results:
                # Get first date range from strategy results
                return list(strategy_results.keys())[0]
        return "unknown_date_range"
    
    def _extract_tickers(self, results: Dict[str, Any]) -> List[str]:
        """Extract ticker list from results."""
        tickers = []
        if results:
            strategy_results = list(results.values())[0]
            if strategy_results:
                date_results = list(strategy_results.values())[0]
                if date_results:
                    tickers = list(date_results.keys())
        return tickers
    
    def _run_simple_analysis(self, results: Dict[str, Any]) -> None:
        """Run simple analysis as fallback."""
        try:
            analysis_results = self.portfolio_analyzer.analyze_results(results)
            if analysis_results:
                self._save_simple_analysis_results(analysis_results)
                self._log_analysis_summary(analysis_results)
        except Exception as e:
            self.logger.error(f"Simple analysis also failed: {e}")
    
    def _save_simple_analysis_results(self, analysis_results: Dict[str, Any]) -> None:
        """Save simple analysis results to output directory."""
        # Create output directory
        output_dir = Path(self.config.base_dir) / self.config.output.output_dir / self.config.run_id / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate trade reports using the analyzer
        if 'trades' in analysis_results and analysis_results['trades']:
            self.portfolio_analyzer.generate_trade_reports(
                analysis_results['trades'], 
                output_dir
            )
            self.logger.info(f"Trade reports saved to {output_dir}")
        
        # Save portfolio metrics
        if 'portfolio_metrics' in analysis_results:
            metrics_file = output_dir / "portfolio_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(analysis_results['portfolio_metrics'], f, indent=2)
            self.logger.info(f"Portfolio metrics saved to {metrics_file}")
    
    def _log_analysis_summary(self, analysis_results: Dict[str, Any]) -> None:
        """Log a summary of the analysis results."""
        if 'portfolio_metrics' in analysis_results:
            metrics = analysis_results['portfolio_metrics']
            
            self.logger.info("=" * 50)
            self.logger.info("📊 PORTFOLIO ANALYSIS SUMMARY")
            self.logger.info("=" * 50)
            
            # Basic metrics
            total_trades = metrics.get('total_trades', 0)
            win_rate = metrics.get('win_rate', 0) * 100
            total_return = metrics.get('total_return', 0)
            
            self.logger.info(f"Total Trades: {total_trades}")
            self.logger.info(f"Win Rate: {win_rate:.2f}%")
            self.logger.info(f"Total Return: {total_return:.2f}%")
            
            # Risk metrics
            max_drawdown = metrics.get('max_drawdown', 0)
            risk_adj_return = metrics.get('risk_adjusted_return', 0)
            
            self.logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
            self.logger.info(f"Risk-Adjusted Return: {risk_adj_return:.2f}")
            
            self.logger.info("=" * 50)
