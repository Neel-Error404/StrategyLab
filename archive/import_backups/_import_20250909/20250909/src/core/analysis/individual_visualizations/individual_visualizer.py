"""
Individual ticker visualizer - main orchestrator for individual ticker charts.

This module provides the main interface for creating individual ticker visualizations,
maintaining the same API as the original monolith while delegating to specialized
chart components.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..utils.base_visualizer import BaseVisualizer
from ..utils.data_loader import DataLoader
from .performance_summary import PerformanceSummaryChart
from .trade_distribution import TradeDistributionChart
from .trade_timeline import TradeTimelineChart


class IndividualTickerVisualizer(BaseVisualizer):
    """
    Main visualizer for individual ticker analysis.
    
    Creates comprehensive individual ticker visualizations by coordinating
    specialized chart components while maintaining the original API.
    """
    
    def __init__(self, output_dir: Optional[Path] = None, trade_source: str = "auto", 
                 use_enhanced_naming: bool = False):
        """
        Initialize the individual ticker visualizer.
        
        Args:
            output_dir: Directory to save visualizations  
            trade_source: Trade data source preference
            use_enhanced_naming: Whether to use enhanced trader-focused naming scheme
        """
        super().__init__(output_dir, trade_source, use_enhanced_naming)
        self.data_loader = DataLoader(trade_source)
        
        # Initialize chart components with enhanced naming support
        self.performance_chart = PerformanceSummaryChart(trade_source, use_enhanced_naming)
        self.distribution_chart = TradeDistributionChart(trade_source, use_enhanced_naming)
        self.timeline_chart = TradeTimelineChart(trade_source, use_enhanced_naming)
        
    def create_individual_ticker_dashboard(self, strategy_run_dir: Path, ticker: str,
                                           date_range: str) -> Dict[str, Path]:
        """
        Create comprehensive individual ticker visualizations.
        
        This method preserves the exact same API and behavior as the original monolith.
        
        Creates three types of individual ticker analysis:
        1. Performance Summary - Individual ticker performance metrics
        2. Trade Distribution - Trade patterns and distribution analysis  
        3. Trade Timeline - Trade execution timeline and patterns
        
        Args:
            strategy_run_dir: Path to strategy run directory
            ticker: Ticker symbol
            date_range: Date range string
            
        Returns:
            Dictionary of created visualization files
        """
        visualizations = {}
        
        try:
            # Create ticker-specific subdirectory
            ticker_dir = self.output_dir / ticker
            self._ensure_directory(ticker_dir)
            
            # Load ticker data
            ticker_data = self.data_loader.load_individual_ticker_data(
                strategy_run_dir, ticker, date_range
            )
            
            if not ticker_data or not self._has_sufficient_data(ticker_data):
                self.logger.warning(f"No data available for {ticker} individual visualization")
                return visualizations
            
            self._log_data_availability(ticker_data, f"Individual visualization for {ticker}")
            
            # 1. Create Performance Summary
            performance_path = self.performance_chart.create(
                ticker_data, ticker, date_range, ticker_dir
            )
            if performance_path:
                visualizations['performance_summary'] = performance_path
            
            # 2. Create Trade Distribution Analysis  
            distribution_path = self.distribution_chart.create(
                ticker_data, ticker, date_range, ticker_dir
            )
            if distribution_path:
                visualizations['trade_distribution'] = distribution_path
            
            # 3. Create Trade Timeline
            timeline_path = self.timeline_chart.create(
                ticker_data, ticker, date_range, ticker_dir
            )
            if timeline_path:
                visualizations['trade_timeline'] = timeline_path
            
            self.logger.info(f"Created {len(visualizations)} individual visualizations for {ticker}")
            
        except Exception as e:
            self.logger.error(f"Error creating individual ticker dashboard for {ticker}: {e}")
        
        return visualizations
        
    def _has_sufficient_data(self, ticker_data: Dict[str, Any]) -> bool:
        """
        Check if ticker data is sufficient for visualization.
        
        Args:
            ticker_data: Dictionary containing ticker data
            
        Returns:
            True if sufficient data is available
        """
        # At minimum, we need some form of data
        has_base_data = (
            ticker_data.get('base_data') is not None and 
            not ticker_data['base_data'].empty
        )
        
        has_trade_data = (
            ticker_data.get('active_trades') is not None and 
            not ticker_data['active_trades'].empty
        )
        
        has_analytics = ticker_data.get('analytics') is not None
        
        # We can create visualizations with any of these data sources
        return has_base_data or has_trade_data or has_analytics
