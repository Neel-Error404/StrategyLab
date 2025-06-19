"""
Portfolio-level visualizer - main orchestrator for portfolio charts.

This module provides the main interface for creating portfolio-level visualizations,
maintaining the same API as the original monolith while delegating to specialized
chart components.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..utils.base_visualizer import BaseVisualizer
from ..utils.data_loader import DataLoader


class PortfolioLevelVisualizer(BaseVisualizer):
    """
    Main visualizer for portfolio-level analysis.
    
    Creates comprehensive portfolio visualizations by coordinating with the original
    portfolio dashboard methods while maintaining the same API.
    
    This is a transitional implementation that delegates to the original methods
    in the monolith until we complete the full extraction in later phases.
    """
    
    def __init__(self, output_dir: Optional[Path] = None, trade_source: str = "auto"):
        """
        Initialize the portfolio-level visualizer.
        
        Args:
            output_dir: Directory to save visualizations  
            trade_source: Trade data source preference
        """
        super().__init__(output_dir, trade_source)
        self.data_loader = DataLoader(trade_source)
        
    def create_portfolio_dashboard(self, strategy_run_dir: Path, date_range: str, 
                                   tickers: List[str]) -> Dict[str, Path]:
        """
        Create comprehensive portfolio dashboard.
        
        This method preserves the exact same API and behavior as the original monolith.
        For now, it delegates to the original portfolio methods until we complete
        the extraction in later phases.
        
        Args:
            strategy_run_dir: Path to strategy run directory
            date_range: Date range string
            tickers: List of tickers in portfolio
            
        Returns:
            Dictionary of created visualization files
        """
        visualizations = {}
        
        try:
            # Limit tickers to max 12 for portfolio visualizations
            limited_tickers = self.data_loader.limit_portfolio_tickers(
                strategy_run_dir, date_range, tickers
            )
            if len(limited_tickers) < len(tickers):
                self.logger.info(f"Limited portfolio visualizations from {len(tickers)} to {len(limited_tickers)} tickers: {limited_tickers}")
            
            # Load portfolio data
            portfolio_data = self.data_loader.load_portfolio_data(
                strategy_run_dir, date_range, limited_tickers
            )
            
            if not portfolio_data or not portfolio_data.get('ticker_data'):
                self.logger.warning("No portfolio data available for visualization")
                return visualizations
            
            # For now, delegate to the original portfolio methods
            # These will be extracted in later phases of the refactoring
            visualizations.update(self._create_performance_dashboard_legacy(portfolio_data, date_range))
            visualizations.update(self._create_risk_dashboard_legacy(portfolio_data, date_range))
            visualizations.update(self._create_trade_analysis_dashboard_legacy(portfolio_data, date_range))
            visualizations.update(self._create_signal_analysis_dashboard_legacy(portfolio_data, date_range))
            visualizations.update(self._create_three_file_comparison_dashboard_legacy(portfolio_data, date_range))
            
            # Create master dashboard
            master_dashboard = self._create_master_dashboard_legacy(portfolio_data, date_range)
            if master_dashboard:
                visualizations['master_dashboard'] = master_dashboard
            
            self.logger.info(f"Created {len(visualizations)} portfolio visualizations")
            
        except Exception as e:
            self.logger.error(f"Error creating portfolio dashboard: {e}")
        
        return visualizations
        
    # Legacy delegation methods - these will be replaced in later phases
    def _create_performance_dashboard_legacy(self, portfolio_data: Dict, date_range: str) -> Dict[str, Path]:
        """Delegate to original performance dashboard method."""
        # Import here to avoid circular dependency during transition
        from ..portfolio_visualization import PortfolioVisualizer
        temp_visualizer = PortfolioVisualizer(self.output_dir.parent, self.trade_source)
        return temp_visualizer._create_performance_dashboard(portfolio_data, date_range)
        
    def _create_risk_dashboard_legacy(self, portfolio_data: Dict, date_range: str) -> Dict[str, Path]:
        """Delegate to original risk dashboard method."""
        from ..portfolio_visualization import PortfolioVisualizer
        temp_visualizer = PortfolioVisualizer(self.output_dir.parent, self.trade_source)
        return temp_visualizer._create_risk_dashboard(portfolio_data, date_range)
        
    def _create_trade_analysis_dashboard_legacy(self, portfolio_data: Dict, date_range: str) -> Dict[str, Path]:
        """Delegate to original trade analysis dashboard method."""
        from ..portfolio_visualization import PortfolioVisualizer
        temp_visualizer = PortfolioVisualizer(self.output_dir.parent, self.trade_source)
        return temp_visualizer._create_trade_analysis_dashboard(portfolio_data, date_range)
        
    def _create_signal_analysis_dashboard_legacy(self, portfolio_data: Dict, date_range: str) -> Dict[str, Path]:
        """Delegate to original signal analysis dashboard method."""
        from ..portfolio_visualization import PortfolioVisualizer
        temp_visualizer = PortfolioVisualizer(self.output_dir.parent, self.trade_source)
        return temp_visualizer._create_signal_analysis_dashboard(portfolio_data, date_range)
        
    def _create_three_file_comparison_dashboard_legacy(self, portfolio_data: Dict, date_range: str) -> Dict[str, Path]:
        """Delegate to original three file comparison dashboard method."""
        from ..portfolio_visualization import PortfolioVisualizer
        temp_visualizer = PortfolioVisualizer(self.output_dir.parent, self.trade_source)
        return temp_visualizer._create_three_file_comparison_dashboard(portfolio_data, date_range)
        
    def _create_master_dashboard_legacy(self, portfolio_data: Dict, date_range: str) -> Optional[Path]:
        """Delegate to original master dashboard method."""
        from ..portfolio_visualization import PortfolioVisualizer
        temp_visualizer = PortfolioVisualizer(self.output_dir.parent, self.trade_source)
        return temp_visualizer._create_master_dashboard(portfolio_data, date_range)
