"""
Base classes for portfolio visualization system.

This module provides the foundational classes and interfaces used by both
individual ticker and portfolio-level visualizations.
"""

import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json

from .chart_styling import ChartStyler


class BaseVisualizer:
    """
    Base class for all visualization components.
    
    Provides common functionality for data loading, styling, and file management    shared across individual ticker and portfolio visualizations.
    """
    
    def __init__(self, output_dir: Optional[Path] = None, trade_source: str = "auto", 
                 use_enhanced_naming: bool = False):
        """
        Initialize the base visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            trade_source: Trade data source for visualizations:
                         - "strategy_trades": Use raw strategy output (always available)
                         - "risk_approved_trades": Use post-risk-management trades (may be empty)
                         - "auto": Try risk_approved_trades first, fallback to strategy_trades if empty
            use_enhanced_naming: Whether to use enhanced trader-focused naming scheme
        """
        self.output_dir = output_dir or Path("visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trade_source = trade_source
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Add enhanced naming attribute
        self.use_enhanced_naming = use_enhanced_naming
          # Initialize enhanced naming if requested
        if use_enhanced_naming:
            from .enhanced_naming import EnhancedNamingScheme, ChartMetadataEnhancer
            self.naming_scheme = EnhancedNamingScheme(use_enhanced_naming=True)
            self.metadata_enhancer = ChartMetadataEnhancer(self.naming_scheme)
        else:
            self.naming_scheme = None
            self.metadata_enhancer = None
            
        # Initialize chart styling
        self.chart_styler = ChartStyler()
        self.chart_styler.apply_global_style()
        
    def _ensure_directory(self, directory: Path) -> None:
        """Ensure a directory exists, creating it if necessary."""
        directory.mkdir(parents=True, exist_ok=True)
        
    def _safe_save_plot(self, fig, filepath: Path, dpi: int = 300) -> Optional[Path]:
        """
        Safely save a plot with error handling.
        
        Args:
            fig: Matplotlib figure object
            filepath: Path to save the plot
            dpi: Resolution for saved image
            
        Returns:
            Path to saved file if successful, None otherwise
        """
        try:
            plt.figure(fig.number)  # Ensure we're working with the right figure
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            self.logger.debug(f"Saved visualization: {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error saving plot to {filepath}: {e}")
            plt.close(fig)
            return None
            
    def _create_no_data_plot(self, title: str, message: str = "No data available") -> plt.Figure:
        """
        Create a standardized "no data" plot.
        
        Args:
            title: Title for the plot
            message: Message to display
            
        Returns:
            Matplotlib figure with no data message
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, message, ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        ax.axis('off')
        return fig
        
    def _log_data_availability(self, data_dict: Dict[str, Any], context: str) -> None:
        """
        Log the availability of data sources for debugging.
        
        Args:
            data_dict: Dictionary containing data sources
            context: Context string for logging (e.g., ticker name)
        """
        available_sources = []
        for key, value in data_dict.items():
            if value is not None and not (isinstance(value, pd.DataFrame) and value.empty):
                available_sources.append(key)
        
        self.logger.debug(f"{context} - Available data sources: {available_sources}")


class VisualizationOrchestrator:
    """
    Main orchestrator that maintains the same API as the original monolith.
    
    This class acts as a facade, delegating to the appropriate specialized
    visualizers while preserving backward compatibility.
    """
    
    def __init__(self, output_dir: Optional[Path] = None, trade_source: str = "auto",
                 use_enhanced_naming: bool = False):
        """
        Initialize the visualization orchestrator.
        
        Args:
            output_dir: Directory to save visualizations
            trade_source: Trade data source preference
            use_enhanced_naming: Whether to use enhanced trader-focused naming scheme
        """
        self.output_dir = output_dir or Path("visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trade_source = trade_source
        self.use_enhanced_naming = use_enhanced_naming
        self.logger = logging.getLogger("VisualizationOrchestrator")
        
        # Create organized subdirectories (preserve original structure)
        self.portfolio_dir = self.output_dir / "portfolio"
        self.individual_dir = self.output_dir / "individual"
        self.portfolio_dir.mkdir(parents=True, exist_ok=True)
        self.individual_dir.mkdir(parents=True, exist_ok=True)
          # Initialize sub-components (will be imported when created)
        self._portfolio_visualizer = None
        self._individual_visualizer = None
    
    def _get_portfolio_visualizer(self):
        """Lazy load portfolio visualizer to avoid circular imports."""
        if self._portfolio_visualizer is None:
            from ..portfolio_visualizations.portfolio_visualizer import PortfolioLevelVisualizer
            self._portfolio_visualizer = PortfolioLevelVisualizer(
                output_dir=self.portfolio_dir, 
                trade_source=self.trade_source
            )
        return self._portfolio_visualizer
    
    def _get_individual_visualizer(self):
        """Lazy load individual visualizer to avoid circular imports."""
        if self._individual_visualizer is None:
            from ..individual_visualizations.individual_visualizer import IndividualTickerVisualizer
            self._individual_visualizer = IndividualTickerVisualizer(
                output_dir=self.individual_dir,
                trade_source=self.trade_source,
                use_enhanced_naming=self.use_enhanced_naming
            )
        return self._individual_visualizer
    
    def create_portfolio_dashboard(self, strategy_run_dir: Path, date_range: str, 
                                   tickers: List[str]) -> Dict[str, Path]:
        """
        Create comprehensive portfolio dashboard.
        
        This method preserves the exact same API as the original monolith.
        
        Args:
            strategy_run_dir: Path to strategy run directory
            date_range: Date range string
            tickers: List of tickers in portfolio
            
        Returns:
            Dictionary of created visualization files
        """
        return self._get_portfolio_visualizer().create_portfolio_dashboard(
            strategy_run_dir, date_range, tickers
        )
        
    def create_individual_ticker_dashboard(self, strategy_run_dir: Path, ticker: str,
                                           date_range: str) -> Dict[str, Path]:
        """
        Create comprehensive individual ticker visualizations.
        
        This method preserves the exact same API as the original monolith.
        
        Args:
            strategy_run_dir: Path to strategy run directory
            ticker: Ticker symbol
            date_range: Date range string
            
        Returns:
            Dictionary of created visualization files
        """
        return self._get_individual_visualizer().create_individual_ticker_dashboard(
            strategy_run_dir, ticker, date_range
        )
