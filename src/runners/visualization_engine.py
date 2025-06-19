#!/usr/bin/env python3
"""
Visualization Engine Module for Unified Backtester

Handles all portfolio visualization and dashboard generation.
Uses the comprehensive PortfolioVisualizer with trade source fallback logic.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import comprehensive visualizer with trade source logic instead of simplified one
from src.core.analysis.portfolio_visualization import PortfolioVisualizer


class VisualizationEngine:
    """
    Manages portfolio visualization and dashboard generation.
    Uses the comprehensive PortfolioVisualizer with trade source fallback logic.
    """
    
    def __init__(self, config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Get trade source from config or default to 'auto' for fallback logic
        trade_source = getattr(config.output, 'visualization_trade_source', 'auto')
          # Initialize the comprehensive visualizer with trade source configuration
        output_dir = Path(config.base_dir) / config.output.output_dir / config.run_id / "visualizations"
        self.portfolio_visualizer = PortfolioVisualizer(
            output_dir=output_dir,
            trade_source=trade_source
        )
    def generate_visualizations(self, results: Dict[str, Any]) -> bool:
        """
        Generate comprehensive portfolio visualizations using the comprehensive visualizer.
        
        Args:
            results: Structured dictionary of backtest results
            
        Returns:
            bool: True if visualizations were generated successfully
        """
        self.logger.info("📊 Generating portfolio visualizations")
        
        if not results:
            self.logger.warning("No backtest results available for visualization")
            return False
        
        try:
            # Extract necessary information from results
            strategy_run_info = self._extract_strategy_run_info(results)
            if not strategy_run_info:
                self.logger.warning("Could not extract strategy run information from results")
                return False
                
            strategy_run_dir = strategy_run_info['strategy_run_dir']
            date_range = strategy_run_info['date_range']
            tickers = strategy_run_info['tickers']
            
            # CRITICAL FIX: In visualize mode, we need to create the three-file system first
            # so the PortfolioVisualizer can find trade data
            self._ensure_three_file_system(results, strategy_run_dir, date_range, tickers)
            
            # Generate portfolio dashboard using comprehensive visualizer
            self.logger.info(f"Generating portfolio dashboard for {len(tickers)} tickers: {tickers}")
            portfolio_dashboards = self.portfolio_visualizer.create_portfolio_dashboard(
                strategy_run_dir, date_range, tickers
            )
            
            # Generate individual ticker dashboards
            individual_visualizations = {}
            for ticker in tickers:
                self.logger.info(f"Generating individual dashboard for {ticker}")
                ticker_visualizations = self.portfolio_visualizer.create_individual_ticker_dashboard(
                    strategy_run_dir, ticker, date_range
                )
                if ticker_visualizations:
                    individual_visualizations[ticker] = ticker_visualizations
            
            # Log success
            total_individual_count = sum(len(viz) for viz in individual_visualizations.values())
            total_visualizations = len(portfolio_dashboards) + total_individual_count
            self.logger.info(f"✅ Generated {len(portfolio_dashboards)} portfolio + {total_individual_count} individual visualizations = {total_visualizations} total")
            self._log_visualization_summary(self.portfolio_visualizer.output_dir)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
            return False
    
    def _extract_strategy_run_info(self, results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract strategy run information needed for visualization from results.
        
        Args:
            results: Structured backtest results dictionary
            
        Returns:
            Dictionary with strategy_run_dir, date_range, and tickers, or None if extraction fails
        """
        try:
            # Handle different result structures
            if 'output_dir' in results:
                # Direct output directory reference
                strategy_run_dir = Path(results['output_dir'])
            elif 'runs' in results:
                # Multiple runs - use the first one's output directory
                first_run = next(iter(results['runs'].values()))
                if isinstance(first_run, dict) and 'output_dir' in first_run:
                    strategy_run_dir = Path(first_run['output_dir'])
                else:
                    # Fallback: construct from config
                    strategy_run_dir = Path(self.config.base_dir) / self.config.output.output_dir / self.config.run_id
            else:
                # Fallback: construct from config
                strategy_run_dir = Path(self.config.base_dir) / self.config.output.output_dir / self.config.run_id
            
            # Extract date range - try multiple sources
            date_range = None
            
            # Try to get from runs data
            if 'runs' in results:
                for run_data in results['runs'].values():
                    if isinstance(run_data, dict):
                        for ticker_data in run_data.values():
                            if isinstance(ticker_data, dict) and 'date_range' in ticker_data:
                                date_range = ticker_data['date_range']
                                break
                    if date_range:
                        break            # Try date_ranges from config if available (priority over single dates)
            if not date_range and hasattr(self.config, 'date_ranges') and self.config.date_ranges:
                if isinstance(self.config.date_ranges, list) and len(self.config.date_ranges) > 0:
                    date_range = self.config.date_ranges[0]
                elif isinstance(self.config.date_ranges, str):
                    date_range = self.config.date_ranges
                    
            # Try config dates - handle both single dates and date ranges
            if not date_range and hasattr(self.config, 'dates') and self.config.dates:
                if isinstance(self.config.dates, list) and len(self.config.dates) > 0:
                    # If single date, use it; if multiple dates, create range
                    if len(self.config.dates) == 1:
                        date_range = self.config.dates[0]
                    else:
                        # Create range from first to last date
                        start_date = min(self.config.dates)
                        end_date = max(self.config.dates)
                        date_range = f"{start_date}_to_{end_date}"
                elif isinstance(self.config.dates, str):
                    date_range = self.config.dates# Try to extract from output directory path
            if not date_range and strategy_run_dir:
                # Use os.sep for proper path splitting on all platforms
                import os
                dir_parts = str(strategy_run_dir).split(os.sep)
                for part in dir_parts:
                    if '_to_' in part and len(part.split('_to_')) == 2:
                        try:
                            start_date, end_date = part.split('_to_')
                            # Validate date format
                            if len(start_date) == 10 and len(end_date) == 10:
                                date_range = part
                                break
                        except (ValueError, IndexError):
                            continue
              # Final fallback: extract from current date or use minimal range
            if not date_range:
                from datetime import datetime
                current_date = datetime.now().strftime('%Y-%m-%d')
                self.logger.warning(f"No date range found, using current date: {current_date}")
                date_range = f"{current_date}_to_{current_date}"
              # Extract tickers - check actual tickers directory first
            tickers = []
            
            # Primary: Check tickers directory in output
            tickers_dir = strategy_run_dir / "tickers"
            if tickers_dir.exists():
                ticker_dirs = [d.name for d in tickers_dir.iterdir() if d.is_dir()]
                if ticker_dirs:
                    tickers = ticker_dirs
              # Secondary: Extract from results structure - handle both 'runs' and direct strategy structure
            if not tickers:
                # Handle 'runs' structure (if it exists)
                if 'runs' in results:
                    for run_data in results['runs'].values():
                        if isinstance(run_data, dict):
                            for ticker_data in run_data.values():
                                if isinstance(ticker_data, dict) and 'ticker' in ticker_data:
                                    ticker = ticker_data['ticker']
                                    if ticker not in tickers:
                                        tickers.append(ticker)
                
                # Handle direct strategy structure: strategy -> date_range -> ticker
                if not tickers:
                    for strategy_name, strategy_data in results.items():
                        if isinstance(strategy_data, dict):
                            for date_range_key, date_data in strategy_data.items():
                                if isinstance(date_data, dict):
                                    for ticker_name, ticker_data in date_data.items():
                                        if isinstance(ticker_data, dict) and ticker_name not in tickers:
                                            tickers.append(ticker_name)
            
            # Tertiary: Fallback to config tickers
            if not tickers and hasattr(self.config, 'tickers'):
                tickers = self.config.tickers
              # Final fallback: warn and use empty list if no tickers found anywhere
            if not tickers:
                self.logger.warning("No tickers found in directory, config, or results. Using empty ticker list.")
                tickers = []
            
            # Look for actual strategy directory (may have strategy name in path)
            potential_dirs = []
            if strategy_run_dir.exists():
                potential_dirs.append(strategy_run_dir)
            
            # Look for subdirectories with strategy names
            for strategy_dir in strategy_run_dir.parent.glob("**/"):
                if any(strategy in strategy_dir.name for strategy in ['mse', 'strategy']):
                    potential_dirs.append(strategy_dir)
            
            # Use the first existing directory or the original
            actual_strategy_dir = strategy_run_dir
            for potential_dir in potential_dirs:
                if potential_dir.exists() and (potential_dir / 'data').exists():
                    actual_strategy_dir = potential_dir
                    break
            
            self.logger.info(f"Extracted info: dir={actual_strategy_dir}, date_range={date_range}, tickers={tickers}")
            
            return {
                'strategy_run_dir': actual_strategy_dir,
                'date_range': date_range,
                'tickers': tickers            }
            
        except Exception as e:
            self.logger.error(f"Error extracting strategy run info: {e}")
            return None

    def _log_visualization_summary(self, output_dir: Path) -> None:
        """Log a summary of generated visualizations."""
        # Search recursively for visualization files
        viz_files = list(output_dir.rglob("*.png")) + list(output_dir.rglob("*.html"))
        
        self.logger.info("=" * 50)
        self.logger.info("📈 VISUALIZATION SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"Output Directory: {output_dir}")
        self.logger.info(f"Generated Files: {len(viz_files)}")
        
        for viz_file in viz_files:
            # Show relative path from output_dir for better readability
            relative_path = viz_file.relative_to(output_dir)
            self.logger.info(f"  - {relative_path}")
            
        self.logger.info("=" * 50)
    
    def _ensure_three_file_system(self, results: Dict[str, Any], strategy_run_dir: Path, 
                                  date_range: str, tickers: List[str]) -> None:
        """
        Create minimal three-file system data required by PortfolioVisualizer.
        
        Args:
            results: Backtest results from task executor
            strategy_run_dir: Strategy run directory path
            date_range: Date range string
            tickers: List of ticker symbols
        """
        try:
            from src.core.output.three_file_system import ThreeFileOutputSystem
            
            self.logger.info("Creating minimal three-file system for visualization...")
            three_file_system = ThreeFileOutputSystem(strategy_run_dir)
            
            # Process each ticker's data from results
            for ticker in tickers:
                # Extract ticker data from results structure
                ticker_data = self._extract_ticker_data_from_results(results, ticker, date_range)
                if not ticker_data:
                    self.logger.warning(f"No data found for {ticker} in results")
                    continue
                
                # Save base data if available
                base_data = ticker_data.get('base_data')
                if base_data is not None and len(base_data) > 0:
                    if isinstance(base_data, dict):
                        import pandas as pd
                        base_data = pd.DataFrame(base_data)
                    if not base_data.empty:
                        three_file_system.save_base_file(ticker, date_range, base_data)
                
                # Save strategy trades
                trades = ticker_data.get('trades', [])
                strategy_metadata = ticker_data.get('strategy_metadata', {})
                three_file_system.save_strategy_trades_file(ticker, date_range, trades, strategy_metadata)
                
                # Save risk-approved trades (for now, same as strategy trades)
                risk_analysis = ticker_data.get('risk_analysis', {})
                three_file_system.save_risk_approved_trades_file(ticker, date_range, trades, risk_analysis)
                
            self.logger.info(f"Created three-file system for {len(tickers)} tickers")
                
        except Exception as e:
            self.logger.error(f"Error creating three-file system: {e}")
    
    def _extract_ticker_data_from_results(self, results: Dict[str, Any], ticker: str, 
                                          date_range: str) -> Optional[Dict[str, Any]]:
        """
        Extract ticker-specific data from backtest results.
        
        Args:
            results: Backtest results structure
            ticker: Ticker symbol
            date_range: Date range string
            
        Returns:
            Ticker data dictionary or None if not found
        """
        try:
            # Handle direct strategy structure: strategy -> date_range -> ticker
            for strategy_name, strategy_data in results.items():
                if isinstance(strategy_data, dict) and date_range in strategy_data:
                    date_data = strategy_data[date_range]
                    if isinstance(date_data, dict) and ticker in date_data:
                        return date_data[ticker]
            
            # Handle 'runs' structure if it exists
            if 'runs' in results:
                for run_data in results['runs'].values():
                    if isinstance(run_data, dict) and ticker in run_data:
                        return run_data[ticker]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting ticker data for {ticker}: {e}")
            return None
