"""
Data loading utilities for visualization system.

This module provides common data loading and processing functionality
shared across individual ticker and portfolio visualizations.
"""

import logging
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


class DataLoader:
    """
    Handles loading and processing of data for visualizations.
    
    Provides standardized methods for loading base data, trade data,
    analytics reports, and setting active trades based on preferences.
    """
    
    def __init__(self, trade_source: str = "auto"):
        """
        Initialize the data loader.
        
        Args:
            trade_source: Trade data source preference:
                         - "strategy_trades": Use raw strategy output
                         - "risk_approved_trades": Use post-risk-management trades
                         - "auto": Try risk_approved_trades first, fallback to strategy_trades
        """
        self.trade_source = trade_source
        self.logger = logging.getLogger("DataLoader")
        
    def load_individual_ticker_data(self, strategy_run_dir: Path, ticker: str, 
                                    date_range: str) -> Dict[str, Any]:
        """
        Load all data needed for individual ticker visualization.
        
        Args:
            strategy_run_dir: Path to strategy run directory
            ticker: Ticker symbol
            date_range: Date range string
            
        Returns:
            Dictionary containing all ticker data sources
        """
        ticker_data = {
            'ticker': ticker,
            'date_range': date_range,
            'base_data': None,
            'strategy_trades': None,
            'risk_approved_trades': None,
            'active_trades': None,
            'analytics': None
        }
        
        try:
            # Load base data
            base_file = strategy_run_dir / 'data' / 'base_data' / f"{ticker}_Base_{date_range}.csv"
            if base_file.exists():
                ticker_data['base_data'] = pd.read_csv(base_file)
                self.logger.debug(f"Loaded base data for {ticker}: {len(ticker_data['base_data'])} rows")
            
            # Load strategy trades
            strategy_trades_file = strategy_run_dir / 'data' / 'strategy_trades' / f"{ticker}_StrategyTrades_{date_range}.csv"
            if strategy_trades_file.exists():
                ticker_data['strategy_trades'] = pd.read_csv(strategy_trades_file)
                self.logger.debug(f"Loaded strategy trades for {ticker}: {len(ticker_data['strategy_trades'])} trades")
            
            # Load risk approved trades
            risk_trades_file = strategy_run_dir / 'data' / 'risk_approved_trades' / f"{ticker}_RiskApprovedTrades_{date_range}.csv"
            if risk_trades_file.exists():
                ticker_data['risk_approved_trades'] = pd.read_csv(risk_trades_file)
                self.logger.debug(f"Loaded risk approved trades for {ticker}: {len(ticker_data['risk_approved_trades'])} trades")
            
            # Load analytics
            analytics_file = strategy_run_dir / 'analysis_reports' / 'individual' / f"{ticker}_Analysis_{date_range}.json"
            if analytics_file.exists():
                with open(analytics_file, 'r') as f:
                    ticker_data['analytics'] = json.load(f)
                self.logger.debug(f"Loaded analytics for {ticker}")
            
            # Set active trades based on trade source preference
            self._set_active_trades(ticker_data)
            
        except Exception as e:
            self.logger.error(f"Error loading data for {ticker}: {e}")
            
        return ticker_data
        
    def load_portfolio_data(self, strategy_run_dir: Path, date_range: str, 
                           tickers: List[str]) -> Dict[str, Any]:
        """
        Load portfolio-level data for all tickers.
        
        Args:
            strategy_run_dir: Path to strategy run directory
            date_range: Date range string
            tickers: List of ticker symbols
            
        Returns:
            Dictionary containing portfolio data for all tickers
        """
        portfolio_data = {
            'tickers': tickers,
            'date_range': date_range,
            'ticker_data': {},
            'portfolio_analytics': None
        }
        
        try:
            # Load data for each ticker
            for ticker in tickers:
                portfolio_data['ticker_data'][ticker] = self.load_individual_ticker_data(
                    strategy_run_dir, ticker, date_range
                )
            
            # Load portfolio-level analytics if available
            portfolio_analytics_file = strategy_run_dir / 'analysis_reports' / f"Portfolio_Analysis_{date_range}.json"
            if portfolio_analytics_file.exists():
                with open(portfolio_analytics_file, 'r') as f:
                    portfolio_data['portfolio_analytics'] = json.load(f)
                self.logger.debug("Loaded portfolio analytics")
            
            self.logger.info(f"Loaded portfolio data for {len(tickers)} tickers")
            
        except Exception as e:
            self.logger.error(f"Error loading portfolio data: {e}")
            
        return portfolio_data
        
    def _set_active_trades(self, ticker_data: Dict[str, Any]) -> None:
        """
        Set active trades for ticker based on trade_source preference.
        
        Args:
            ticker_data: Dictionary containing ticker data
        """
        ticker = ticker_data['ticker']
        
        if self.trade_source == "strategy_trades":
            ticker_data['active_trades'] = ticker_data['strategy_trades']
        elif self.trade_source == "risk_approved_trades":
            ticker_data['active_trades'] = ticker_data['risk_approved_trades']
        elif self.trade_source == "auto":
            # Try risk approved first, fallback to strategy
            if (ticker_data['risk_approved_trades'] is not None and 
                not ticker_data['risk_approved_trades'].empty):
                ticker_data['active_trades'] = ticker_data['risk_approved_trades']
                self.logger.debug(f"Using risk_approved_trades for {ticker}")
            else:
                ticker_data['active_trades'] = ticker_data['strategy_trades']
                self.logger.debug(f"Falling back to strategy_trades for {ticker}")
        
        # Log the final choice
        if ticker_data['active_trades'] is not None and not ticker_data['active_trades'].empty:
            trade_count = len(ticker_data['active_trades'])
            self.logger.debug(f"Active trades for {ticker}: {trade_count} trades")
        else:
            self.logger.warning(f"No active trades found for {ticker}")
            
    def limit_portfolio_tickers(self, strategy_run_dir: Path, date_range: str, 
                               tickers: List[str], max_tickers: int = 12) -> List[str]:
        """
        Limit portfolio tickers to a maximum number based on selection criteria.
        
        This method preserves the original logic from the monolith for selecting
        which tickers to include in portfolio visualizations.
        
        Args:
            strategy_run_dir: Path to strategy run directory
            date_range: Date range string
            tickers: Original list of tickers
            max_tickers: Maximum number of tickers to include
            
        Returns:
            Limited list of tickers
        """
        if len(tickers) <= max_tickers:
            return tickers
        
        # Use the same selection logic as original - prioritize by trade count
        ticker_scores = []
        
        for ticker in tickers:
            try:
                # Check strategy trades file
                strategy_trades_file = strategy_run_dir / 'data' / 'strategy_trades' / f"{ticker}_StrategyTrades_{date_range}.csv"
                trade_count = 0
                
                if strategy_trades_file.exists():
                    trades_df = pd.read_csv(strategy_trades_file)
                    trade_count = len(trades_df) if not trades_df.empty else 0
                
                ticker_scores.append((ticker, trade_count))
                
            except Exception as e:
                self.logger.warning(f"Error evaluating {ticker} for portfolio inclusion: {e}")
                ticker_scores.append((ticker, 0))
        
        # Sort by trade count (descending) and take top max_tickers
        ticker_scores.sort(key=lambda x: x[1], reverse=True)
        limited_tickers = [ticker for ticker, _ in ticker_scores[:max_tickers]]
        
        self.logger.info(f"Limited portfolio from {len(tickers)} to {len(limited_tickers)} tickers based on trade activity")
        
        return limited_tickers
