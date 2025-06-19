"""
Analysis components for the backtester.

This module provides portfolio and trade analysis functionality.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import logging

# Constants for field names
PROFIT_PCT_FIELD = 'Profit (%)'
RETURNS_PCT_FIELD = 'returns_pct'
TRADE_DURATION_MIN_FIELD = 'Trade Duration (min)'
DURATION_DAYS_FIELD = 'duration_days'

class PortfolioAnalyzer:
    """
    Portfolio analysis component for backtesting results.
    
    Provides:
    - Portfolio performance metrics
    - Trade analysis statistics
    - Risk metrics calculation
    """
    
    def __init__(self, config):
        """
        Initialize the analyzer with configuration.
        
        Args:
            config: Configuration object with analysis settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze backtesting results and compute portfolio metrics.
        
        Args:
            results: Dictionary of backtesting results
            
        Returns:
            Dictionary of portfolio analysis metrics
        """
        self.logger.info("Running portfolio analysis")
        
        if not results:
            self.logger.warning("No results available for analysis")
            return {}
        
        # Collect all trades and metrics for portfolio analysis
        all_trades = []
        all_metrics = {}
        
        # Process results for each strategy, date, and ticker
        for strategy_name, strategy_results in results.items():
            all_metrics[strategy_name] = {}
            
            for date_range, date_results in strategy_results.items():
                strategy_trades = []
                trade_returns = []
                
                for ticker, ticker_result in date_results.items():
                    # Collect trades
                    if 'trades' in ticker_result and ticker_result['trades']:
                        for trade in ticker_result['trades']:
                            # Add metadata to each trade
                            trade_with_meta = trade.copy()
                            trade_with_meta['strategy'] = strategy_name
                            trade_with_meta['date_range'] = date_range
                            trade_with_meta['ticker'] = ticker
                            strategy_trades.append(trade_with_meta)                            # Extract returns for analysis - handle different field names
                            if PROFIT_PCT_FIELD in trade:
                                trade_returns.append(trade[PROFIT_PCT_FIELD])
                            elif RETURNS_PCT_FIELD in trade:
                                trade_returns.append(trade[RETURNS_PCT_FIELD])
                
                # Calculate metrics for this strategy and date range
                if strategy_trades:
                    metrics = self._calculate_performance_metrics(strategy_trades, trade_returns)
                    all_metrics[strategy_name][date_range] = metrics
                    all_trades.extend(strategy_trades)
        
        # Calculate overall portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(all_trades)
        
        return {
            'portfolio_metrics': portfolio_metrics,
            'strategy_metrics': all_metrics,
            'trades': all_trades
        }
        
    def _calculate_performance_metrics(self, trades: List[Dict], returns: List[float]) -> Dict[str, Any]:
        """
        Calculate performance metrics for a set of trades.
        
        Args:
            trades: List of trade dictionaries
            returns: List of trade returns (percentages)
            
        Returns:
            Dictionary of performance metrics
        """
        if not trades or not returns:
            return {}
            
        # Convert to numpy array for calculations
        returns_array = np.array(returns)
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = sum(1 for r in returns if r > 0)
        losing_trades = sum(1 for r in returns if r <= 0)
        
        # Return metrics
        total_return = np.sum(returns_array)
        avg_return = np.mean(returns_array) if total_trades > 0 else 0
        median_return = np.median(returns_array) if total_trades > 0 else 0
        
        # Risk metrics
        std_dev = np.std(returns_array) if total_trades > 1 else 0
        max_drawdown = self._calculate_max_drawdown(returns_array)
        
        # Ratio metrics
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        loss_rate = losing_trades / total_trades if total_trades > 0 else 0
        
        # Calculate Sharpe-like ratio (assuming risk-free rate = 0)
        risk_adjusted_return = avg_return / std_dev if std_dev > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'total_return': total_return,
            'avg_return': avg_return,
            'median_return': median_return,
            'std_dev': std_dev,
            'max_drawdown': max_drawdown,
            'risk_adjusted_return': risk_adjusted_return
        }
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown from a series of returns.
        
        Args:
            returns: Array of returns
            
        Returns:
            Maximum drawdown value
        """
        if len(returns) <= 1:
            return 0.0
            
        # Calculate cumulative returns
        cum_returns = (1 + returns / 100).cumprod()
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cum_returns)
        
        # Calculate drawdown
        drawdown = (cum_returns / running_max - 1) * 100
        
        # Get the maximum drawdown
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        return abs(max_drawdown)
        
    def _calculate_portfolio_metrics(self, all_trades: List[Dict]) -> Dict[str, Any]:
        """
        Calculate overall portfolio metrics across all strategies.
        
        Args:
            all_trades: List of all trade dictionaries
            
        Returns:
            Dictionary of portfolio-level metrics
        """
        if not all_trades:
            return {}        # Extract relevant data - handle different field names from extract_trades
        returns = []
        durations = []
        
        for trade in all_trades:
            # Try different field names for returns
            if PROFIT_PCT_FIELD in trade:
                returns.append(trade[PROFIT_PCT_FIELD])
            elif RETURNS_PCT_FIELD in trade:
                returns.append(trade[RETURNS_PCT_FIELD])
            
            # Try different field names for duration  
            if TRADE_DURATION_MIN_FIELD in trade:
                durations.append(trade[TRADE_DURATION_MIN_FIELD] / (60 * 24))  # Convert minutes to days
            elif DURATION_DAYS_FIELD in trade:
                durations.append(trade[DURATION_DAYS_FIELD])
        
        # Strategy diversity
        unique_strategies = len(set(trade.get('strategy') for trade in all_trades))
        unique_tickers = len(set(trade.get('ticker') for trade in all_trades))
        
        # Calculate metrics
        metrics = self._calculate_performance_metrics(all_trades, returns)
        
        # Add additional portfolio-specific metrics
        metrics.update({
            'total_strategies': unique_strategies,
            'total_tickers': unique_tickers,
            'avg_trade_duration': np.mean(durations) if durations else 0,
            'median_trade_duration': np.median(durations) if durations else 0
        })
        
        return metrics
        
    def generate_trade_reports(self, trades: List[Dict], output_dir: Path) -> Dict[str, Path]:
        """
        Generate detailed trade reports from trade data.
        
        Args:
            trades: List of trade dictionaries
            output_dir: Directory to save trade reports
            
        Returns:
            Dictionary mapping report type to file path
        """
        if not trades:
            self.logger.warning("No trades available for reporting")
            return {}
            
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(trades)
        
        # Generate trade summary report
        summary_path = output_dir / "trade_summary.csv"
        trades_df.to_csv(summary_path, index=False)
        
        # Generate strategy-specific reports
        strategy_reports = {}
        for strategy, group in trades_df.groupby('strategy'):
            strategy_path = output_dir / f"{strategy}_trades.csv"
            group.to_csv(strategy_path, index=False)
            strategy_reports[strategy] = strategy_path
        
        return {
            'summary': summary_path,
            'strategy_reports': strategy_reports
        }
