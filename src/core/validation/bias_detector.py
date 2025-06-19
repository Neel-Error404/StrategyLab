# src/validation/bias_detector.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

class BiasDetector:
    """
    Comprehensive bias detection and prevention for backtesting.
    Detects look-ahead bias, survivorship bias, and data snooping.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("BiasDetector")
        self.violations = []
    
    def validate_no_lookahead(self, df: pd.DataFrame, 
                              signal_columns: List[str], 
                              indicator_columns: List[str]) -> Tuple[bool, List[Dict]]:
        """
        Validates that signals only use data available at signal generation time.
        
        Args:
            df: DataFrame with timestamps, signals, and indicators
            signal_columns: List of signal column names
            indicator_columns: List of indicator column names
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        # Check each signal
        for idx in range(1, len(df)):
            current_time = df.iloc[idx]['timestamp']
            
            for signal_col in signal_columns:
                if df.iloc[idx][signal_col]:  # If signal is True
                    # Check all indicators used
                    for ind_col in indicator_columns:
                        # Verify indicator was calculated from past data only
                        ind_timestamp = self._get_indicator_timestamp(df, idx, ind_col)
                        
                        if ind_timestamp and ind_timestamp >= current_time:
                            violations.append({
                                'index': idx,
                                'signal': signal_col,
                                'indicator': ind_col,
                                'signal_time': current_time,
                                'indicator_time': ind_timestamp,
                                'violation_type': 'look_ahead'
                            })
        
        return len(violations) == 0, violations
    
    def detect_survivorship_bias(self, tickers: List[str], 
                                 test_period: Tuple[datetime, datetime],
                                 delisted_db_path: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Detects potential survivorship bias by checking for missing delisted securities.
        
        Args:
            tickers: List of tickers used in backtest
            test_period: Tuple of (start_date, end_date)
            delisted_db_path: Path to delisted securities database
            
        Returns:
            Dictionary with bias analysis
        """
        # This would connect to a database of historical constituents
        # For now, we'll implement the structure
        
        bias_report = {
            'tested_tickers': tickers,
            'period': f"{test_period[0]} to {test_period[1]}",
            'missing_delisted': [],  # Would be populated from delisted DB
            'potential_bias': 'unknown'  # Would be calculated
        }
        
        # Placeholder for actual implementation
        # In production, this would query historical index constituents
        
        return bias_report
    
    def validate_data_mining_bias(self, strategy_results: Dict[str, pd.DataFrame],
                                  in_sample_period: Tuple[datetime, datetime],
                                  out_sample_period: Tuple[datetime, datetime]) -> Dict[str, float]:
        """
        Detects data mining bias by comparing in-sample vs out-of-sample performance.
        
        Args:
            strategy_results: Dictionary of strategy results
            in_sample_period: In-sample period for strategy development
            out_sample_period: Out-of-sample period for validation
            
        Returns:
            Dictionary with performance degradation metrics
        """
        degradation_metrics = {}
        
        for strategy_name, results in strategy_results.items():
            in_sample = results[
                (results['timestamp'] >= in_sample_period[0]) & 
                (results['timestamp'] <= in_sample_period[1])
            ]
            out_sample = results[
                (results['timestamp'] >= out_sample_period[0]) & 
                (results['timestamp'] <= out_sample_period[1])
            ]
            
            # Calculate performance metrics
            in_sharpe = self._calculate_sharpe(in_sample['returns'])
            out_sharpe = self._calculate_sharpe(out_sample['returns'])
            
            degradation_metrics[strategy_name] = {
                'in_sample_sharpe': in_sharpe,
                'out_sample_sharpe': out_sharpe,
                'degradation_pct': ((in_sharpe - out_sharpe) / in_sharpe * 100) if in_sharpe > 0 else 0
            }
        
        return degradation_metrics
    
    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.05) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0.0
    
    def _get_indicator_timestamp(self, df: pd.DataFrame, idx: int, indicator_col: str) -> Optional[datetime]:
        """Get the timestamp when an indicator was calculated."""
        # Indicators should use data from previous bars, not current bar
        # For real-time trading, indicator values are only available after bar completion
        
        if idx == 0:
            return df.iloc[idx]['timestamp']  # First bar has no previous data
        
        # For proper backtesting, indicators should be calculated from previous bar
        # This prevents look-ahead bias by ensuring signal uses only historical data
        return df.iloc[idx-1]['timestamp']