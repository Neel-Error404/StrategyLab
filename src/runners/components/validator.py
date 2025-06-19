# src/runners/components/validator.py
"""
Data Validator Component

This module provides comprehensive data validation capabilities for the backtester,
including data quality checks, format validation, and anomaly detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging


class DataValidator:
    """
    Comprehensive data validation component for backtesting.
    
    This class provides extensive validation capabilities including:
    - Required column checks
    - Data quality validation
    - Price consistency verification
    - Anomaly detection
    - Missing data handling
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the data validator.
        
        Args:
            logger: Optional logger instance for validation messages
        """
        self.logger = logger or logging.getLogger(__name__)
        
    def validate_market_data(self, data: pd.DataFrame, ticker: str = "") -> Dict[str, Any]:
        """
        Comprehensive validation of market data.
        
        Args:
            data: DataFrame containing market data
            ticker: Ticker symbol for context in error messages
            
        Returns:
            Dictionary of validation results
        """
        if data is None or data.empty:
            return {
                "is_valid": False,
                "issues": ["Empty or missing market data"],
                "warnings": []
            }
        
        issues = []
        warnings = []
        
        # Run validation checks
        self._check_required_columns(data, issues)
        self._check_nan_values(data, issues)
        self._check_duplicate_dates(data, issues)
        self._check_price_consistency(data, issues)
        self._check_negative_values(data, issues)
        self._check_price_gaps(data, warnings)
        self._check_volume_outliers(data, warnings)
        self._check_stale_prices(data, warnings)
        
        is_valid = len(issues) == 0
        return {
            "is_valid": is_valid,
            "issues": issues,
            "warnings": warnings
        }
    
    def _check_required_columns(self, data: pd.DataFrame, issues: List[str]) -> None:
        """Check for required columns in data."""
        # Accept either 'date' or 'timestamp' for the date column
        date_columns = ['date', 'timestamp']
        has_date_column = any(col in data.columns for col in date_columns)
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not has_date_column:
            required_columns = ['date'] + required_columns
            
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {', '.join(missing_columns)}")
    
    def _check_nan_values(self, data: pd.DataFrame, issues: List[str]) -> None:
        """Check for NaN values in data."""
        nan_counts = data.isna().sum()
        nan_columns = [col for col in data.columns if nan_counts[col] > 0]
        if nan_columns:
            nan_details = ', '.join([f"{col}: {nan_counts[col]}" for col in nan_columns])
            issues.append(f"NaN values found in columns: {nan_details}")
    
    def _check_duplicate_dates(self, data: pd.DataFrame, issues: List[str]) -> None:
        """Check for duplicate dates in data."""
        date_col = None
        if 'date' in data.columns:
            date_col = 'date'
        elif 'timestamp' in data.columns:
            date_col = 'timestamp'
            
        if date_col:
            duplicate_dates = data[date_col].duplicated().sum()
            if duplicate_dates > 0:
                issues.append(f"Found {duplicate_dates} duplicate dates")
    
    def _check_price_consistency(self, data: pd.DataFrame, issues: List[str]) -> None:
        """Check for price consistency (high/low/open/close)."""
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            # High should be >= low
            high_low_issues = (data['high'] < data['low']).sum()
            if high_low_issues > 0:
                issues.append(f"Found {high_low_issues} rows where high < low")
            
            # High should be >= open and close
            high_open_issues = (data['high'] < data['open']).sum()
            high_close_issues = (data['high'] < data['close']).sum()
            if high_open_issues > 0 or high_close_issues > 0:
                issues.append(f"High price inconsistency: {high_open_issues} rows where high < open, "
                            f"{high_close_issues} rows where high < close")
            
            # Low should be <= open and close
            low_open_issues = (data['low'] > data['open']).sum()
            low_close_issues = (data['low'] > data['close']).sum()
            if low_open_issues > 0 or low_close_issues > 0:
                issues.append(f"Low price inconsistency: {low_open_issues} rows where low > open, "
                            f"{low_close_issues} rows where low > close")
    
    def _check_negative_values(self, data: pd.DataFrame, issues: List[str]) -> None:
        """Check for negative values in price/volume columns."""
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in data.columns:
                negative_count = (data[col] < 0).sum()
                if negative_count > 0:
                    issues.append(f"Found {negative_count} negative values in {col}")
    
    def _check_price_gaps(self, data: pd.DataFrame, warnings: List[str]) -> None:
        """Check for unusual price gaps (warnings only)."""
        if 'close' in data.columns and len(data) > 1:
            # Calculate price changes
            price_changes = data['close'].pct_change().abs()
            
            # Flag gaps > 10% as potential issues
            large_gaps = price_changes > 0.10
            gap_count = large_gaps.sum()
            
            if gap_count > 0:
                max_gap = price_changes.max() * 100
                warnings.append(f"Found {gap_count} large price gaps (>10%), max gap: {max_gap:.1f}%")
    
    def _check_volume_outliers(self, data: pd.DataFrame, warnings: List[str]) -> None:
        """Check for volume outliers (warnings only)."""
        if 'volume' in data.columns and len(data) > 10:
            volume_median = data['volume'].median()
            volume_outliers = data['volume'] > (volume_median * 10)
            outlier_count = volume_outliers.sum()
            
            if outlier_count > 0:
                warnings.append(f"Found {outlier_count} volume outliers (>10x median)")
    
    def _check_stale_prices(self, data: pd.DataFrame, warnings: List[str]) -> None:
        """Check for stale/unchanged prices (warnings only)."""
        if 'close' in data.columns and len(data) > 5:
            # Check for consecutive identical prices
            stale_prices = data['close'].eq(data['close'].shift()).sum()
            
            if stale_prices > len(data) * 0.1:  # More than 10% stale
                stale_percentage = (stale_prices / len(data)) * 100
                warnings.append(f"High percentage of stale prices: {stale_percentage:.1f}%")
    
    def validate_trades_data(self, data: pd.DataFrame, ticker: str = "") -> Dict[str, Any]:
        """
        Validate trade execution data.
        
        Args:
            data: DataFrame containing trade data
            ticker: Ticker symbol for context
            
        Returns:
            Dictionary of validation results
        """
        if data is None or data.empty:
            return {
                "is_valid": True,  # Empty trades data is valid (no trades executed)
                "issues": [],
                "warnings": ["No trades executed"]
            }
        
        issues = []
        warnings = []
        
        # Check required trade columns
        required_trade_columns = ['entry_time', 'exit_time', 'entry_price', 'exit_price', 'quantity']
        missing_columns = [col for col in required_trade_columns if col not in data.columns]
        if missing_columns:
            issues.append(f"Missing required trade columns: {', '.join(missing_columns)}")
        
        # Check for negative quantities or prices
        if 'quantity' in data.columns:
            negative_qty = (data['quantity'] <= 0).sum()
            if negative_qty > 0:
                issues.append(f"Found {negative_qty} trades with non-positive quantity")
        
        for price_col in ['entry_price', 'exit_price']:
            if price_col in data.columns:
                negative_prices = (data[price_col] <= 0).sum()
                if negative_prices > 0:
                    issues.append(f"Found {negative_prices} trades with non-positive {price_col}")
        
        # Check for realistic trade timing
        if 'entry_time' in data.columns and 'exit_time' in data.columns:
            try:
                entry_times = pd.to_datetime(data['entry_time'])
                exit_times = pd.to_datetime(data['exit_time'])
                invalid_timing = (exit_times <= entry_times).sum()
                if invalid_timing > 0:
                    issues.append(f"Found {invalid_timing} trades with exit_time <= entry_time")
            except Exception as e:
                issues.append(f"Error parsing trade timestamps: {str(e)}")
        
        is_valid = len(issues) == 0
        return {
            "is_valid": is_valid,
            "issues": issues,
            "warnings": warnings
        }
    
    def validate_portfolio_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate portfolio performance data.
        
        Args:
            data: DataFrame containing portfolio data
            
        Returns:
            Dictionary of validation results
        """
        if data is None or data.empty:
            return {
                "is_valid": False,
                "issues": ["Empty portfolio data"],
                "warnings": []
            }
        
        issues = []
        warnings = []
        
        # Check for required portfolio columns
        required_portfolio_columns = ['timestamp', 'portfolio_value', 'cash', 'total_pnl']
        missing_columns = [col for col in required_portfolio_columns if col not in data.columns]
        if missing_columns:
            issues.append(f"Missing required portfolio columns: {', '.join(missing_columns)}")
        
        # Check for negative portfolio values
        if 'portfolio_value' in data.columns:
            negative_value = (data['portfolio_value'] < 0).sum()
            if negative_value > 0:
                warnings.append(f"Found {negative_value} periods with negative portfolio value")
        
        # Check for excessive drawdowns
        if 'portfolio_value' in data.columns and len(data) > 1:
            cummax = data['portfolio_value'].cummax()
            drawdown = (data['portfolio_value'] - cummax) / cummax
            max_drawdown = abs(drawdown.min())
            
            if max_drawdown > 0.5:  # 50% drawdown
                warnings.append(f"Excessive maximum drawdown: {max_drawdown:.1%}")
        
        is_valid = len(issues) == 0
        return {
            "is_valid": is_valid,
            "issues": issues,
            "warnings": warnings
        }
    
    def validate_data(self, dates: List[str], tickers: List[str]) -> bool:
        """
        High-level data validation method compatible with the original monolithic runner.
        
        Args:
            dates: List of dates or date ranges
            tickers: List of ticker symbols
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        from src.core.etl.loader import load_base_data
        
        self.logger.info(f"Validating data for {len(dates)} dates and {len(tickers)} tickers")
        
        validation_results = {}
        total_checks = 0
        failed_checks = 0
        
        for date in dates:
            for ticker in tickers:
                total_checks += 1
                try:
                    # Load data for validation
                    data = load_base_data(date, ticker)
                    
                    if data is None or data.empty:
                        self.logger.warning(f"No data found for {ticker} on {date}")
                        validation_results[f"{ticker}_{date}"] = "no_data"
                        failed_checks += 1
                        continue
                    
                    # Validate market data using the comprehensive validator
                    result = self.validate_market_data(data, ticker)
                    
                    if result["is_valid"]:
                        validation_results[f"{ticker}_{date}"] = "passed"
                        self.logger.debug(f"Validation passed for {ticker} on {date}")
                    else:
                        validation_results[f"{ticker}_{date}"] = "failed"
                        failed_checks += 1
                        self.logger.warning(f"Validation failed for {ticker} on {date}: {result.get('errors', [])}")
                        
                except Exception as e:
                    validation_results[f"{ticker}_{date}"] = "error"
                    failed_checks += 1
                    self.logger.error(f"Error validating {ticker} on {date}: {e}")
        
        # Calculate success rate
        success_rate = (total_checks - failed_checks) / total_checks if total_checks > 0 else 0
        
        self.logger.info(f"Data validation completed: {total_checks - failed_checks}/{total_checks} passed ({success_rate:.2%})")
        
        # Return True if all validations passed, or if we have reasonable success rate
        if failed_checks == 0:
            return True
        elif success_rate >= 0.8:  # 80% success rate threshold
            self.logger.warning(f"Some validations failed but success rate ({success_rate:.2%}) is acceptable")
            return True
        else:
            self.logger.error(f"Too many validation failures (success rate: {success_rate:.2%})")
            return False

    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary of data characteristics.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary containing data summary
        """
        if data is None or data.empty:
            return {"error": "No data provided"}
        
        summary = {
            "row_count": len(data),
            "column_count": len(data.columns),
            "columns": list(data.columns),
            "date_range": {},
            "data_types": data.dtypes.to_dict(),
            "memory_usage": data.memory_usage(deep=True).sum()
        }
        
        # Get date range if date/timestamp column exists
        date_col = None
        if 'date' in data.columns:
            date_col = 'date'
        elif 'timestamp' in data.columns:
            date_col = 'timestamp'
            
        if date_col:
            try:
                dates = pd.to_datetime(data[date_col])
                summary["date_range"] = {
                    "start": dates.min().isoformat(),
                    "end": dates.max().isoformat(),
                    "duration_days": (dates.max() - dates.min()).days
                }
            except Exception as e:
                summary["date_range"] = {"error": f"Could not parse dates: {str(e)}"}
        
        # Get numeric column statistics
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            summary["numeric_stats"] = data[numeric_columns].describe().to_dict()
        
        return summary
