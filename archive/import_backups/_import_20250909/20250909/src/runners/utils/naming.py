"""
Naming utilities for the backtesting framework.

This module provides functions for generating deterministic names
for output directories, files, and other resources.
"""

import hashlib
from typing import List, Optional, Dict, Any
import re


def sanitize_name(name: str) -> str:
    """
    Sanitize a name for file system use.
    
    Args:
        name: The name to sanitize
        
    Returns:
        Sanitized name suitable for file system paths
    """
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[^\w\-\.]', '_', name)
    # Remove leading/trailing spaces and underscores
    sanitized = sanitized.strip(' _')
    return sanitized


def create_deterministic_name(strategies: List[str], 
                             dates: List[str],
                             tickers: Optional[List[str]] = None,
                             mode: str = "backtest") -> str:
    """
    Create a deterministic output name based on run parameters.
    
    Args:
        strategies: List of strategy names
        dates: List of date ranges
        tickers: Optional list of tickers
        mode: Run mode (backtest, optimize, etc.)
        
    Returns:
        Deterministic output name
    """
    # Sort inputs for deterministic ordering
    strategies_sorted = sorted([s.lower() for s in strategies])
    dates_sorted = sorted(dates)
    tickers_sorted = sorted([t.upper() for t in (tickers or [])])
    
    # Create a string representation
    strategies_str = "_".join(strategies_sorted)
    dates_str = "_".join(dates_sorted)
    tickers_str = "_".join(tickers_sorted) if tickers_sorted else "all"
    
    # Create a compact representation
    if len(strategies_str) > 30:
        # If too long, use a hash of the strategies
        hash_obj = hashlib.md5(strategies_str.encode())
        strategies_str = f"multi_{hash_obj.hexdigest()[:8]}"
    
    if len(dates_str) > 30:
        # If too many dates, use first and count
        dates_count = len(dates_sorted)
        dates_str = f"{dates_sorted[0]}_plus_{dates_count - 1}"
    
    if len(tickers_str) > 30:
        # If too many tickers, use count
        tickers_count = len(tickers_sorted)
        if tickers_count > 3:
            sample_tickers = "_".join(tickers_sorted[:3])
            tickers_str = f"{sample_tickers}_plus_{tickers_count - 3}"
    
    # Combine components
    name_components = [
        mode,
        strategies_str,
        dates_str,
        tickers_str
    ]
    
    return sanitize_name("_".join(name_components))


def create_timestamp_name(prefix: str = "") -> str:
    """
    Create a name with a timestamp.
    
    Args:
        prefix: Optional prefix for the name
        
    Returns:
        Name with timestamp
    """
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if prefix:
        return f"{sanitize_name(prefix)}_{timestamp}"
    return timestamp


def create_resource_name(resource_type: str, identifier: str, extension: str = "") -> str:
    """
    Create a standardized resource name.
    
    Args:
        resource_type: Type of resource (e.g., "backtest", "portfolio", "analysis")
        identifier: Unique identifier for the resource
        extension: Optional file extension
        
    Returns:
        Standardized resource name
    """
    name = f"{resource_type}_{sanitize_name(identifier)}"
    if extension:
        # Ensure extension starts with a dot
        if not extension.startswith('.'):
            extension = f".{extension}"
        name = f"{name}{extension}"
    return name


def create_monolith_directory_structure(base_output_dir: str, 
                                       strategy: str,
                                       date_range: str,
                                       timestamp: Optional[str] = None) -> str:
    """
    Create directory structure following the monolith pattern.
    
    Pattern: {timestamp}/{strategy}/{date_range}
    Example: 20250616_231605/mse/2024-12-12_to_2025-06-09
    
    Args:
        base_output_dir: Base output directory path
        strategy: Strategy name (e.g., 'mse')
        date_range: Date range string (e.g., '2024-12-12_to_2025-06-09')
        timestamp: Optional timestamp, generated if not provided
        
    Returns:
        Full path to the strategy-specific directory
    """
    import datetime
    from pathlib import Path
    
    if not timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the clean directory structure
    strategy_dir = Path(base_output_dir) / timestamp / strategy / date_range
    strategy_dir.mkdir(parents=True, exist_ok=True)
      # Create standard subdirectories
    subdirs = ['data/base_data', 'data/strategy_trades', 'data/risk_approved_trades', 
               'analysis_reports', 'analysis_reports/individual', 'analysis_reports/portfolio',
               'visualizations', 'visualizations/individual', 'visualizations/portfolio',
               'reports', 'tickers']
    
    for subdir in subdirs:
        (strategy_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return str(strategy_dir)


def extract_strategy_and_date_from_results(results: Dict[str, Any]) -> tuple:
    """
    Extract strategy name and date range from results structure.
    
    Args:
        results: Results dictionary from backtest execution
        
    Returns:
        Tuple of (strategy_name, date_range)
    """
    if not results:
        return 'unknown_strategy', 'unknown_date_range'
    
    # Try 'runs' structure first
    if 'runs' in results:
        return _extract_from_runs_structure(results['runs'])
    
    # Try direct structure: strategy -> date_range -> ticker -> data
    return _extract_from_direct_structure(results)


def _extract_from_runs_structure(runs_data: Dict[str, Any]) -> tuple:
    """Extract from runs structure."""
    for run_data in runs_data.values():
        if isinstance(run_data, dict):
            for ticker_data in run_data.values():
                if isinstance(ticker_data, dict):
                    strategy = ticker_data.get('strategy')
                    date_range = ticker_data.get('date_range')
                    if strategy and date_range:
                        return strategy, date_range
    return 'unknown_strategy', 'unknown_date_range'


def _extract_from_direct_structure(results: Dict[str, Any]) -> tuple:
    """Extract from direct structure."""
    for strategy, strategy_data in results.items():
        if isinstance(strategy_data, dict):
            for date_key in strategy_data.keys():
                return strategy, date_key
    return 'unknown_strategy', 'unknown_date_range'
