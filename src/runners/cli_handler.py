#!/usr/bin/env python3
"""
CLI Handler Module for Unified Backtester

Handles command-line argument parsing, configuration loading, and date utilities.
This module extracts all CLI-related functionality from the monolithic unified_runner.py.
"""

import argparse
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import List

from config.unified_config import (
    BacktestConfig,
    get_conservative_config,
    get_aggressive_config,
    get_options_config,
    get_minimal_config
)


def create_argument_parser():
    """
    Create and return the argument parser for the unified backtester CLI.
    Combines the best features from both backtester_runner.py and enhanced_runner.py.
    """
    parser = argparse.ArgumentParser(
        description="Unified Backtester with Smart Workflow Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,        epilog="""
Examples:
  # Minimal usage - auto-discover tickers from date range data pools
  python unified_runner.py --mode backtest --date-ranges 2024-12-12_to_2025-06-09
  python unified_runner.py --mode analyze --date-ranges 2024-12-12_to_2025-06-09
  python unified_runner.py --mode visualize --date-ranges 2024-12-12_to_2025-06-09
  python unified_runner.py --mode validate --date-ranges 2024-12-12_to_2025-06-09
  
  # Interactive fetch mode (no arguments required)
  python unified_runner.py --mode fetch
  
  # Specific tickers (overrides auto-discovery)
  python unified_runner.py --mode backtest --date-ranges 2024-12-12_to_2025-06-09 --tickers RELIANCE TCS
  
  # Full control with advanced features
  python unified_runner.py --mode backtest --date-ranges 2024-12-12_to_2025-06-09 --tickers RELIANCE TCS --strategies sma_crossover --template aggressive --parallel
  
  # Explicit fetch with parameters
  python unified_runner.py --mode fetch --date-ranges 2024-12-12_to_2025-06-09 --tickers RELIANCE TCS
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['validate', 'backtest', 'analyze', 'visualize', 'fetch'],
        required=True,
        help="Mode to run: 'backtest' (full workflow), 'analyze' (analysis only), 'visualize' (visualization only), 'validate' (data validation), 'fetch' (download market data)"
    )
    
    parser.add_argument(
        '--config',        type=str,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        '--template',
        choices=['conservative', 'aggressive', 'options', 'minimal'],
        help="Use a predefined configuration template"
    )
    
    parser.add_argument(
        '--dates',
        nargs='+',
        help="List of dates in YYYY-MM-DD format"
    )
    
    parser.add_argument(
        '--date-ranges',
        nargs='+',
        help="List of date ranges in YYYY-MM-DD_to_YYYY-MM-DD format"    )
    
    parser.add_argument(
        '--tickers',
        nargs='+',
        help="List of ticker symbols (optional - auto-discovered from data pools if not provided)"
    )
    
    parser.add_argument(
        '--strategies',
        nargs='+',
        default=['sma_crossover'],
        help="List of strategy names (default: ['sma_crossover'])"
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help="Enable parallel processing"    )
    
    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help="Skip visualization generation"
    )
    
    parser.add_argument(
        '--trade-source',
        choices=['auto', 'strategy_trades', 'risk_approved_trades'],
        default='auto',
        help="Trade data source for visualizations: 'auto' (fallback logic), 'strategy_trades' (raw strategy output), 'risk_approved_trades' (risk-adjusted trades)"
    )
    
    parser.add_argument(
        '--optimization-params',
        type=str,
        help="JSON string with optimization parameters"
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help="Output directory for results"
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Logging level"
    )
    
    return parser


def parse_dates(dates: List[str]) -> List[str]:
    """
    Parse and normalize date strings to ensure consistency.
    
    Args:
        dates: List of date strings in various formats
        
    Returns:
        List of normalized date strings
    """
    normalized_dates = []
    
    for date_str in dates:
        try:
            # Handle different date formats
            if '_to_' in date_str:
                # Already a date range
                start_str, end_str = date_str.split('_to_')
                start_date = datetime.strptime(start_str, "%Y-%m-%d")
                end_date = datetime.strptime(end_str, "%Y-%m-%d")
                normalized_dates.append(f"{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}")
            else:
                # Single date
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                normalized_dates.append(date_obj.strftime("%Y-%m-%d"))
                
        except ValueError as e:
            raise ValueError(f"Invalid date format '{date_str}': {e}")
    
    return normalized_dates


def load_config_from_args(args) -> BacktestConfig:
    """
    Load configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        BacktestConfig instance
    """
    # Load base configuration
    if args.config:
        # Load from YAML file
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        config = BacktestConfig.from_dict(config_data)
    elif args.template:        # Load from template
        if args.template == 'conservative':
            config = get_conservative_config()
        elif args.template == 'aggressive':
            config = get_aggressive_config()
        elif args.template == 'options':
            config = get_options_config()
        elif args.template == 'minimal':
            config = get_minimal_config()
        else:
            raise ValueError(f"Unknown template: {args.template}")
    else:
        # Use default conservative configuration
        config = get_conservative_config()
      # Override configuration with CLI arguments
    if args.dates:
        normalized_dates = parse_dates(args.dates)
        config.strategy.date_ranges = normalized_dates
    elif args.date_ranges:
        normalized_dates = parse_dates(args.date_ranges)
        config.strategy.date_ranges = normalized_dates
    
    if args.tickers:
        config.strategy.tickers = args.tickers
    
    if args.strategies:
        config.strategy.names = args.strategies
    
    if args.mode:
        config.mode = args.mode
    
    if args.output_dir:
        config.output.output_dir = args.output_dir
    
    if args.log_level:
        config.logging.level = args.log_level
    
    if args.optimization_params:
        try:
            config.optimization_params = json.loads(args.optimization_params)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid optimization parameters JSON: {e}")
    
    if args.trade_source:
        config.output.visualization_trade_source = args.trade_source
    
    return config


class CLIHandler:
    """
    Command Line Interface handler for the unified backtester.
    
    Provides a clean interface for parsing command line arguments,
    loading configuration, and preparing the system for execution.
    """
    
    def __init__(self):
        """Initialize the CLI handler."""
        self.parser = create_argument_parser()
    
    def parse_arguments(self, args=None):
        """
        Parse command line arguments.
        
        Args:
            args: Optional list of arguments (for testing)
            
        Returns:
            Parsed arguments namespace
        """
        return self.parser.parse_args(args)
    
    def load_config(self, args) -> BacktestConfig:
        """
        Load configuration from parsed arguments.
        
        Args:
            args: Parsed arguments namespace
            
        Returns:
            BacktestConfig: Loaded configuration object
        """
        return load_config_from_args(args)
    
    def validate_arguments(self, args) -> bool:
        """
        Validate parsed arguments for consistency.
        
        Args:
            args: Parsed arguments namespace
            
        Returns:
            bool: True if arguments are valid
        """
        # Fetch mode can run with zero arguments (interactive mode)
        if args.mode == 'fetch' and not args.dates and not args.date_ranges and not args.tickers:
            return True
            
        # All other modes require at least date-ranges (tickers are now optional)
        if args.mode in ['backtest', 'analyze', 'visualize', 'validate']:
            if not args.dates and not args.date_ranges:
                print("Error: Date ranges must be specified using --dates or --date-ranges")
                return False
        
        # Fetch mode with partial arguments still needs date-ranges
        if args.mode == 'fetch' and (args.dates or args.date_ranges or args.tickers):
            if not args.dates and not args.date_ranges:
                print("Error: When providing any arguments to fetch mode, date ranges must be specified using --dates or --date-ranges")
                return False
            
        return True
