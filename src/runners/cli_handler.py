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
    get_minimal_config,
    get_debug_config
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
  # Minimal usage - defaults to debug template for pure strategy testing
  python unified_runner.py --mode backtest --date-ranges 2024-12-12_to_2025-06-09
  python unified_runner.py --mode analyze --date-ranges 2024-12-12_to_2025-06-09
  python unified_runner.py --mode visualize --date-ranges 2024-12-12_to_2025-06-09
  python unified_runner.py --mode validate --date-ranges 2024-12-12_to_2025-06-09
  
  # Interactive fetch mode (no arguments required)
  python unified_runner.py --mode fetch
  
  # Specific tickers (overrides auto-discovery) - still uses debug template
  python unified_runner.py --mode backtest --date-ranges 2024-12-12_to_2025-06-09 --tickers RELIANCE TCS
  
  # Production backtesting with risk management
  python unified_runner.py --mode backtest --date-ranges 2024-12-12_to_2025-06-09 --tickers RELIANCE TCS --template conservative --parallel
  
  # Explicit fetch with parameters
  python unified_runner.py --mode fetch --date-ranges 2024-12-12_to_2025-06-09 --tickers RELIANCE TCS
  
  # Incremental pool update
  python unified_runner.py --mode update --pool-path data/pools/2025-04-01_to_2025-10-08 --dry-run
  python unified_runner.py --mode update --pool-path data/pools/2025-04-01_to_2025-10-08 --yes
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['validate', 'backtest', 'analyze', 'visualize', 'fetch', 'replay', 'update'],
        required=True,
        help="Mode to run: 'backtest' (full workflow), 'analyze' (analysis only), 'visualize' (visualization only), 'validate' (data validation), 'fetch' (download market data), 'update' (incremental pool update)"
    )
    
    parser.add_argument(
        '--config',        type=str,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        '--manifest',
        type=str,
        help="Path to session manifest for replay mode"
    )
    parser.add_argument(
        '--template',
        choices=['conservative', 'aggressive', 'minimal', 'debug'],
        help="Use a predefined configuration template (default: debug - for strategy testing)"
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
        help="List of strategy names (required)"
    )
    
    parser.add_argument(
        '--timeframes',
        nargs='+',
        default=['1m'],
        help="List of timeframes for data fetching (e.g., '1m', '5m', '15m', '30m', '1h', 'day') (default: ['1m'])"
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help="Enable parallel processing"    )

    parser.add_argument(
        '--max-workers',
        type=int,
        help="Maximum number of parallel workers (default: 4, max recommended: CPU_COUNT * 2)"
    )
    
    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help="Skip visualization generation"
    )

    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help="Skip data validation"
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
    
    # Update mode specific arguments
    parser.add_argument(
        '--pool-path',
        type=str,
        help="Path to existing data pool directory for update mode (e.g., data/pools/2025-04-01_to_2025-10-08)"
    )
    
    parser.add_argument(
        '--target-end-date',
        type=str,
        help="Target end date for incremental update (default: today, format: YYYY-MM-DD)"
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Preview changes without executing (update mode)"
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help="Only validate pool integrity without updating (update mode)"
    )
    
    parser.add_argument(
        '--yes',
        action='store_true',
        help="Skip confirmation prompts (update mode)"
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help="Skip backup before merging data (update mode - USE WITH CAUTION)"
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
        elif args.template == 'minimal':
            config = get_minimal_config()
        elif args.template == 'debug':
            config = get_debug_config()
        else:
            raise ValueError(f"Unknown template: {args.template}")
    else:
        # Use default debug configuration for strategy testing
        config = get_debug_config()
      # Override configuration with CLI arguments
    if args.dates:
        normalized_dates = parse_dates(args.dates)
        config.strategy.date_ranges = normalized_dates
    elif args.date_ranges:
        normalized_dates = parse_dates(args.date_ranges)
        config.strategy.date_ranges = normalized_dates
    
    if args.tickers:
        # Handle both space-separated and comma-separated tickers
        processed_tickers = []
        for ticker_arg in args.tickers:
            if ',' in ticker_arg:
                # Split comma-separated tickers
                processed_tickers.extend([t.strip() for t in ticker_arg.split(',') if t.strip()])
            else:
                processed_tickers.append(ticker_arg.strip())
        config.strategy.tickers = processed_tickers
    
    if args.strategies:
        config.strategy.names = args.strategies
        config.strategy.name = args.strategies[0]
    
    if args.mode:
        config.mode = args.mode
    if getattr(args, 'manifest', None):
        config.replay_manifest = args.manifest

    
    if args.output_dir:
        config.output.output_dir = args.output_dir
    
    if args.log_level:
        config.logging.level = args.log_level

    # Execution overrides
    if hasattr(args, 'parallel') and args.parallel:
        config.execution.parallel_processing = True
    if hasattr(args, 'max_workers') and args.max_workers:
        config.execution.max_workers = int(args.max_workers)
    
    if args.optimization_params:
        try:
            config.optimization_params = json.loads(args.optimization_params)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid optimization parameters JSON: {e}")
    
    if args.trade_source:
        config.output.visualization_trade_source = args.trade_source

    if args.timeframes:
        # Store timeframes in config for fetch mode
        config.timeframes = args.timeframes

    # Validation toggle
    if hasattr(args, 'skip_validation') and args.skip_validation:
        config.validation.enabled = False

    # Update mode specific arguments
    if hasattr(args, 'pool_path') and args.pool_path:
        config.pool_path = args.pool_path
    if hasattr(args, 'target_end_date') and args.target_end_date:
        config.target_end_date = args.target_end_date
    if hasattr(args, 'dry_run') and args.dry_run:
        config.dry_run = True
    if hasattr(args, 'validate_only') and args.validate_only:
        config.validate_only = True
    if hasattr(args, 'yes') and args.yes:
        config.yes = True
    if hasattr(args, 'no_backup') and args.no_backup:
        config.no_backup = True

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
        if args.mode == 'replay':
            if not args.manifest:
                print("Error: Replay mode requires --manifest")
                return False
            return True

        if args.mode == 'fetch' and not args.dates and not args.date_ranges and not args.tickers:
            return True
            
        # All other modes require at least date-ranges (tickers are now optional)
        if args.mode in ['backtest', 'analyze', 'visualize', 'validate']:
            if not args.dates and not args.date_ranges:
                print("Error: Date ranges must be specified using --dates or --date-ranges")
                return False
            
            # Backtest mode requires strategies
            if args.mode == 'backtest' and not args.strategies:
                print("Error: Strategies must be specified using --strategies for backtest mode")
                return False
        
        # Fetch mode with partial arguments still needs date-ranges
        if args.mode == 'fetch' and (args.dates or args.date_ranges or args.tickers):
            if not args.dates and not args.date_ranges:
                print("Error: When providing any arguments to fetch mode, date ranges must be specified using --dates or --date-ranges")
                return False
            
        return True
