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
from multiprocessing import cpu_count

from config.unified_config import (
    BacktestConfig,
    get_conservative_config,
    get_aggressive_config,
    get_minimal_config,
    get_debug_config,
    ExitConfig,
    ThresholdConfig,
    TimeoutConfig,
    SquareOffConfig,
    RiskConfig,
)


def _load_exit_template(path: Path) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Exit template must be a YAML object")
    if 'exit' in data:
        return data['exit']
    if 'strategy' in data and isinstance(data['strategy'], dict) and 'exit' in data['strategy']:
        return data['strategy']['exit']
    return data


def _exit_config_from_dict(exit_dict: dict) -> ExitConfig:
    stop_loss_cfg = exit_dict.get('stop_loss')
    take_profit_cfg = exit_dict.get('take_profit')
    timeout_cfg = exit_dict.get('timeout')
    square_off_cfg = exit_dict.get('square_off')

    return ExitConfig(
        mode=exit_dict.get('mode', 'manual'),
        stop_loss=ThresholdConfig(**stop_loss_cfg) if stop_loss_cfg else ThresholdConfig(),
        take_profit=ThresholdConfig(**take_profit_cfg) if take_profit_cfg else ThresholdConfig(enabled=False, value=0.04),
        timeout=TimeoutConfig(**timeout_cfg) if timeout_cfg else TimeoutConfig(),
        square_off=SquareOffConfig(**square_off_cfg) if square_off_cfg else SquareOffConfig()
    )

def _risk_config_from_dict(risk_dict: dict) -> RiskConfig:
    """
    Convert a dict into a RiskConfig dataclass. Missing fields fall back to defaults.
    """
    params = {}
    for field_name in RiskConfig.__dataclass_fields__.keys():
        if field_name in risk_dict:
            params[field_name] = risk_dict[field_name]
    return RiskConfig(**params)


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
        '--run-label',
        type=str,
        help="Optional label prefix for the run directory (requires --output-dir)"
    )
    
    parser.add_argument(
        '--timeframes',
        nargs='+',
        default=None,
        help="List of timeframes for data fetching (e.g., '1m', '5m', '15m', '30m', '1h', 'day'). "
             "Defaults to configuration or 1m if unspecified."
    )
    
    parser.add_argument(
        '--fetch-max-retries',
        type=int,
        help="Override the maximum retry attempts per API chunk during fetch operations (default 5)."
    )
    
    parser.add_argument(
        '--fetch-failure-threshold',
        type=float,
        help="Abort a ticker if the chunk failure ratio exceeds this threshold (default 0.5)."
    )
    
    parser.add_argument(
        '--fetch-min-chunks-before-abort',
        type=int,
        help="Minimum number of chunks processed before the failure threshold is evaluated (default 5)."
    )

    parser.add_argument(
        '--exit-template',
        type=str,
        help="Path to a YAML file containing an exit configuration block"
    )
    
    parser.add_argument(
        '--risk-template',
        type=str,
        help="Path to a YAML file containing a risk configuration block"
    )
    
    parser.add_argument(
        '--skip-symbol-validation',
        action='store_true',
        help="Skip upfront symbol/instrument validation before fetching (enabled by default)."
    )
    
    parallel_group = parser.add_mutually_exclusive_group()
    parallel_group.add_argument(
        '--parallel',
        action='store_true',
        help="Force parallel processing (overrides template defaults)"
    )
    parallel_group.add_argument(
        '--sequential',
        action='store_true',
        help="Force sequential execution (disables multiprocessing)"
    )

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


def parse_dates(date_ranges: List[str]) -> List[str]:
    """
    Normalize explicit date ranges in YYYY-MM-DD_to_YYYY-MM-DD format.
    """
    normalized = []

    for raw in date_ranges:
        if "_to_" not in raw:
            raise ValueError(
                f"Invalid date range '{raw}': expected format YYYY-MM-DD_to_YYYY-MM-DD"
            )

        start_str, end_str = raw.split("_to_", 1)
        try:
            start_date = datetime.strptime(start_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_str, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"Invalid date range '{raw}': {exc}") from exc

        if end_date < start_date:
            raise ValueError(
                f"Invalid date range '{raw}': end date precedes start date"
            )

        normalized.append(
            f"{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}"
        )

    return normalized


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
    if args.date_ranges:
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

    if args.run_label:
        if not args.output_dir:
            raise ValueError("--run-label requires --output-dir to keep experiment folders grouped")
        config.run_label = args.run_label
    
    if args.output_dir:
        config.output.output_dir = args.output_dir
    
    if args.log_level:
        config.logging.level = args.log_level

    # Execution overrides
    if getattr(args, 'parallel', False):
        config.execution.parallel_processing = True
    elif getattr(args, 'sequential', False):
        config.execution.parallel_processing = False

    if hasattr(args, 'max_workers') and args.max_workers:
        requested_workers = int(args.max_workers)
        cpu_cores = cpu_count()
        recommended_max = cpu_cores * 2
        if requested_workers > recommended_max:
            print(f"⚠️  Requested {requested_workers} workers exceeds recommended maximum ({recommended_max}). Limiting to {recommended_max}.")
            requested_workers = recommended_max
        if requested_workers < 1:
            raise ValueError("--max-workers must be at least 1")
        config.execution.max_workers = requested_workers
    
    if args.optimization_params:
        try:
            config.optimization_params = json.loads(args.optimization_params)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid optimization parameters JSON: {e}")
    
    if args.trade_source:
        config.output.visualization_trade_source = args.trade_source

    if args.timeframes:
        if args.mode != 'fetch':
            raise ValueError("--timeframes override is only supported for fetch mode. Update your YAML template to change backtest timeframes.")
        normalized_timeframes = [tf.strip() for tf in args.timeframes if tf.strip()]
        if not normalized_timeframes:
            raise ValueError("At least one valid timeframe must be provided with --timeframes")

        config.timeframes = normalized_timeframes

        if hasattr(config, 'fetch'):
            config.fetch.timeframes = normalized_timeframes
    
    if args.fetch_max_retries is not None:
        if hasattr(config, 'fetch'):
            config.fetch.max_retries = args.fetch_max_retries
        config.fetch_max_retries = args.fetch_max_retries
    
    if args.fetch_failure_threshold is not None:
        if hasattr(config, 'fetch'):
            config.fetch.failure_threshold = args.fetch_failure_threshold
        config.fetch_failure_threshold = args.fetch_failure_threshold
    
    if args.fetch_min_chunks_before_abort is not None:
        if hasattr(config, 'fetch'):
            config.fetch.min_chunks_before_abort = args.fetch_min_chunks_before_abort
        config.fetch_min_chunks_before_abort = args.fetch_min_chunks_before_abort
    
    if getattr(args, 'skip_symbol_validation', False):
        if hasattr(config, 'fetch'):
            config.fetch.validate_symbols = False
        config.fetch_validate_symbols = False

    # Visualization toggle (stored on output config so workflow can honor it)
    if hasattr(config.output, 'skip_visualization'):
        config.output.skip_visualization = bool(args.skip_visualization)
    else:
        setattr(config.output, 'skip_visualization', bool(args.skip_visualization))

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

    if args.exit_template:
        template_path = Path(args.exit_template)
        if not template_path.exists():
            raise FileNotFoundError(f"Exit template not found: {template_path}")
        exit_data = _load_exit_template(template_path)
        config.strategy.exit = _exit_config_from_dict(exit_data)
        config.exit_template_path = str(template_path)
    
    if getattr(args, 'risk_template', None):
        risk_path = Path(args.risk_template)
        if not risk_path.exists():
            raise FileNotFoundError(f"Risk template not found: {risk_path}")
        with open(risk_path, "r", encoding="utf-8") as risk_file:
            risk_data = yaml.safe_load(risk_file) or {}
        if 'risk' in risk_data:
            risk_data = risk_data['risk']
        config.risk = _risk_config_from_dict(risk_data)
        config.risk_template_path = str(risk_path)

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

        if args.mode == 'fetch' and not args.date_ranges and not args.tickers:
            return True
            
        # All other modes require at least date-ranges (tickers are now optional)
        if args.mode in ['backtest', 'analyze', 'visualize', 'validate']:
            if not args.date_ranges:
                print("Error: Date ranges must be specified using --date-ranges")
                return False
            
            # Backtest mode requires exactly one strategy
            if args.mode == 'backtest':
                if not args.strategies:
                    print("Error: Strategies must be specified using --strategies for backtest mode")
                    return False
                if len(args.strategies) != 1:
                    print("Error: Only one strategy may be run at a time. Please provide a single --strategies value.")
                    return False
        
        # Fetch mode with partial arguments still needs date-ranges
        if args.mode == 'fetch' and (args.date_ranges or args.tickers):
            if not args.date_ranges:
                print("Error: When providing any arguments to fetch mode, date ranges must be specified using --date-ranges")
                return False
            
        return True
