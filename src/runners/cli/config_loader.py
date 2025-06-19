"""
Configuration loading utilities for the unified backtester CLI.
"""

import json
import yaml
from pathlib import Path
from typing import Any

from config.unified_config import (
    BacktestConfig,
    get_conservative_config,
    get_aggressive_config,
    get_options_config
)
from .date_utils import parse_dates


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
        
        config = BacktestConfig(**config_data)
    elif args.template:
        # Load from template
        if args.template == 'conservative':
            config = get_conservative_config()
        elif args.template == 'aggressive':
            config = get_aggressive_config()
        elif args.template == 'options':
            config = get_options_config()
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
    
    return config
