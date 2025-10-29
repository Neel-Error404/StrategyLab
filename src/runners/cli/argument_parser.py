"""
Argument parser for the unified backtester CLI.
Combines features from both backtester_runner.py and enhanced_runner.py.
"""

import argparse


def create_argument_parser():
    """
    Create and return the argument parser for the unified backtester CLI.
    Combines the best features from both backtester_runner.py and enhanced_runner.py.
    """
    parser = argparse.ArgumentParser(
        description="Unified Backtester with Smart Workflow Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full workflow (backtest + analysis + visualization)
  python unified_runner.py --mode backtest --dates 2024-01-01 2024-01-02 --tickers RELIANCE TCS
  
  # Run only analysis (runs own backtest)
  python unified_runner.py --mode analyze --dates 2024-01-01 --tickers RELIANCE
  
  # Run only visualization (runs own backtest)
  python unified_runner.py --mode visualize --dates 2024-01-01 --tickers RELIANCE TCS
  
  # Validate data
  python unified_runner.py --mode validate --dates 2024-01-01 2024-01-02
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['validate', 'backtest', 'analyze', 'visualize', 'fetch', 'replay', 'update', 'optimize'],
        required=True,
        help="Mode to run: 'backtest' (full workflow), 'analyze' (analysis only), "
             "'visualize' (visualization only), 'validate' (data checks), "
             "'fetch' (download market data), 'replay' (manifest replay), "
             "'update' (incremental pool maintenance), or 'optimize' (parameter sweeps)."
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        '--template',
        choices=['conservative', 'aggressive'],
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
        help="List of date ranges in YYYY-MM-DD_to_YYYY-MM-DD format"
    )
    
    parser.add_argument(
        '--tickers',
        nargs='+',
        help="List of ticker symbols"
    )
    
    parser.add_argument(
        '--strategies',
        nargs='+',
        default=['mse'],
        help="List of strategy names (default: ['mse'])"
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help="Enable parallel processing"
    )

    # Update mode arguments
    parser.add_argument(
        '--pool-path',
        type=str,
        help="Path to existing data pool directory when using --mode update."
    )
    parser.add_argument(
        '--extend-to',
        type=str,
        help="Target end date for update mode (YYYY-MM-DD, defaults to today)."
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Preview changes without modifying any files in update mode."
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help="Only validate pool integrity without writing new data in update mode."
    )
    parser.add_argument(
        '--yes',
        action='store_true',
        help="Bypass confirmation prompts in update mode."
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help="Skip backup creation before merges (update mode only, use with caution)."
    )
    
    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help="Skip visualization generation"
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
