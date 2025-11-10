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
        choices=['validate', 'backtest', 'analyze', 'visualize', 'optimize'],
        required=True,
        help="Mode to run: 'backtest' (full workflow), 'analyze' (analysis only), 'visualize' (visualization only)"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        '--template',
        choices=['conservative', 'aggressive', 'options'],
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
