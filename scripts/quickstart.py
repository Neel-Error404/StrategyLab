"""
StrategyLab Backtester - Interactive Quickstart Script

This script provides an interactive, guided experience for running your first
backtest. It walks through strategy selection, configuration, and execution
with helpful prompts and explanations.

Usage:
    python scripts/quickstart.py [--advanced] [--non-interactive]

Options:
    --advanced        : Show advanced configuration options
    --non-interactive : Run with default settings (no prompts)
"""

import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(message: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{message:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_section(title: str):
    """Print a section header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{title}{Colors.END}")
    print(f"{Colors.CYAN}{'-' * len(title)}{Colors.END}")

def print_success(message: str):
    """Print a success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")

def print_info(message: str):
    """Print an info message"""
    print(f"{Colors.BLUE}ℹ {message}{Colors.END}")

def print_warning(message: str):
    """Print a warning message"""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")

def print_error(message: str):
    """Print an error message"""
    print(f"{Colors.RED}✗ {message}{Colors.END}")

def prompt_user(question: str, default: str = "") -> str:
    """Prompt user for input with optional default"""
    if default:
        prompt = f"{Colors.CYAN}? {question} [{default}]:{Colors.END} "
    else:
        prompt = f"{Colors.CYAN}? {question}:{Colors.END} "

    response = input(prompt).strip()
    return response if response else default

def prompt_choice(question: str, choices: List[Tuple[str, str]], default_index: int = 0) -> str:
    """Prompt user to select from a list of choices"""
    print(f"\n{Colors.CYAN}? {question}{Colors.END}")

    for i, (choice, description) in enumerate(choices, 1):
        default_marker = f" {Colors.GREEN}(recommended){Colors.END}" if i - 1 == default_index else ""
        print(f"  {i}. {Colors.BOLD}{choice}{Colors.END}{default_marker}")
        print(f"     {description}")

    while True:
        response = prompt_user(f"Enter choice [1-{len(choices)}]", str(default_index + 1))
        try:
            choice_index = int(response) - 1
            if 0 <= choice_index < len(choices):
                return choices[choice_index][0]
            else:
                print_error(f"Please enter a number between 1 and {len(choices)}")
        except ValueError:
            print_error("Please enter a valid number")

def confirm(question: str, default: bool = True) -> bool:
    """Ask user for yes/no confirmation"""
    default_str = "Y/n" if default else "y/N"
    response = prompt_user(f"{question} ({default_str})", "y" if default else "n").lower()

    if response in ('y', 'yes'):
        return True
    elif response in ('n', 'no'):
        return False
    else:
        return default

def check_prerequisites() -> bool:
    """Check if prerequisites are met"""
    print_section("Checking Prerequisites")

    all_good = True

    # Check if we're in the right directory
    if not Path('src/runners/unified_runner.py').exists():
        print_error("unified_runner.py not found. Are you in the backtester directory?")
        all_good = False
    else:
        print_success("Backtester directory confirmed")

    # Check if strategies are registered
    try:
        from strategies.register_strategies import StrategyFactory, register_all_strategies
        register_all_strategies()
        strategies = StrategyFactory.list_strategies()

        if strategies:
            print_success(f"Strategies registered ({len(strategies)} available)")
        else:
            print_error("No strategies registered")
            all_good = False
    except ImportError as e:
        print_error(f"Cannot import strategy module: {e}")
        all_good = False

    # Check .env file exists
    if Path('.env').exists():
        print_success(".env file found")
    else:
        print_warning(".env file not found - you may not be able to fetch data")
        print_info("See docs/BROKER_SETUP.md for API configuration")

    return all_good

def get_available_strategies() -> List[str]:
    """Get list of available strategies"""
    try:
        from strategies.register_strategies import StrategyFactory, register_all_strategies
        register_all_strategies()
        return StrategyFactory.list_strategies()
    except:
        return ['open_source_baseline', 'sma_crossover', 'bollinger_bands']

def select_strategy() -> str:
    """Interactive strategy selection"""
    print_section("Step 1: Select Strategy")

    strategies = get_available_strategies()

    # Create choices with descriptions
    strategy_descriptions = {
        'open_source_baseline': 'Trend + momentum hybrid (best for learning)',
        'sma_crossover': 'Simple moving average crossover (beginner-friendly)',
        'bollinger_bands': 'Volatility-based strategy (intermediate)',
        'mse_strategy_backtesting': 'Multi-signal ensemble strategy (advanced)'
    }

    choices = []
    for strategy in strategies:
        description = strategy_descriptions.get(strategy, 'Custom strategy')
        choices.append((strategy, description))

    # Default to open_source_baseline if available
    default_index = 0
    if 'open_source_baseline' in strategies:
        default_index = strategies.index('open_source_baseline')

    selected = prompt_choice("Which strategy would you like to test?", choices, default_index)

    print_info(f"Selected: {selected}")
    return selected

def select_tickers(advanced: bool = False) -> List[str]:
    """Interactive ticker selection"""
    print_section("Step 2: Select Tickers")

    print_info("Tickers are the stocks/crypto you want to backtest")

    # Popular tickers
    popular_equity = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
    popular_crypto = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

    print(f"\n{Colors.BOLD}Popular Equity Tickers:{Colors.END}")
    print(f"  {', '.join(popular_equity)}")

    print(f"\n{Colors.BOLD}Popular Crypto Tickers:{Colors.END}")
    print(f"  {', '.join(popular_crypto)}")

    if advanced:
        # Advanced: custom ticker input
        ticker_input = prompt_user(
            "Enter ticker(s) separated by spaces",
            "RELIANCE"
        )
        tickers = [t.strip().upper() for t in ticker_input.split()]
    else:
        # Simple: single ticker selection
        choices = [
            ('RELIANCE', 'Reliance Industries (Indian equity)'),
            ('TCS', 'Tata Consultancy Services (Indian equity)'),
            ('BTCUSDT', 'Bitcoin (Cryptocurrency - 24/7 trading)')
        ]
        ticker = prompt_choice("Select a ticker", choices, 0)
        tickers = [ticker]

    print_info(f"Testing: {', '.join(tickers)}")
    return tickers

def select_date_range(advanced: bool = False) -> str:
    """Interactive date range selection"""
    print_section("Step 3: Select Date Range")

    if advanced:
        # Advanced: custom date range
        start_date = prompt_user("Start date (YYYY-MM-DD)", "2024-01-01")
        end_date = prompt_user("End date (YYYY-MM-DD)", "2024-01-31")
        return f"{start_date}_to_{end_date}"
    else:
        # Simple: predefined ranges
        today = datetime.now()

        choices = [
            ('last_month', f'Last month ({(today - timedelta(days=30)).strftime("%Y-%m-%d")} to {today.strftime("%Y-%m-%d")})'),
            ('2024-01-01_to_2024-01-31', 'January 2024 (good for testing)'),
            ('2024-Q1', 'Q1 2024 (Jan-Mar)'),
            ('2024-01-01_to_2024-06-30', 'First half of 2024 (6 months)')
        ]

        date_range = prompt_choice("Select time period", choices, 1)

        # Handle "last_month" special case
        if date_range == 'last_month':
            start = (today - timedelta(days=30)).strftime("%Y-%m-%d")
            end = today.strftime("%Y-%m-%d")
            date_range = f"{start}_to_{end}"

        print_info(f"Date range: {date_range}")
        return date_range

def select_template() -> str:
    """Interactive risk template selection"""
    print_section("Step 4: Select Risk Template")

    print_info("Risk templates control position sizing and stop losses")

    choices = [
        ('conservative', 'Low risk - 15% max position, suitable for learning'),
        ('minimal', 'Ultra safe - 5% max position, very conservative'),
        ('aggressive', 'High risk - 20% max position, higher returns/losses')
    ]

    template = prompt_choice("Select risk template", choices, 0)

    print_info(f"Using template: {template}")
    return template

def configure_advanced_options() -> Dict[str, any]:
    """Configure advanced options"""
    print_section("Advanced Options")

    options = {}

    if confirm("Enable parallel processing? (faster for multiple tickers)", True):
        options['parallel'] = True
        workers = prompt_user("Number of workers (0 for auto)", "0")
        if int(workers) > 0:
            options['max_workers'] = int(workers)

    if confirm("Generate analysis reports?", True):
        options['analyze'] = True

    if confirm("Generate visualizations?", True):
        options['visualize'] = True

    return options

def build_command(strategy: str, tickers: List[str], date_range: str,
                  template: str, advanced_options: Dict = None) -> str:
    """Build the CLI command"""

    cmd_parts = [
        'python',
        'src/runners/unified_runner.py',
        '--mode', 'backtest',
        '--strategies', strategy,
        '--template', template,
        '--date-ranges', date_range,
        '--tickers', ' '.join(tickers)
    ]

    if advanced_options:
        if advanced_options.get('parallel'):
            cmd_parts.append('--parallel')
            if 'max_workers' in advanced_options:
                cmd_parts.extend(['--max-workers', str(advanced_options['max_workers'])])

        # Note: analyze and visualize are handled by --mode backtest by default

    return ' '.join(cmd_parts)

def display_summary(strategy: str, tickers: List[str], date_range: str,
                   template: str, command: str):
    """Display configuration summary"""
    print_section("Configuration Summary")

    print(f"{Colors.BOLD}Strategy:{Colors.END} {strategy}")
    print(f"{Colors.BOLD}Tickers:{Colors.END} {', '.join(tickers)}")
    print(f"{Colors.BOLD}Date Range:{Colors.END} {date_range}")
    print(f"{Colors.BOLD}Risk Template:{Colors.END} {template}")

    print(f"\n{Colors.BOLD}Command to execute:{Colors.END}")
    print(f"{Colors.CYAN}{command}{Colors.END}")

def execute_backtest(command: str) -> bool:
    """Execute the backtest command"""
    print_section("Executing Backtest")

    print_info("Starting backtest... This may take a few minutes.")
    print_info("You'll see progress updates as the system works.")

    try:
        # Execute command
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=False
        )

        if result.returncode == 0:
            print_success("Backtest completed successfully!")
            return True
        else:
            print_error(f"Backtest failed with exit code {result.returncode}")
            return False

    except Exception as e:
        print_error(f"Error executing backtest: {e}")
        return False

def show_next_steps(success: bool):
    """Show next steps to the user"""
    print_section("Next Steps")

    if success:
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 Congratulations!{Colors.END} You've run your first backtest.\n")

        print(f"{Colors.BOLD}Your results are in:{Colors.END}")
        print(f"  📁 outputs/[timestamp]/")
        print(f"     ├── metrics/performance_metrics.csv  (key statistics)")
        print(f"     ├── trades/trades.csv                (all trades)")
        print(f"     └── visualizations/                  (charts)\n")

        print(f"{Colors.BOLD}Understand your results:{Colors.END}")
        print(f"  📖 Read: docs/OUTPUT_GUIDE.md")
        print(f"  📊 Key metrics: Total Return, Sharpe Ratio, Max Drawdown\n")

        print(f"{Colors.BOLD}Try next:{Colors.END}")
        print(f"  1. Run this script again with different settings")
        print(f"  2. Test multiple tickers: python scripts/quickstart.py --advanced")
        print(f"  3. Create your own strategy: docs/STRATEGY_GUIDE.md")
        print(f"  4. Fetch more data: python src/runners/unified_runner.py --mode fetch\n")

    else:
        print(f"{Colors.YELLOW}The backtest encountered an error.{Colors.END}\n")

        print(f"{Colors.BOLD}Troubleshooting steps:{Colors.END}")
        print(f"  1. Check if you have market data:")
        print(f"     python src/runners/unified_runner.py --mode fetch")
        print(f"  2. Verify broker API setup:")
        print(f"     docs/BROKER_SETUP.md")
        print(f"  3. Check error reference:")
        print(f"     docs/ERROR_REFERENCE.md")
        print(f"  4. Run verification:")
        print(f"     python scripts/verify_setup.py\n")

def main():
    """Main interactive flow"""
    import argparse

    parser = argparse.ArgumentParser(description='Interactive backtest quickstart')
    parser.add_argument('--advanced', action='store_true', help='Show advanced options')
    parser.add_argument('--non-interactive', action='store_true', help='Run with defaults')
    args = parser.parse_args()

    print_header("StrategyLab Backtester - Interactive Quickstart")

    print(f"{Colors.BOLD}Welcome!{Colors.END} This script will guide you through running your first backtest.")
    print("Answer a few simple questions and we'll configure everything for you.\n")

    # Check prerequisites
    if not check_prerequisites():
        print_error("\nPrerequisites not met. Please run: python scripts/verify_setup.py")
        sys.exit(1)

    if args.non_interactive:
        # Non-interactive mode: use defaults
        print_info("Running in non-interactive mode with default settings...")
        strategy = 'open_source_baseline'
        tickers = ['RELIANCE']
        date_range = '2024-01-01_to_2024-01-31'
        template = 'conservative'
        advanced_options = {}
    else:
        # Interactive mode
        strategy = select_strategy()
        tickers = select_tickers(args.advanced)
        date_range = select_date_range(args.advanced)
        template = select_template()

        advanced_options = {}
        if args.advanced and confirm("Configure advanced options?", False):
            advanced_options = configure_advanced_options()

    # Build command
    command = build_command(strategy, tickers, date_range, template, advanced_options)

    # Display summary
    display_summary(strategy, tickers, date_range, template, command)

    # Confirm execution
    if not args.non_interactive:
        print()
        if not confirm("Ready to run the backtest?", True):
            print_info("Backtest cancelled. You can run the command manually:")
            print(f"{Colors.CYAN}{command}{Colors.END}")
            sys.exit(0)

    # Execute
    success = execute_backtest(command)

    # Show next steps
    show_next_steps(success)

    sys.exit(0 if success else 1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Quickstart cancelled by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
