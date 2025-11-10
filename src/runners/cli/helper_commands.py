"""
CLI Helper Commands for Unified Runner

Provides utility commands for listing strategies, verifying configs,
checking data availability, and describing templates.

These commands are designed to help users verify their setup without
running a full backtest.
"""

import sys
from pathlib import Path
from typing import List
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def handle_list_strategies():
    """List all registered strategies"""
    print("\n" + "="*60)
    print("REGISTERED STRATEGIES")
    print("="*60 + "\n")

    try:
        from strategies.register_strategies import StrategyFactory, register_all_strategies

        # Register all strategies
        register_all_strategies()

        # Get list of strategies
        strategies = StrategyFactory.list_strategies()

        if not strategies:
            print("[X] No strategies registered")
            print("\nCheck: src/strategies/register_strategies.py")
            return

        print(f"[OK] Found {len(strategies)} registered strateg{'y' if len(strategies) == 1 else 'ies'}:\n")

        # Strategy descriptions
        descriptions = {
            'open_source_baseline': 'Trend + momentum hybrid (best for learning)',
            'sma_crossover': 'Simple moving average crossover strategy',
            'bollinger_bands': 'Volatility-based Bollinger Bands strategy',
            'mse_strategy_backtesting': 'Multi-signal ensemble strategy (advanced)'
        }

        for i, strategy in enumerate(sorted(strategies), 1):
            description = descriptions.get(strategy, 'Custom strategy')
            print(f"  {i}. {strategy}")
            print(f"     {description}\n")

        if strategies:
            print("Usage:")
            print(f"  python src/runners/unified_runner.py --mode backtest --strategies {strategies[0]} ...")

    except ImportError as e:
        print(f"[X] Error importing strategies: {e}")
        print("\nTroubleshooting:")
        print("  1. Check: src/strategies/register_strategies.py exists")
        print("  2. Run: python scripts/verify_setup.py")
    except Exception as e:
        print(f"[X] Error listing strategies: {e}")


def handle_verify_config(args):
    """Verify YAML configuration template"""
    print("\n" + "="*60)
    print("VERIFY CONFIGURATION TEMPLATE")
    print("="*60 + "\n")

    # Determine which template to verify
    template_name = args.template if hasattr(args, 'template') and args.template else 'conservative'
    config_path = args.config if hasattr(args, 'config') and args.config else None

    if config_path:
        # Verify custom config file
        config_file = Path(config_path)
        if not config_file.exists():
            print(f"[X] Config file not found: {config_path}")
            return

        print(f"Verifying: {config_path}\n")

        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            print("[OK] YAML syntax is valid")

            # Basic validation
            if 'strategy' in config:
                print(f"[OK] Strategy configured: {config.get('strategy', {}).get('name', 'N/A')}")
            else:
                print("[!] Warning: No strategy section found")

            if 'risk' in config:
                print("[OK] Risk management configured")
                risk = config['risk']
                if 'max_position_size' in risk:
                    print(f"  - Max position size: {risk['max_position_size']*100}%")
                if 'stop_loss_pct' in risk:
                    print(f"  - Stop loss: {risk['stop_loss_pct']*100}%")
            else:
                print("[!] Warning: No risk section found")

            print("\n[OK] Configuration file is valid")

        except yaml.YAMLError as e:
            print(f"[X] YAML syntax error: {e}")
        except Exception as e:
            print(f"[X] Error reading config: {e}")

    else:
        # Verify template
        template_path = Path(f'config/templates/{template_name}.yaml')

        if not template_path.exists():
            print(f"[X] Template not found: {template_path}")
            print("\nAvailable templates:")
            templates_dir = Path('config/templates')
            if templates_dir.exists():
                for template in sorted(templates_dir.glob('*.yaml')):
                    print(f"  - {template.stem}")
            return

        print(f"Verifying template: {template_name}\n")

        try:
            with open(template_path, 'r') as f:
                config = yaml.safe_load(f)

            print("[OK] Template YAML syntax is valid")
            print(f"[OK] Template file: {template_path}")

            # Display key settings
            if 'risk' in config:
                print("\nRisk Management Settings:")
                risk = config['risk']
                if 'max_position_size' in risk:
                    print(f"  - Max position size: {risk['max_position_size']*100}%")
                if 'stop_loss_pct' in risk:
                    print(f"  - Stop loss: {risk['stop_loss_pct']*100}%")
                if 'portfolio_risk' in risk:
                    print(f"  - Portfolio risk limit: {risk['portfolio_risk']*100}%")

            print("\n[OK] Template is valid and ready to use")

        except yaml.YAMLError as e:
            print(f"[X] YAML syntax error: {e}")
        except Exception as e:
            print(f"[X] Error reading template: {e}")


def handle_check_data(ticker: str):
    """Check data availability for a ticker"""
    print("\n" + "="*60)
    print(f"DATA AVAILABILITY CHECK: {ticker.upper()}")
    print("="*60 + "\n")

    data_pools_dir = Path('data/pools')

    if not data_pools_dir.exists():
        print("[X] Data pools directory not found")
        print(f"  Expected: {data_pools_dir}")
        print("\nAction needed:")
        print("  1. Fetch data: python src/runners/unified_runner.py --mode fetch --tickers", ticker)
        print("  2. Or run: python src/core/etl/data_fetcher.py")
        return

    # Search for ticker data
    ticker_upper = ticker.upper()
    found_files = []

    for pool_dir in data_pools_dir.iterdir():
        if pool_dir.is_dir():
            # Look for ticker files in this pool
            ticker_files = list(pool_dir.rglob(f'*{ticker_upper}*.parquet')) + \
                          list(pool_dir.rglob(f'*{ticker_upper}*.csv'))

            if ticker_files:
                found_files.extend([(pool_dir.name, f) for f in ticker_files])

    if found_files:
        print(f"[OK] Data found for {ticker_upper}\n")
        print("Data pools containing this ticker:")
        for pool_name, file_path in found_files:
            rel_path = file_path.relative_to(data_pools_dir)
            print(f"  - {pool_name}/")
            print(f"    {rel_path}")
            # Try to get file info
            try:
                file_size = file_path.stat().st_size
                size_mb = file_size / (1024 * 1024)
                print(f"    Size: {size_mb:.2f} MB")
            except:
                pass
        print(f"\n[OK] {ticker_upper} is ready for backtesting")
    else:
        print(f"[X] No data found for {ticker_upper}\n")
        print("Data pools searched:")
        pools_found = [d.name for d in data_pools_dir.iterdir() if d.is_dir()]
        if pools_found:
            for pool in pools_found:
                print(f"  - {pool}/")
        else:
            print("  (no pools found)")

        print("\nAction needed:")
        print(f"  Fetch data: python src/runners/unified_runner.py --mode fetch --tickers {ticker_upper}")


def handle_describe_template(template_name: str):
    """Describe a risk template in detail"""
    print("\n" + "="*60)
    print(f"TEMPLATE DESCRIPTION: {template_name}")
    print("="*60 + "\n")

    template_path = Path(f'config/templates/{template_name}.yaml')

    if not template_path.exists():
        print(f"[X] Template not found: {template_name}")
        print("\nAvailable templates:")
        templates_dir = Path('config/templates')
        if templates_dir.exists():
            for template in sorted(templates_dir.glob('*.yaml')):
                print(f"  - {template.stem}")
        else:
            print("  (templates directory not found)")
        return

    try:
        with open(template_path, 'r') as f:
            config = yaml.safe_load(f)

        # Template descriptions
        descriptions = {
            'conservative': 'Low-risk template suitable for learning and stable returns',
            'aggressive': 'High-risk template for experienced traders seeking higher returns',
            'minimal': 'Ultra-safe template with very conservative position sizing',
            'portfolio_diversified': 'Multi-asset portfolio management template'
        }

        description = descriptions.get(template_name, 'Custom template')
        print(f"Description: {description}\n")

        # Risk settings
        if 'risk' in config:
            print("Risk Management Parameters:")
            print("-" * 40)
            risk = config['risk']

            if 'max_position_size' in risk:
                pct = risk['max_position_size'] * 100
                print(f"  Max Position Size:     {pct:.1f}%")
                print(f"    (Maximum {pct:.1f}% of portfolio per trade)")

            if 'stop_loss_pct' in risk:
                pct = risk['stop_loss_pct'] * 100
                print(f"  Stop Loss:             {pct:.1f}%")
                print(f"    (Exit if loss exceeds {pct:.1f}%)")

            if 'portfolio_risk' in risk:
                pct = risk['portfolio_risk'] * 100
                print(f"  Portfolio Risk Limit:  {pct:.1f}%")
                print(f"    (Max total exposure: {pct:.1f}%)")

            if 'max_trades_per_day' in risk:
                print(f"  Max Trades Per Day:    {risk['max_trades_per_day']}")

        # Strategy settings
        if 'strategy' in config:
            print("\nStrategy Configuration:")
            print("-" * 40)
            strategy = config['strategy']

            if 'name' in strategy:
                print(f"  Strategy Name:         {strategy['name']}")

            if 'parameters' in strategy:
                print("  Parameters:")
                for key, value in strategy['parameters'].items():
                    print(f"    - {key}: {value}")

        print("\nUsage:")
        print(f"  python src/runners/unified_runner.py --mode backtest --template {template_name} ...")

    except yaml.YAMLError as e:
        print(f"[X] YAML syntax error: {e}")
    except Exception as e:
        print(f"[X] Error reading template: {e}")
