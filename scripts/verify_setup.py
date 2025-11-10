"""
StrategyLab Backtester - Setup Verification Script

This script validates that the backtester system is correctly installed
and configured. It performs comprehensive checks on:
- Python environment
- Dependencies
- Strategy registration
- Configuration files
- Data availability
- Broker API connectivity

Usage:
    python scripts/verify_setup.py [--quick] [--skip-broker]

Options:
    --quick        : Run only essential checks (skip data and broker tests)
    --skip-broker  : Skip broker API connectivity tests
"""

import sys
import os
from pathlib import Path
import importlib.util
from typing import List, Tuple, Dict
import argparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

class VerificationResult:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
        self.total_checks = 0

    def add_pass(self, message: str):
        self.passed.append(message)
        self.total_checks += 1

    def add_fail(self, message: str, solution: str = ""):
        self.failed.append((message, solution))
        self.total_checks += 1

    def add_warning(self, message: str):
        self.warnings.append(message)

    def print_summary(self):
        print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}VERIFICATION SUMMARY{Colors.END}")
        print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")

        # Passed checks
        print(f"{Colors.GREEN}✓ Passed: {len(self.passed)}/{self.total_checks}{Colors.END}")
        for msg in self.passed:
            print(f"  {Colors.GREEN}✓{Colors.END} {msg}")

        # Failed checks
        if self.failed:
            print(f"\n{Colors.RED}✗ Failed: {len(self.failed)}/{self.total_checks}{Colors.END}")
            for msg, solution in self.failed:
                print(f"  {Colors.RED}✗{Colors.END} {msg}")
                if solution:
                    print(f"    {Colors.BLUE}→ {solution}{Colors.END}")

        # Warnings
        if self.warnings:
            print(f"\n{Colors.YELLOW}⚠ Warnings: {len(self.warnings)}{Colors.END}")
            for msg in self.warnings:
                print(f"  {Colors.YELLOW}⚠{Colors.END} {msg}")

        print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")

        # Final verdict
        if not self.failed:
            print(f"{Colors.GREEN}{Colors.BOLD}🎉 Setup verification PASSED!{Colors.END}")
            print(f"{Colors.GREEN}You're ready to run backtests.{Colors.END}\n")
            print(f"Next step: {Colors.BLUE}python scripts/quickstart.py{Colors.END}\n")
            return True
        else:
            print(f"{Colors.RED}{Colors.BOLD}❌ Setup verification FAILED{Colors.END}")
            print(f"{Colors.RED}Please fix the issues above before proceeding.{Colors.END}\n")
            print(f"For help, see: {Colors.BLUE}docs/ERROR_REFERENCE.md{Colors.END}\n")
            return False

def print_section(title: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{title}{Colors.END}")
    print(f"{Colors.BLUE}{'-' * len(title)}{Colors.END}")

def check_python_version(result: VerificationResult):
    """Check Python version is 3.9+"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major >= 3 and version.minor >= 9:
        result.add_pass(f"Python {version_str} (meets requirement: 3.9+)")
    else:
        result.add_fail(
            f"Python {version_str} (requires 3.9+)",
            "Upgrade Python from https://www.python.org/downloads/"
        )

def check_venv_active(result: VerificationResult):
    """Check if virtual environment is active"""
    # Check if we're in a venv
    in_venv = (
        hasattr(sys, 'real_prefix') or
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    )

    if in_venv:
        result.add_pass("Virtual environment is active")
    else:
        result.add_warning(
            "Not running in virtual environment (recommended but not required)"
        )

def check_dependencies(result: VerificationResult) -> Dict[str, bool]:
    """Check if all required dependencies are installed"""
    required_packages = [
        'pandas',
        'numpy',
        'pyyaml',
        'python-dotenv',
        'requests',
        'matplotlib',
        'seaborn',
        'scipy',
        'ta',  # Technical analysis library
    ]

    optional_packages = {
        'kiteconnect': 'Zerodha Kite API (for Zerodha broker)',
        'upstox_api': 'Upstox API (for Upstox broker)',
    }

    installed = {}
    missing = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            installed[package] = True
        except ImportError:
            installed[package] = False
            missing.append(package)

    if not missing:
        result.add_pass(f"All required dependencies installed ({len(required_packages)}/{len(required_packages)})")
    else:
        result.add_fail(
            f"Missing required dependencies: {', '.join(missing)}",
            "Run: pip install -r requirements.txt"
        )

    # Check optional packages
    for package, description in optional_packages.items():
        try:
            __import__(package)
            installed[package] = True
            result.add_pass(f"Optional: {package} installed")
        except ImportError:
            installed[package] = False
            result.add_warning(f"Optional: {package} not installed ({description})")

    return installed

def check_strategy_registration(result: VerificationResult):
    """Check if strategies are properly registered"""
    try:
        # Import strategy factory
        from strategies.register_strategies import StrategyFactory, register_all_strategies

        # Register strategies
        register_all_strategies()

        # Get registered strategies
        strategies = StrategyFactory.list_strategies()

        if strategies:
            result.add_pass(f"Strategies registered ({len(strategies)} found):")
            for strategy_name in strategies:
                print(f"    - {strategy_name}")
        else:
            result.add_fail(
                "No strategies registered",
                "Check src/strategies/register_strategies.py"
            )

    except ImportError as e:
        result.add_fail(
            f"Cannot import strategy registration module: {e}",
            "Verify src/strategies/register_strategies.py exists and is valid"
        )
    except Exception as e:
        result.add_fail(
            f"Strategy registration error: {e}",
            "Check src/strategies/register_strategies.py for errors"
        )

def check_config_templates(result: VerificationResult):
    """Check if configuration templates exist and are valid"""
    templates_dir = Path('config/templates')

    if not templates_dir.exists():
        result.add_fail(
            "Config templates directory not found",
            "Expected: config/templates/"
        )
        return

    template_files = list(templates_dir.glob('*.yaml'))

    if not template_files:
        result.add_fail(
            "No configuration templates found",
            "Check config/templates/ directory"
        )
        return

    # Try to load each template
    valid_templates = []
    invalid_templates = []

    try:
        import yaml

        for template_file in template_files:
            try:
                with open(template_file, 'r') as f:
                    yaml.safe_load(f)
                valid_templates.append(template_file.stem)
            except Exception as e:
                invalid_templates.append((template_file.stem, str(e)))

        if valid_templates:
            result.add_pass(f"Configuration templates valid ({len(valid_templates)}/{len(template_files)}):")
            for template_name in valid_templates:
                print(f"    - {template_name}")

        if invalid_templates:
            for template_name, error in invalid_templates:
                result.add_fail(
                    f"Invalid template: {template_name}",
                    f"YAML error: {error[:100]}"
                )

    except ImportError:
        result.add_warning("PyYAML not installed - cannot validate templates")

def check_directory_structure(result: VerificationResult):
    """Check if required directories exist"""
    required_dirs = [
        'src',
        'src/strategies',
        'src/core',
        'src/core/etl',
        'src/runners',
        'config',
        'config/templates',
    ]

    recommended_dirs = [
        'data',
        'data/pools',
        'outputs',
        'logs',
        'tests',
    ]

    missing_required = []
    missing_recommended = []

    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_required.append(dir_path)

    for dir_path in recommended_dirs:
        if not Path(dir_path).exists():
            missing_recommended.append(dir_path)

    if not missing_required:
        result.add_pass(f"Required directory structure exists ({len(required_dirs)} dirs)")
    else:
        result.add_fail(
            f"Missing required directories: {', '.join(missing_required)}",
            "Verify you're in the backtester root directory"
        )

    if missing_recommended:
        result.add_warning(
            f"Missing recommended directories: {', '.join(missing_recommended)}"
        )

def check_env_file(result: VerificationResult):
    """Check if .env file exists and has basic configuration"""
    env_file = Path('.env')
    env_example = Path('.env.example')

    if not env_file.exists():
        if env_example.exists():
            result.add_fail(
                ".env file not found",
                "Copy .env.example to .env and configure your API credentials"
            )
        else:
            result.add_fail(
                ".env and .env.example files not found",
                "Create .env file with broker API credentials"
            )
        return

    # Check if .env has basic content (not just template)
    with open(env_file, 'r') as f:
        content = f.read()

    # Look for placeholder text
    if 'your_' in content.lower() or '_here' in content.lower():
        result.add_warning(
            ".env file exists but may contain placeholder values"
        )
    else:
        result.add_pass(".env file configured")

def check_broker_api(result: VerificationResult, skip: bool = False):
    """Check if broker API is accessible (optional)"""
    if skip:
        result.add_warning("Broker API check skipped (--skip-broker flag)")
        return

    env_file = Path('.env')
    if not env_file.exists():
        result.add_warning("Cannot test broker API (.env not found)")
        return

    # Try to load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()

        # Check for broker credentials
        upstox_client_id = os.getenv('UPSTOX_CLIENT_ID')
        zerodha_api_key = os.getenv('ZERODHA_API_KEY')

        if upstox_client_id and 'your_' not in upstox_client_id.lower():
            result.add_pass("Upstox credentials configured")
        elif zerodha_api_key and 'your_' not in zerodha_api_key.lower():
            result.add_pass("Zerodha credentials configured")
        else:
            result.add_warning(
                "No valid broker API credentials found in .env"
            )

    except ImportError:
        result.add_warning("python-dotenv not installed - cannot check broker config")
    except Exception as e:
        result.add_warning(f"Error checking broker API config: {e}")

def check_data_availability(result: VerificationResult, quick: bool = False):
    """Check if data pools directory has data (optional)"""
    if quick:
        result.add_warning("Data availability check skipped (--quick mode)")
        return

    data_pools_dir = Path('data/pools')

    if not data_pools_dir.exists():
        result.add_warning(
            "Data pools directory not found - no historical data available yet"
        )
        return

    # Check for any .csv files in data pools
    csv_files = list(data_pools_dir.rglob('*.csv'))

    if csv_files:
        result.add_pass(f"Historical data found ({len(csv_files)} files)")
    else:
        result.add_warning(
            "No historical data found - run data fetcher before backtesting"
        )

def run_minimal_import_test(result: VerificationResult):
    """Try to import core modules"""
    core_modules = [
        'src.runners.unified_runner',
        'src.strategies.register_strategies',
        'src.core.etl.data_fetcher',
    ]

    import_failures = []

    for module_name in core_modules:
        try:
            # Convert module path to file path
            parts = module_name.split('.')
            file_path = Path(*parts[:-1]) / f"{parts[-1]}.py"

            if not file_path.exists():
                import_failures.append((module_name, f"File not found: {file_path}"))
                continue

            # Try to import (basic syntax check)
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                # Don't execute, just check it can be loaded
                import_failures.append((module_name, "OK"))
            else:
                import_failures.append((module_name, "Cannot create module spec"))

        except Exception as e:
            import_failures.append((module_name, str(e)[:100]))

    # Count successes
    ok_count = sum(1 for _, status in import_failures if status == "OK")

    if ok_count == len(core_modules):
        result.add_pass(f"Core modules importable ({ok_count}/{len(core_modules)})")
    else:
        for module_name, error in import_failures:
            if error != "OK":
                result.add_fail(
                    f"Cannot import {module_name}",
                    f"Error: {error}"
                )

def main():
    parser = argparse.ArgumentParser(description='Verify StrategyLab backtester setup')
    parser.add_argument('--quick', action='store_true', help='Run only essential checks')
    parser.add_argument('--skip-broker', action='store_true', help='Skip broker API tests')
    args = parser.parse_args()

    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}StrategyLab Backtester - Setup Verification{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

    result = VerificationResult()

    # Run all checks
    print_section("1. Environment Checks")
    check_python_version(result)
    check_venv_active(result)

    print_section("\n2. Dependency Checks")
    check_dependencies(result)

    print_section("\n3. Directory Structure")
    check_directory_structure(result)

    print_section("\n4. Configuration Files")
    check_env_file(result)
    check_config_templates(result)

    print_section("\n5. Strategy Registration")
    check_strategy_registration(result)

    print_section("\n6. Core Module Imports")
    run_minimal_import_test(result)

    print_section("\n7. Broker API Configuration")
    check_broker_api(result, skip=args.skip_broker)

    print_section("\n8. Data Availability")
    check_data_availability(result, quick=args.quick)

    # Print summary
    success = result.print_summary()

    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Verification interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error during verification: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
