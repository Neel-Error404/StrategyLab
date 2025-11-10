"""
StrategyLab Backtester - Automated Setup Script

This script automates the installation and configuration process for the
backtester system. It handles virtual environment creation, dependency
installation, and initial configuration.

Usage:
    python setup.py [--auto] [--skip-venv] [--skip-deps]

Options:
    --auto      : Run in non-interactive mode (use defaults)
    --skip-venv : Skip virtual environment creation (use if already in venv)
    --skip-deps : Skip dependency installation (use if already installed)
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
from typing import Tuple, Optional

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

# Check if running on Windows (PowerShell encoding issues)
IS_WINDOWS = platform.system() == 'Windows'

def print_header(message: str):
    """Print a formatted header message"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{message:^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_success(message: str):
    """Print a success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")

def print_warning(message: str):
    """Print a warning message"""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")

def print_error(message: str):
    """Print an error message"""
    print(f"{Colors.RED}✗ {message}{Colors.END}")

def print_info(message: str):
    """Print an info message"""
    print(f"{Colors.BLUE}ℹ {message}{Colors.END}")

def check_python_version() -> Tuple[bool, str]:
    """Check if Python version meets minimum requirements (3.9+)"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major < 3 or (version.major == 3 and version.minor < 9):
        return False, version_str
    return True, version_str

def check_git_installed() -> bool:
    """Check if git is installed"""
    try:
        subprocess.run(['git', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def create_virtual_environment() -> bool:
    """Create a virtual environment in .venv directory"""
    venv_path = Path('.venv')

    if venv_path.exists():
        print_warning(f"Virtual environment already exists at {venv_path}")
        response = input("Do you want to recreate it? (y/N): ").strip().lower()
        if response == 'y':
            print_info(f"Removing existing virtual environment...")
            shutil.rmtree(venv_path)
        else:
            print_info("Using existing virtual environment")
            return True

    try:
        print_info(f"Creating virtual environment in {venv_path}...")
        subprocess.run([sys.executable, '-m', 'venv', str(venv_path)], check=True)
        print_success("Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to create virtual environment: {e}")
        return False

def get_venv_python() -> Optional[Path]:
    """Get the path to the Python executable in the virtual environment"""
    venv_path = Path('.venv')

    if IS_WINDOWS:
        python_path = venv_path / 'Scripts' / 'python.exe'
    else:
        python_path = venv_path / 'bin' / 'python'

    if python_path.exists():
        return python_path
    return None

def install_dependencies(python_path: Path) -> bool:
    """Install dependencies from requirements.txt"""
    requirements_file = Path('requirements.txt')

    if not requirements_file.exists():
        print_error("requirements.txt not found!")
        print_info("Please ensure you're running this from the backtester directory")
        return False

    try:
        print_info("Installing dependencies from requirements.txt...")
        print_info("This may take several minutes...")

        # Upgrade pip first
        subprocess.run(
            [str(python_path), '-m', 'pip', 'install', '--upgrade', 'pip'],
            check=True,
            capture_output=True
        )

        # Install requirements
        result = subprocess.run(
            [str(python_path), '-m', 'pip', 'install', '-r', str(requirements_file)],
            check=True,
            capture_output=True,
            text=True
        )

        print_success("All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        if e.stderr:
            print_error(f"Error details: {e.stderr[:500]}")
        return False

def create_env_file() -> bool:
    """Create .env file from .env.example template"""
    env_example = Path('.env.example')
    env_file = Path('.env')

    if not env_example.exists():
        print_warning(".env.example not found - skipping .env creation")
        print_info("You'll need to create .env manually for broker API access")
        return True

    if env_file.exists():
        print_warning(".env file already exists")
        response = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if response != 'y':
            print_info("Keeping existing .env file")
            return True

    try:
        # Copy template to .env
        shutil.copy(env_example, env_file)
        print_success(".env file created from template")
        print_info("Please edit .env and add your broker API credentials")
        print_info("See docs/BROKER_SETUP.md for detailed instructions")
        return True
    except Exception as e:
        print_error(f"Failed to create .env file: {e}")
        return False

def create_required_directories():
    """Create required directory structure"""
    directories = [
        'data/pools',
        'outputs',
        'logs',
        'config/access_tokens'
    ]

    for dir_path in directories:
        path = Path(dir_path)
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                print_success(f"Created directory: {dir_path}")
            except Exception as e:
                print_warning(f"Could not create directory {dir_path}: {e}")

def display_next_steps():
    """Display next steps for the user"""
    print_header("Setup Complete!")

    print(f"{Colors.BOLD}Next Steps:{Colors.END}\n")

    print(f"{Colors.GREEN}1. Activate the virtual environment:{Colors.END}")
    if IS_WINDOWS:
        print(f"   {Colors.BLUE}.\.venv\Scripts\Activate.ps1{Colors.END}  (PowerShell)")
        print(f"   {Colors.BLUE}.\.venv\Scripts\activate.bat{Colors.END}    (CMD)")
    else:
        print(f"   {Colors.BLUE}source .venv/bin/activate{Colors.END}")

    print(f"\n{Colors.GREEN}2. Configure broker API credentials:{Colors.END}")
    print(f"   Edit the {Colors.BLUE}.env{Colors.END} file with your API credentials")
    print(f"   See {Colors.BLUE}docs/BROKER_SETUP.md{Colors.END} for detailed instructions")

    print(f"\n{Colors.GREEN}3. Verify installation:{Colors.END}")
    print(f"   {Colors.BLUE}python scripts/verify_setup.py{Colors.END}")

    print(f"\n{Colors.GREEN}4. Run your first backtest:{Colors.END}")
    print(f"   {Colors.BLUE}python scripts/quickstart.py{Colors.END}  (interactive)")
    print(f"   OR")
    print(f"   {Colors.BLUE}python src/runners/unified_runner.py --mode backtest --strategies open_source_baseline --template conservative --date-ranges 2024-01-01_to_2024-01-31 --tickers RELIANCE{Colors.END}")

    print(f"\n{Colors.BOLD}Documentation:{Colors.END}")
    print(f"   Quick Start:   {Colors.BLUE}QUICKSTART.md{Colors.END}")
    print(f"   Setup Guide:   {Colors.BLUE}docs/SETUP_GUIDE.md{Colors.END}")
    print(f"   CLI Reference: {Colors.BLUE}docs/CLI_REFERENCE.md{Colors.END}")
    print(f"   Troubleshoot:  {Colors.BLUE}docs/ERROR_REFERENCE.md{Colors.END}")

    print(f"\n{Colors.BOLD}{Colors.GREEN}Happy Backtesting! 🚀{Colors.END}\n")

def main():
    """Main setup process"""
    # Parse command line arguments
    auto_mode = '--auto' in sys.argv
    skip_venv = '--skip-venv' in sys.argv
    skip_deps = '--skip-deps' in sys.argv

    print_header("StrategyLab Backtester - Automated Setup")

    # Step 1: Check Python version
    print_info("Step 1/6: Checking Python version...")
    is_valid, version = check_python_version()
    if is_valid:
        print_success(f"Python {version} detected (meets requirement: 3.9+)")
    else:
        print_error(f"Python {version} detected (requires 3.9 or higher)")
        print_info("Please upgrade Python from https://www.python.org/downloads/")
        sys.exit(1)

    # Step 2: Check Git (optional but recommended)
    print_info("\nStep 2/6: Checking for Git...")
    if check_git_installed():
        print_success("Git is installed")
    else:
        print_warning("Git not found (optional but recommended for version control)")

    # Step 3: Create virtual environment
    if not skip_venv:
        print_info("\nStep 3/6: Setting up virtual environment...")
        if not create_virtual_environment():
            print_error("Virtual environment setup failed")
            sys.exit(1)
    else:
        print_info("\nStep 3/6: Skipping virtual environment creation (--skip-venv)")

    # Get venv Python path
    venv_python = get_venv_python()
    if venv_python is None:
        print_warning("Virtual environment Python not found, using system Python")
        venv_python = Path(sys.executable)

    # Step 4: Install dependencies
    if not skip_deps:
        print_info("\nStep 4/6: Installing dependencies...")
        if not install_dependencies(venv_python):
            print_error("Dependency installation failed")
            print_info("You can try running: pip install -r requirements.txt manually")
            sys.exit(1)
    else:
        print_info("\nStep 4/6: Skipping dependency installation (--skip-deps)")

    # Step 5: Create .env file
    print_info("\nStep 5/6: Creating configuration files...")
    create_env_file()

    # Step 6: Create required directories
    print_info("\nStep 6/6: Creating required directories...")
    create_required_directories()

    # Display next steps
    display_next_steps()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print_warning("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"\n\nUnexpected error during setup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
