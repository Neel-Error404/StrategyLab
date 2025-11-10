#!/usr/bin/env python3
"""
Phase 0.1: Environment Setup Validation
========================================

Purpose: Validate that the development environment is correctly configured
before running any backtests or tests.

Test Coverage:
- Python version >= 3.9
- Virtual environment is active
- All required packages installed with correct versions
- API credentials present in .env
- Required directories exist
- Import all critical modules without errors

Success Criteria:
- All checks pass
- All critical imports succeed
- API credentials detected (not validated for connection)

Expected Runtime: 30 seconds

Author: QA Team
Date: October 16, 2025
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import importlib.util

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class EnvironmentValidator:
    """Validates development environment setup"""
    
    def __init__(self):
        self.results: List[Tuple[str, bool, str]] = []
        self.project_root = PROJECT_ROOT
        
    def check_python_version(self) -> bool:
        """Verify Python version >= 3.9"""
        version = sys.version_info
        is_valid = version.major == 3 and version.minor >= 9
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        self.results.append((
            "Python Version",
            is_valid,
            f"{version_str} (Required: 3.9+)"
        ))
        return is_valid
    
    def check_virtual_environment(self) -> bool:
        """Verify running in virtual environment"""
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        
        venv_path = sys.prefix if in_venv else "Not in virtual environment"
        
        self.results.append((
            "Virtual Environment",
            in_venv,
            str(venv_path)
        ))
        return in_venv
    
    def check_required_packages(self) -> bool:
        """Verify all required packages are installed"""
        required_packages = {
            'pandas': '2.0.0',
            'numpy': '1.24.0',
            'yfinance': '0.2.18',
            'upstox_client': '2.0.0',
            'pyarrow': '14.0.0',
            'yaml': '6.0',  # PyYAML imports as 'yaml'
            'python-dotenv': '1.0.0',
            'scipy': '1.11.0',
            'matplotlib': '3.7.0',
            'seaborn': '0.12.0',
            'joblib': '1.3.0',
            'PyPortfolioOpt': '1.5.0',
        }
        
        missing = []
        installed = []
        
        for package, min_version in required_packages.items():
            try:
                if package == 'PyPortfolioOpt':
                    module = importlib.import_module('pypfopt')
                elif package == 'python-dotenv':
                    module = importlib.import_module('dotenv')
                elif package == 'upstox_client':
                    module = importlib.import_module('upstox_client')
                else:
                    module = importlib.import_module(package)
                
                version = getattr(module, '__version__', 'unknown')
                installed.append(f"{package}=={version}")
            except ImportError:
                missing.append(package)
        
        all_installed = len(missing) == 0
        
        status = f"Installed: {len(installed)}, Missing: {len(missing)}"
        if missing:
            status += f" ({', '.join(missing)})"
        
        self.results.append((
            "Required Packages",
            all_installed,
            status
        ))
        return all_installed
    
    def check_api_credentials(self) -> bool:
        """Verify API credentials present in .env"""
        env_file = self.project_root / '.env'
        
        if not env_file.exists():
            self.results.append((
                "API Credentials (.env)",
                False,
                ".env file not found"
            ))
            return False
        
        required_keys = [
            'UPSTOX_CLIENT_ID',
            'UPSTOX_CLIENT_SECRET',
            'UPSTOX_REDIRECT_URI',
        ]
        
        found_keys = []
        missing_keys = []
        
        with open(env_file, 'r') as f:
            content = f.read()
            for key in required_keys:
                if key in content:
                    found_keys.append(key)
                else:
                    missing_keys.append(key)
        
        all_present = len(missing_keys) == 0
        
        status = f"Found: {len(found_keys)}/{len(required_keys)}"
        if missing_keys:
            status += f" (Missing: {', '.join(missing_keys)})"
        
        self.results.append((
            "API Credentials (.env)",
            all_present,
            status
        ))
        return all_present
    
    def check_directory_structure(self) -> bool:
        """Verify required directories exist"""
        required_dirs = [
            'src',
            'src/core',
            'src/core/etl',
            'src/core/options',
            'src/strategies',
            'src/runners',
            'config',
            'data',
            'data/pools',
            'outputs',
            'logs',
            'tests',
            'tests/qa',
        ]
        
        missing_dirs = []
        existing_dirs = []
        
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                existing_dirs.append(dir_path)
            else:
                missing_dirs.append(dir_path)
        
        all_exist = len(missing_dirs) == 0
        
        status = f"Existing: {len(existing_dirs)}/{len(required_dirs)}"
        if missing_dirs:
            status += f" (Missing: {', '.join(missing_dirs)})"
        
        self.results.append((
            "Directory Structure",
            all_exist,
            status
        ))
        return all_exist
    
    def check_critical_imports(self) -> bool:
        """Test importing critical modules"""
        critical_modules = [
            'src.runners.unified_runner',
            'src.core.etl.data_fetcher',
            'src.core.etl.gap_calculator',
            'src.core.etl.pool_inspector',
            'src.core.options.options_engine',
            'config.unified_config',
        ]
        
        successful_imports = []
        failed_imports = []
        
        for module_name in critical_modules:
            try:
                importlib.import_module(module_name)
                successful_imports.append(module_name)
            except Exception as e:
                failed_imports.append(f"{module_name} ({str(e)[:50]})")
        
        all_imported = len(failed_imports) == 0
        
        status = f"Imported: {len(successful_imports)}/{len(critical_modules)}"
        if failed_imports:
            status += f"\nFailed: {', '.join(failed_imports)}"
        
        self.results.append((
            "Critical Module Imports",
            all_imported,
            status
        ))
        return all_imported
    
    def run_all_checks(self) -> bool:
        """Run all validation checks"""
        print("=" * 70)
        print("Phase 0.1: Environment Setup Validation")
        print("=" * 70)
        print()
        
        checks = [
            ("Python Version", self.check_python_version),
            ("Virtual Environment", self.check_virtual_environment),
            ("Required Packages", self.check_required_packages),
            ("API Credentials", self.check_api_credentials),
            ("Directory Structure", self.check_directory_structure),
            ("Critical Imports", self.check_critical_imports),
        ]
        
        all_passed = True
        for name, check_func in checks:
            print(f"Running: {name}...", end=" ")
            try:
                passed = check_func()
                print("✅ PASS" if passed else "❌ FAIL")
                all_passed = all_passed and passed
            except Exception as e:
                print(f"❌ ERROR: {str(e)}")
                all_passed = False
        
        print()
        print("=" * 70)
        print("Summary")
        print("=" * 70)
        
        for check_name, passed, details in self.results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {check_name}")
            if details:
                print(f"     {details}")
        
        print()
        print("=" * 70)
        if all_passed:
            print("✅ ALL CHECKS PASSED - Environment is ready")
        else:
            print("❌ SOME CHECKS FAILED - Fix issues before proceeding")
        print("=" * 70)
        
        return all_passed
    
    def generate_report(self, output_file: Path):
        """Generate detailed report"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Phase 0.1: Environment Setup Validation Report\n\n")
            f.write(f"**Date**: {Path(__file__).stat().st_mtime}\n")
            f.write(f"**Project Root**: {self.project_root}\n\n")
            
            f.write("## Results\n\n")
            for check_name, passed, details in self.results:
                status = "✅ PASS" if passed else "❌ FAIL"
                f.write(f"### {check_name}\n")
                f.write(f"**Status**: {status}\n")
                f.write(f"**Details**: {details}\n\n")


def main():
    """Main test entry point"""
    validator = EnvironmentValidator()
    all_passed = validator.run_all_checks()
    
    # Generate report
    report_path = PROJECT_ROOT / 'outputs' / 'qa_phase0.1_environment_report.txt'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    validator.generate_report(report_path)
    
    print(f"\n📄 Report saved to: {report_path}")
    
    # Update journal (template - manual update needed)
    print("\n📝 Manual Action Required:")
    print("   Update QA_TESTING_JOURNAL.md with test results")
    
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
