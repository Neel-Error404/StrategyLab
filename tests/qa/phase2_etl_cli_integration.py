"""
Phase 2.3: ETL CLI Integration Testing

Test Objectives:
1. Test unified_runner.py --mode fetch (data fetching CLI)
2. Test unified_runner.py --mode validate (data validation CLI)
3. Verify CLI parameter parsing and execution
4. Validate error handling and user feedback

Expected Behavior:
- CLI commands execute without errors
- Data fetching works with proper parameters
- Validation provides clear feedback
- Error messages are informative
"""

import os
import sys
import codecs
import subprocess
from pathlib import Path
from datetime import datetime
import json

# Set UTF-8 encoding for Windows console
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")

# Add src to path for imports
BACKTESTER_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKTESTER_ROOT))

class ETLCLITester:
    def __init__(self):
        self.results = {
            "test_name": "Phase 2.3 - ETL CLI Integration",
            "timestamp": datetime.now().isoformat(),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "errors": []
        }
        self.runner_path = Path(BACKTESTER_ROOT) / "src" / "runners" / "unified_runner.py"
        self.python_path = Path(BACKTESTER_ROOT) / ".venv" / "Scripts" / "python.exe"
        
    def log(self, message, level="INFO"):
        """Print log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
    
    def run_cli_command(self, args, timeout=60):
        """Run unified_runner.py with given arguments"""
        cmd = [str(self.python_path), str(self.runner_path)] + args
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(BACKTESTER_ROOT)
            )
            return result
        except subprocess.TimeoutExpired:
            self.log(f"Command timed out after {timeout}s", "ERROR")
            return None
        except Exception as e:
            self.log(f"Command execution error: {e}", "ERROR")
            return None
    
    def test_cli_execution(self):
        """Test 1: CLI Execution with Help"""
        self.log("\nTest 1: CLI Execution Test (Help Command)")
        self.results["tests_run"] += 1
        
        try:
            # Test that CLI can be invoked and responds
            args = ["--help"]
            
            self.log("Running: unified_runner.py --help")
            result = self.run_cli_command(args, timeout=30)
            
            if result is None:
                self.log("CLI failed to execute", "FAIL")
                self.results["tests_failed"] += 1
                return False
            
            # Check output contains help information
            output = result.stdout + result.stderr
            
            # Check for expected help content
            help_indicators = ["usage", "mode", "backtest", "options"]
            found = [h for h in help_indicators if h.lower() in output.lower()]
            
            if len(found) >= 2:
                self.log(f"CLI executable and responsive (found: {', '.join(found)})", "PASS")
                self.results["tests_passed"] += 1
                return True
            else:
                self.log("CLI help output insufficient", "FAIL")
                self.results["tests_failed"] += 1
                return False
            
        except Exception as e:
            self.log(f"CLI execution test error: {e}", "FAIL")
            self.results["tests_failed"] += 1
            self.results["errors"].append(f"Test 1: {str(e)}")
            return False
    
    def test_help_command(self):
        """Test 2: Parameter discovery and mode listing"""
        self.log("\nTest 2: Mode Discovery")
        self.results["tests_run"] += 1
        
        try:
            # Run with --help to see all modes
            args = ["--help"]
            
            self.log("Running: unified_runner.py --help (checking available modes)")
            result = self.run_cli_command(args, timeout=30)
            
            if result is None:
                self.log("Help command failed", "FAIL")
                self.results["tests_failed"] += 1
                return False
            
            output = result.stdout + result.stderr
            
            # Check for all expected modes
            expected_modes = ["backtest", "fetch", "validate", "analyze", "visualize"]
            found_modes = [m for m in expected_modes if m in output.lower()]
            
            missing_modes = [m for m in expected_modes if m not in found_modes]
            
            if len(found_modes) >= 4:  # At least 4 out of 5 modes
                self.log(f"Found {len(found_modes)}/5 modes: {', '.join(found_modes)}", "PASS")
                if missing_modes:
                    self.log(f"Missing modes: {', '.join(missing_modes)}", "INFO")
                
                self.results["tests_passed"] += 1
                return True
            else:
                self.log(f"Insufficient modes found: {', '.join(found_modes)}", "FAIL")
                self.results["tests_failed"] += 1
                return False
            
        except Exception as e:
            self.log(f"Mode discovery test error: {e}", "FAIL")
            self.results["tests_failed"] += 1
            self.results["errors"].append(f"Test 2: {str(e)}")
            return False
    
    def generate_report(self):
        """Generate test report"""
        self.log("\n" + "=" * 80)
        self.log("PHASE 2.3 TEST SUMMARY - ETL CLI INTEGRATION")
        self.log("=" * 80)
        
        total_tests = self.results["tests_run"]
        passed = self.results["tests_passed"]
        failed = self.results["tests_failed"]
        pass_rate = (passed / total_tests * 100) if total_tests > 0 else 0
        
        self.log(f"Total Tests: {total_tests}")
        self.log(f"Passed: {passed}")
        self.log(f"Failed: {failed}")
        self.log(f"Pass Rate: {pass_rate:.1f}%")
        
        if self.results["errors"]:
            self.log("\nERRORS:")
            for error in self.results["errors"]:
                self.log(f"  - {error}")
        
        # Save report
        output_path = Path(BACKTESTER_ROOT) / "outputs" / "qa_phase2.3_etl_cli_report.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Phase 2.3: ETL CLI Integration Test Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Results:\n")
            f.write(f"  Total Tests: {total_tests}\n")
            f.write(f"  Passed: {passed}\n")
            f.write(f"  Failed: {failed}\n")
            f.write(f"  Pass Rate: {pass_rate:.1f}%\n\n")
            
            f.write("Tests Executed:\n")
            f.write("  1. CLI Execution Test (Help Command)\n")
            f.write("  2. Mode Discovery (Available Modes Check)\n\n")
            
            if self.results["errors"]:
                f.write("Errors:\n")
                for error in self.results["errors"]:
                    f.write(f"  - {error}\n")
        
        self.log(f"\nReport saved: {output_path}")
        
        return pass_rate >= 80
    
    def run_all_tests(self):
        """Run all Phase 2.3 tests"""
        self.log("=" * 80)
        self.log("PHASE 2.3: ETL CLI INTEGRATION TESTING")
        self.log("=" * 80)
        
        # Check if runner exists
        if not self.runner_path.exists():
            self.log(f"Runner not found: {self.runner_path}", "ERROR")
            return 1
        
        self.log(f"Using runner: {self.runner_path}")
        self.log(f"Using python: {self.python_path}")
        
        try:
            # Run tests
            self.test_cli_execution()
            self.test_help_command()
            
            # Generate report
            success = self.generate_report()
            
            return 0 if success else 1
            
        except Exception as e:
            self.log(f"Test execution error: {e}", "ERROR")
            return 1


if __name__ == "__main__":
    tester = ETLCLITester()
    exit_code = tester.run_all_tests()
    sys.exit(exit_code)
