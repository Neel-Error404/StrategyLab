"""
Phase 1.3: Known Truth Validation - P&L Verification

Test Objectives:
1. Extract sample trades from Phase 1.2 results
2. Load corresponding price data
3. Manually calculate expected P&L
4. Compare with system output
5. Investigate root cause of zero P&L issue
6. Verify P&L calculation logic

Expected Behavior:
- P&L = (Exit Price - Entry Price) * Quantity - Transaction Costs
- Win/Loss classification based on P&L sign
- Accurate profit percentage calculation

Root Cause Investigation:
- Check if P&L is in different columns
- Verify transaction cost application
- Check risk manager impact
- Validate price data alignment
"""

import os
import sys
import codecs
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json
import pytest

pytest.skip(
    "QA dataset not bundled with OSS release; skipping Phase 1.3 QA suite.",
    allow_module_level=True,
)

# Set UTF-8 encoding for Windows console
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")

# Add src to path for imports
BACKTESTER_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKTESTER_ROOT))

class KnownTruthValidator:
    def __init__(self):
        self.results = {
            "test_name": "Phase 1.3 - Known Truth Validation",
            "timestamp": datetime.now().isoformat(),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "errors": [],
            "findings": []
        }
        # Use Phase 1.2 output directory
        self.phase12_output = Path(BACKTESTER_ROOT) / "outputs" / "20251017_090125" / "open_source_baseline" / "2022-01-01_to_2025-08-31_extras"
        self.data_pool = Path(BACKTESTER_ROOT) / "data" / "pools" / "2022-01-01_to_2025-08-31_extras"
        
    def log(self, message, level="INFO"):
        """Print log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
    
    def test_trades_file_structure(self):
        """Test 1: Analyze CSV structure and identify P&L columns"""
        self.log("Test 1: Trades File Structure Analysis")
        self.results["tests_run"] += 1
        
        try:
            # Load 360ONE strategy trades
            trades_file = self.phase12_output / "data" / "strategy_trades" / "360ONE_StrategyTrades_2022-01-01_to_2025-08-31_extras.csv"
            
            if not trades_file.exists():
                self.log(f"Trade file not found: {trades_file}", "FAIL")
                self.results["tests_failed"] += 1
                return False
            
            df = pd.read_csv(trades_file)
            self.log(f"Loaded {len(df)} trades from {trades_file.name}")
            
            # Analyze all columns
            self.log(f"\nColumn Analysis ({len(df.columns)} columns):")
            for col in df.columns:
                non_null = df[col].notna().sum()
                unique = df[col].nunique()
                self.log(f"  {col}: {non_null} non-null, {unique} unique values")
                
                # Show sample values for potential P&L columns
                if any(keyword in col.lower() for keyword in ['pnl', 'profit', 'currency', 'return']):
                    sample = df[col].dropna().head(5).tolist()
                    self.log(f"    Sample values: {sample}")
            
            # Check specific P&L-related columns
            pnl_columns = [col for col in df.columns if 'pnl' in col.lower() or 'profit' in col.lower()]
            self.log(f"\nP&L-related columns found: {pnl_columns}")
            
            # Analyze PnL column
            if 'PnL' in df.columns:
                pnl_stats = df['PnL'].describe()
                self.log(f"\nPnL Statistics:")
                self.log(f"  Count: {pnl_stats['count']}")
                self.log(f"  Mean: Rs{pnl_stats['mean']:.2f}")
                self.log(f"  Std: Rs{pnl_stats['std']:.2f}")
                self.log(f"  Min: Rs{pnl_stats['min']:.2f}")
                self.log(f"  Max: Rs{pnl_stats['max']:.2f}")
                self.log(f"  Non-zero count: {(df['PnL'] != 0).sum()}")
            
            # Analyze Profit (Currency) column if exists
            if 'Profit (Currency)' in df.columns:
                profit_stats = df['Profit (Currency)'].describe()
                self.log(f"\nProfit (Currency) Statistics:")
                self.log(f"  Count: {profit_stats['count']}")
                self.log(f"  Mean: Rs{profit_stats['mean']:.2f}")
                self.log(f"  Std: Rs{profit_stats['std']:.2f}")
                self.log(f"  Min: Rs{profit_stats['min']:.2f}")
                self.log(f"  Max: Rs{profit_stats['max']:.2f}")
                self.log(f"  Non-zero count: {(df['Profit (Currency)'] != 0).sum()}")
            
            self.log("CSV structure analysis completed", "PASS")
            self.results["tests_passed"] += 1
            self.df_trades = df
            return True
            
        except Exception as e:
            self.log(f"Structure analysis error: {e}", "FAIL")
            self.results["tests_failed"] += 1
            self.results["errors"].append(f"Test 1: {str(e)}")
            return False
    
    def test_manual_pnl_calculation(self):
        """Test 2: Calculate P&L manually for sample trades"""
        self.log("\nTest 2: Manual P&L Calculation for Sample Trades")
        self.results["tests_run"] += 1
        
        if not hasattr(self, 'df_trades'):
            self.log("No trades data available - skipping test", "FAIL")
            self.results["tests_failed"] += 1
            return False
        
        try:
            # Select first 10 trades for manual verification
            sample_trades = self.df_trades.head(10).copy()
            
            self.log(f"\nAnalyzing {len(sample_trades)} sample trades:")
            
            discrepancies = []
            for idx, trade in sample_trades.iterrows():
                entry_price = trade.get('Entry Price', 0)
                exit_price = trade.get('Exit Price', 0)
                quantity = 1  # Assuming 1 share per trade (baseline configuration)
                
                # Manual calculation
                manual_pnl = (exit_price - entry_price) * quantity
                system_pnl = trade.get('PnL', 0)
                profit_currency = trade.get('Profit (Currency)', 0)
                
                self.log(f"\nTrade {idx + 1}:")
                self.log(f"  Entry: Rs{entry_price:.2f}")
                self.log(f"  Exit: Rs{exit_price:.2f}")
                self.log(f"  Manual P&L: Rs{manual_pnl:.2f}")
                self.log(f"  System PnL: Rs{system_pnl:.2f}")
                self.log(f"  Profit (Currency): Rs{profit_currency:.2f}")
                
                # Check for discrepancy
                if abs(manual_pnl - system_pnl) > 0.01:
                    discrepancy = {
                        'trade_num': idx + 1,
                        'manual': manual_pnl,
                        'system': system_pnl,
                        'difference': manual_pnl - system_pnl
                    }
                    discrepancies.append(discrepancy)
                    self.log(f"  WARNING: Discrepancy detected! Diff: Rs{discrepancy['difference']:.2f}", "WARN")
            
            if discrepancies:
                self.log(f"\nFound {len(discrepancies)} P&L discrepancies", "WARN")
                self.results["findings"].append(f"P&L discrepancies in {len(discrepancies)}/10 sample trades")
            else:
                self.log("\nAll sample trades match manual calculation", "PASS")
            
            self.results["tests_passed"] += 1
            return True
            
        except Exception as e:
            self.log(f"Manual calculation error: {e}", "FAIL")
            self.results["tests_failed"] += 1
            self.results["errors"].append(f"Test 2: {str(e)}")
            return False
    
    def test_root_cause_analysis(self):
        """Test 3: Root cause analysis of zero P&L issue"""
        self.log("\nTest 3: Root Cause Analysis")
        self.results["tests_run"] += 1
        
        if not hasattr(self, 'df_trades'):
            self.log("No trades data available - skipping test", "FAIL")
            self.results["tests_failed"] += 1
            return False
        
        try:
            findings = []
            
            # Hypothesis 1: Check if all P&L values are truly zero
            if 'PnL' in self.df_trades.columns:
                zero_count = (self.df_trades['PnL'] == 0).sum()
                total_count = len(self.df_trades)
                self.log(f"Hypothesis 1: All PnL values are zero")
                self.log(f"  Zero P&L trades: {zero_count}/{total_count} ({zero_count/total_count*100:.1f}%)")
                
                if zero_count == total_count:
                    findings.append("ALL P&L values are exactly Rs0.00")
                    self.log("  CONFIRMED: 100% of trades have zero P&L", "WARN")
                else:
                    self.log(f"  {total_count - zero_count} trades have non-zero P&L", "INFO")
            
            # Hypothesis 2: Check if Profit (Currency) has values
            if 'Profit (Currency)' in self.df_trades.columns:
                zero_profit = (self.df_trades['Profit (Currency)'] == 0).sum()
                total_count = len(self.df_trades)
                self.log(f"\nHypothesis 2: Profit (Currency) has actual values")
                self.log(f"  Zero Profit trades: {zero_profit}/{total_count} ({zero_profit/total_count*100:.1f}%)")
                
                if zero_profit < total_count:
                    non_zero = self.df_trades[self.df_trades['Profit (Currency)'] != 0]['Profit (Currency)']
                    self.log(f"  Non-zero Profit count: {len(non_zero)}")
                    self.log(f"  Non-zero Profit range: Rs{non_zero.min():.2f} to Rs{non_zero.max():.2f}")
                    findings.append(f"Profit (Currency) has {len(non_zero)} non-zero values")
            
            # Hypothesis 3: Check entry/exit price validity
            self.log(f"\nHypothesis 3: Entry and Exit prices are valid")
            entry_valid = (self.df_trades['Entry Price'] > 0).sum()
            exit_valid = (self.df_trades['Exit Price'] > 0).sum()
            self.log(f"  Valid Entry Prices: {entry_valid}/{len(self.df_trades)}")
            self.log(f"  Valid Exit Prices: {exit_valid}/{len(self.df_trades)}")
            
            if entry_valid > 0 and exit_valid > 0:
                # Calculate expected P&L
                self.df_trades['Expected_PnL'] = self.df_trades['Exit Price'] - self.df_trades['Entry Price']
                non_zero_expected = (self.df_trades['Expected_PnL'] != 0).sum()
                self.log(f"  Expected non-zero P&L: {non_zero_expected}/{len(self.df_trades)}")
                
                if non_zero_expected > 0:
                    findings.append(f"Prices are valid, should generate {non_zero_expected} non-zero P&L trades")
                    self.log("  FINDING: Prices suggest P&L should be calculated!", "WARN")
            
            # Hypothesis 4: Check if this is Risk-Approved vs Strategy file issue
            risk_file = self.phase12_output / "data" / "risk_approved_trades" / "360ONE_RiskApprovedTrades_2022-01-01_to_2025-08-31_extras.csv"
            if risk_file.exists():
                self.log(f"\nHypothesis 4: P&L might be in Risk-Approved file")
                df_risk = pd.read_csv(risk_file)
                self.log(f"  Risk-Approved trades: {len(df_risk)}")
                
                if 'PnL' in df_risk.columns:
                    risk_zero = (df_risk['PnL'] == 0).sum()
                    self.log(f"  Zero P&L in Risk file: {risk_zero}/{len(df_risk)}")
                    
                    if risk_zero < len(df_risk):
                        findings.append(f"Risk-Approved file has {len(df_risk) - risk_zero} non-zero P&L trades")
                        self.log("  FINDING: Risk file might have actual P&L!", "WARN")
            
            # Summary
            self.log(f"\n=== ROOT CAUSE SUMMARY ===")
            for i, finding in enumerate(findings, 1):
                self.log(f"{i}. {finding}")
            
            if not findings:
                self.log("No clear root cause identified - needs deeper investigation")
            
            self.results["tests_passed"] += 1
            self.results["findings"].extend(findings)
            return True
            
        except Exception as e:
            self.log(f"Root cause analysis error: {e}", "FAIL")
            self.results["tests_failed"] += 1
            self.results["errors"].append(f"Test 3: {str(e)}")
            return False
    
    def generate_report(self):
        """Generate test report"""
        self.log("\n" + "=" * 80)
        self.log("PHASE 1.3 TEST SUMMARY - KNOWN TRUTH VALIDATION")
        self.log("=" * 80)
        
        total_tests = self.results["tests_run"]
        passed = self.results["tests_passed"]
        failed = self.results["tests_failed"]
        pass_rate = (passed / total_tests * 100) if total_tests > 0 else 0
        
        self.log(f"Total Tests: {total_tests}")
        self.log(f"Passed: {passed}")
        self.log(f"Failed: {failed}")
        self.log(f"Pass Rate: {pass_rate:.1f}%")
        
        if self.results["findings"]:
            self.log("\nKEY FINDINGS:")
            for i, finding in enumerate(self.results["findings"], 1):
                self.log(f"  {i}. {finding}")
        
        if self.results["errors"]:
            self.log("\nERRORS:")
            for error in self.results["errors"]:
                self.log(f"  - {error}")
        
        # Save report
        output_path = Path(BACKTESTER_ROOT) / "outputs" / "qa_phase1.3_known_truth_report.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Phase 1.3: Known Truth Validation Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Test Configuration:\n")
            f.write(f"  Data Source: Phase 1.2 outputs\n")
            f.write(f"  Ticker: 360ONE\n")
            f.write(f"  Strategy: Open Source Baseline\n")
            f.write(f"  Date Range: 2022-01-01 to 2025-08-31\n\n")
            f.write(f"Results:\n")
            f.write(f"  Total Tests: {total_tests}\n")
            f.write(f"  Passed: {passed}\n")
            f.write(f"  Failed: {failed}\n")
            f.write(f"  Pass Rate: {pass_rate:.1f}%\n\n")
            
            if self.results["findings"]:
                f.write("Key Findings:\n")
                for i, finding in enumerate(self.results["findings"], 1):
                    f.write(f"  {i}. {finding}\n")
                f.write("\n")
            
            if self.results["errors"]:
                f.write("Errors:\n")
                for error in self.results["errors"]:
                    f.write(f"  - {error}\n")
        
        self.log(f"\nReport saved: {output_path}")
        
        return pass_rate >= 80
    
    def run_all_tests(self):
        """Run all Phase 1.3 tests"""
        self.log("=" * 80)
        self.log("PHASE 1.3: KNOWN TRUTH VALIDATION - P&L INVESTIGATION")
        self.log("=" * 80)
        self.log(f"Data Source: Phase 1.2 outputs")
        self.log(f"Objective: Investigate zero P&L issue and verify calculations")
        self.log("=" * 80)
        
        # Run tests in sequence
        test1_ok = self.test_trades_file_structure()
        if test1_ok:
            self.test_manual_pnl_calculation()
            self.test_root_cause_analysis()
        else:
            self.log("\nSkipping remaining tests due to structure analysis failure")
        
        # Generate report
        success = self.generate_report()
        
        return 0 if success else 1


if __name__ == "__main__":
    validator = KnownTruthValidator()
    exit_code = validator.run_all_tests()
    sys.exit(exit_code)
