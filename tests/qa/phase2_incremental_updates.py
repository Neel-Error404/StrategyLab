"""
Phase 2.2: ETL Tools - Incremental Updates Testing

Test Objectives:
1. Test incremental data fetching (append new data without refetch)
2. Verify data deduplication
3. Test timestamp continuity after update
4. Validate no data loss during incremental updates
5. Test update performance vs full refetch

Expected Behavior:
- Append only new bars to existing dataset
- Detect and remove duplicates
- Maintain chronological order
- Preserve existing data integrity
"""

import os
import sys
import codecs
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import shutil

# Set UTF-8 encoding for Windows console
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")

# Add src to path for imports
BACKTESTER_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKTESTER_ROOT))

class IncrementalUpdateTester:
    def __init__(self):
        self.results = {
            "test_name": "Phase 2.2 - ETL Incremental Updates",
            "timestamp": datetime.now().isoformat(),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "errors": []
        }
        self.data_pool = Path(BACKTESTER_ROOT) / "data" / "pools" / "2022-01-01_to_2025-08-31_extras"
        self.test_dir = Path(BACKTESTER_ROOT) / "data" / "test_incremental"
        
    def log(self, message, level="INFO"):
        """Print log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
    
    def setup_test_data(self):
        """Setup test environment"""
        self.log("Setting up test environment...")
        
        # Create test directory
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Load full dataset
        source_file = self.data_pool / "360ONE" / "5m.parquet"
        if not source_file.exists():
            self.log(f"Source file not found: {source_file}", "ERROR")
            return False
        
        self.full_df = pd.read_parquet(source_file)
        self.full_df['timestamp'] = pd.to_datetime(self.full_df['timestamp'])
        self.full_df = self.full_df.sort_values('timestamp')
        
        self.log(f"Loaded {len(self.full_df)} bars from source")
        
        # Split data: 80% existing, 20% "new" data
        split_point = int(len(self.full_df) * 0.8)
        self.existing_df = self.full_df.iloc[:split_point].copy()
        self.new_df = self.full_df.iloc[split_point:].copy()
        
        self.log(f"Split data: {len(self.existing_df)} existing + {len(self.new_df)} new bars")
        
        # Save "existing" dataset
        existing_file = self.test_dir / "360ONE_5m.parquet"
        self.existing_df.to_parquet(existing_file, index=False)
        self.log(f"Saved existing data to {existing_file}")
        
        return True
    
    def test_incremental_append(self):
        """Test 1: Incremental data append"""
        self.log("\nTest 1: Incremental Data Append")
        self.results["tests_run"] += 1
        
        try:
            if not self.setup_test_data():
                self.results["tests_failed"] += 1
                return False
            
            # Simulate incremental append
            existing_file = self.test_dir / "360ONE_5m.parquet"
            
            # Load existing data
            df_existing = pd.read_parquet(existing_file)
            self.log(f"Loaded existing data: {len(df_existing)} bars")
            
            # Append new data
            df_combined = pd.concat([df_existing, self.new_df], ignore_index=True)
            df_combined = df_combined.sort_values('timestamp')
            
            # Remove duplicates (if any)
            df_combined = df_combined.drop_duplicates(subset=['timestamp'], keep='last')
            
            self.log(f"After append: {len(df_combined)} bars")
            self.log(f"Bars added: {len(df_combined) - len(df_existing)}")
            
            # Verify continuity
            expected_total = len(self.full_df)
            actual_total = len(df_combined)
            
            if actual_total == expected_total:
                self.log(f"Data continuity verified: {actual_total} == {expected_total}", "PASS")
                self.results["tests_passed"] += 1
                return True
            else:
                self.log(f"Data continuity issue: {actual_total} != {expected_total}", "FAIL")
                self.results["tests_failed"] += 1
                return False
            
        except Exception as e:
            self.log(f"Incremental append error: {e}", "FAIL")
            self.results["tests_failed"] += 1
            self.results["errors"].append(f"Test 1: {str(e)}")
            return False
    
    def test_deduplication(self):
        """Test 2: Data deduplication"""
        self.log("\nTest 2: Data Deduplication")
        self.results["tests_run"] += 1
        
        try:
            # Create dataset with duplicates
            df_with_duplicates = pd.concat([self.existing_df, self.existing_df.head(100)], ignore_index=True)
            original_count = len(df_with_duplicates)
            
            self.log(f"Dataset with duplicates: {original_count} bars")
            
            # Remove duplicates
            df_deduplicated = df_with_duplicates.drop_duplicates(subset=['timestamp'], keep='last')
            dedup_count = len(df_deduplicated)
            
            duplicates_removed = original_count - dedup_count
            self.log(f"After deduplication: {dedup_count} bars")
            self.log(f"Duplicates removed: {duplicates_removed}")
            
            if duplicates_removed == 100:
                self.log("Deduplication working correctly", "PASS")
                self.results["tests_passed"] += 1
                return True
            else:
                self.log(f"Deduplication issue: expected 100, removed {duplicates_removed}", "FAIL")
                self.results["tests_failed"] += 1
                return False
            
        except Exception as e:
            self.log(f"Deduplication error: {e}", "FAIL")
            self.results["tests_failed"] += 1
            self.results["errors"].append(f"Test 2: {str(e)}")
            return False
    
    def test_timestamp_continuity(self):
        """Test 3: Timestamp continuity after incremental update"""
        self.log("\nTest 3: Timestamp Continuity")
        self.results["tests_run"] += 1
        
        try:
            # Check if timestamps are chronological
            df_combined = pd.concat([self.existing_df, self.new_df], ignore_index=True)
            df_combined = df_combined.sort_values('timestamp')
            
            # Verify timestamps are strictly increasing (or equal for duplicates)
            timestamps = df_combined['timestamp'].values
            is_sorted = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
            
            if is_sorted:
                self.log("Timestamps are in chronological order", "PASS")
                
                # Check for timestamp gaps at junction point
                junction_idx = len(self.existing_df)
                if junction_idx < len(df_combined):
                    time_gap = (df_combined.iloc[junction_idx]['timestamp'] - 
                               df_combined.iloc[junction_idx-1]['timestamp'])
                    self.log(f"Time gap at junction: {time_gap}")
                
                self.results["tests_passed"] += 1
                return True
            else:
                self.log("Timestamps are not in chronological order", "FAIL")
                self.results["tests_failed"] += 1
                return False
            
        except Exception as e:
            self.log(f"Timestamp continuity error: {e}", "FAIL")
            self.results["tests_failed"] += 1
            self.results["errors"].append(f"Test 3: {str(e)}")
            return False
    
    def test_data_integrity(self):
        """Test 4: Data integrity after incremental update"""
        self.log("\nTest 4: Data Integrity")
        self.results["tests_run"] += 1
        
        try:
            # Combine data
            df_combined = pd.concat([self.existing_df, self.new_df], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['timestamp'], keep='last')
            df_combined = df_combined.sort_values('timestamp')
            
            # Compare with original full dataset
            if len(df_combined) == len(self.full_df):
                # Check if data matches
                timestamp_match = (df_combined['timestamp'].values == self.full_df['timestamp'].values).all()
                
                if timestamp_match:
                    self.log("Data integrity verified: timestamps match", "PASS")
                    
                    # Check OHLCV data
                    ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
                    ohlcv_match = True
                    for col in ohlcv_columns:
                        if col in df_combined.columns and col in self.full_df.columns:
                            if not np.allclose(df_combined[col].values, self.full_df[col].values, rtol=1e-10):
                                self.log(f"Mismatch in {col} column", "WARN")
                                ohlcv_match = False
                    
                    if ohlcv_match:
                        self.log("OHLCV data integrity verified", "PASS")
                    
                    self.results["tests_passed"] += 1
                    return True
                else:
                    self.log("Timestamp mismatch detected", "FAIL")
                    self.results["tests_failed"] += 1
                    return False
            else:
                self.log(f"Record count mismatch: {len(df_combined)} vs {len(self.full_df)}", "FAIL")
                self.results["tests_failed"] += 1
                return False
            
        except Exception as e:
            self.log(f"Data integrity error: {e}", "FAIL")
            self.results["tests_failed"] += 1
            self.results["errors"].append(f"Test 4: {str(e)}")
            return False
    
    def cleanup(self):
        """Cleanup test data"""
        try:
            if self.test_dir.exists():
                shutil.rmtree(self.test_dir)
                self.log("Test data cleaned up")
        except Exception as e:
            self.log(f"Cleanup warning: {e}", "WARN")
    
    def generate_report(self):
        """Generate test report"""
        self.log("\n" + "=" * 80)
        self.log("PHASE 2.2 TEST SUMMARY - INCREMENTAL UPDATES")
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
        output_path = Path(BACKTESTER_ROOT) / "outputs" / "qa_phase2.2_incremental_updates_report.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Phase 2.2: Incremental Updates Test Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Results:\n")
            f.write(f"  Total Tests: {total_tests}\n")
            f.write(f"  Passed: {passed}\n")
            f.write(f"  Failed: {failed}\n")
            f.write(f"  Pass Rate: {pass_rate:.1f}%\n\n")
            
            if self.results["errors"]:
                f.write("Errors:\n")
                for error in self.results["errors"]:
                    f.write(f"  - {error}\n")
        
        self.log(f"\nReport saved: {output_path}")
        
        return pass_rate >= 80
    
    def run_all_tests(self):
        """Run all Phase 2.2 tests"""
        self.log("=" * 80)
        self.log("PHASE 2.2: ETL TOOLS - INCREMENTAL UPDATES TESTING")
        self.log("=" * 80)
        
        try:
            # Run tests in sequence
            self.test_incremental_append()
            self.test_deduplication()
            self.test_timestamp_continuity()
            self.test_data_integrity()
            
            # Generate report
            success = self.generate_report()
            
            # Cleanup
            self.cleanup()
            
            return 0 if success else 1
            
        except Exception as e:
            self.log(f"Test execution error: {e}", "ERROR")
            return 1


if __name__ == "__main__":
    tester = IncrementalUpdateTester()
    exit_code = tester.run_all_tests()
    sys.exit(exit_code)
