# src/etl/data_integrity.py
import os
import json
import logging
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional

from config import BACKTESTER_CONFIG

class DataIntegrityManager:
    def __init__(self, data_pool_dir: Optional[Path] = None):
        self.data_pool_dir = data_pool_dir or Path(BACKTESTER_CONFIG['DATA_POOL_DIR'])
        self.logger = logging.getLogger('DataIntegrityManager')
    
    def validate_data_file(self, file_path: Path) -> bool:
        """
        Validate individual data file integrity
        
        Checks:
        - File exists
        - Not empty
        - Timestamp columns valid
        - No duplicate timestamps
        """
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return False
        
        try:
            df = pd.read_csv(file_path)
            
            # Check file not empty
            if df.empty:
                self.logger.warning(f"Empty data file: {file_path}")
                return False
            
            # Check timestamp column
            timestamp_cols = ['timestamp', 'datetime', 'time']
            timestamp_col = next((col for col in timestamp_cols if col in df.columns), None)
            
            if not timestamp_col:
                self.logger.error(f"No timestamp column in {file_path}")
                return False
            
            # Convert to datetime
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
            
            # Check for invalid timestamps
            if df[timestamp_col].isnull().any():
                self.logger.warning(f"Invalid timestamps in {file_path}")
                return False
            
            # Check for duplicates
            duplicates = df.duplicated(subset=[timestamp_col]).sum()
            if duplicates > 0:
                self.logger.warning(f"{duplicates} duplicate timestamps in {file_path}")
                return False
            
            return True
        
        except Exception as e:
            self.logger.error(f"Validation error for {file_path}: {e}")
            return False
    
    def scan_data_repository(self, timeframe: str = '1minute') -> Dict:
        """
        Scan entire data repository for integrity issues
        """
        overlapping_data = {}
        
        # Scan all date range directories
        for date_range_dir in self.data_pool_dir.glob('*_to_*'):
            if not date_range_dir.is_dir():
                continue
            
            timeframe_dir = date_range_dir / timeframe
            if not timeframe_dir.exists():
                continue
            
            # Scan files for each ticker
            ticker_files = self._group_files_by_ticker(timeframe_dir)
            
            # Check for overlaps
            for ticker, files in ticker_files.items():
                overlap_results = self._detect_overlaps(files)
                if overlap_results:
                    overlapping_data[ticker] = overlap_results
        
        return self._generate_integrity_report(overlapping_data)
    
    def _group_files_by_ticker(self, timeframe_dir: Path) -> Dict:
        """Group CSV files by ticker"""
        ticker_files = {}
        
        for file_path in timeframe_dir.glob('*.csv'):
            try:
                filename = file_path.stem
                parts = filename.split('_')
                ticker = parts[0]
                
                # Parse date range
                start_date = parts[-3]
                end_date = parts[-1]
                
                if ticker not in ticker_files:
                    ticker_files[ticker] = []
                
                ticker_files[ticker].append({
                    'ticker': ticker,
                    'file_path': str(file_path),
                    'start_date': pd.to_datetime(start_date),
                    'end_date': pd.to_datetime(end_date)
                })
            
            except Exception as e:
                self.logger.warning(f"Could not process file {file_path}: {e}")
        
        return ticker_files
    
    def _detect_overlaps(self, files: List[Dict]) -> List[Dict]:
        """Detect overlapping date ranges"""
        overlaps = []
        sorted_files = sorted(files, key=lambda x: x['start_date'])
        
        for i in range(len(sorted_files) - 1):
            current = sorted_files[i]
            next_file = sorted_files[i + 1]
            
            if current['end_date'] >= next_file['start_date']:
                overlaps.append({
                    'files': [current, next_file],
                    'overlap_details': {
                        'first_file_end': current['end_date'],
                        'second_file_start': next_file['start_date']
                    }
                })
        
        return overlaps
    
    def _generate_integrity_report(self, overlapping_data: Dict) -> Dict:
        """Generate comprehensive integrity report"""
        return {
            'total_tickers_with_overlaps': len(overlapping_data),
            'total_overlap_instances': sum(len(overlaps) for overlaps in overlapping_data.values()),
            'overlapping_tickers': list(overlapping_data.keys()),
            'details': overlapping_data
        }
    
    # Additional method in DataIntegrityManager
    def generate_comprehensive_report(self):
        """
        Generate a detailed report of data repository health
        """
        report = {
            'total_tickers': len(self.scan_data_repository()),
            'data_coverage': self._calculate_data_coverage(),
            'potential_issues': self._identify_potential_issues()
        }
        return report
    
    def perform_data_cleanup(self, overlapping_data: Optional[Dict] = None):
        """
        Clean up overlapping data files
        
        Strategies:
        1. Keep most recent file
        2. Merge overlapping data
        3. Remove duplicate entries
        """
        if overlapping_data is None:
            overlapping_data = self.scan_data_repository()
        
        for ticker, overlaps in overlapping_data.items():
            for overlap in overlaps:
                files = overlap['files']
                
                # Strategy: Keep most recent file, merge data
                sorted_files = sorted(files, key=lambda x: x['start_date'])
                base_file = sorted_files[-1]['file_path']
                
                # Load and merge dataframes
                dataframes = [pd.read_csv(f['file_path']) for f in sorted_files]
                merged_df = pd.concat(dataframes).drop_duplicates(subset=['timestamp'])
                
                # Save to most recent file
                merged_df.to_csv(base_file, index=False)
                
                # Remove other files
                for file_info in sorted_files[:-1]:
                    os.remove(file_info['file_path'])
                
                self.logger.info(f"Cleaned up overlapping files for {ticker}")

# Utility function for easy access
def validate_data_file(file_path: Path) -> bool:
    """Convenience function to validate a single data file"""
    integrity_manager = DataIntegrityManager()
    return integrity_manager.validate_data_file(file_path)