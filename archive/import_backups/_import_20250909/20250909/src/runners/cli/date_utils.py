"""
Date parsing utilities for the unified backtester.
"""

from datetime import datetime
from typing import List


def parse_dates(dates: List[str]) -> List[str]:
    """
    Parse and normalize date strings to ensure consistency.
    
    Args:
        dates: List of date strings in various formats
        
    Returns:
        List of normalized date strings
    """
    normalized_dates = []
    
    for date_str in dates:
        try:
            # Handle different date formats
            if '_to_' in date_str:
                # Already a date range
                start_str, end_str = date_str.split('_to_')
                start_date = datetime.strptime(start_str, "%Y-%m-%d")
                end_date = datetime.strptime(end_str, "%Y-%m-%d")
                normalized_dates.append(f"{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}")
            else:
                # Single date
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                normalized_dates.append(date_obj.strftime("%Y-%m-%d"))
                
        except ValueError as e:
            raise ValueError(f"Invalid date format '{date_str}': {e}")
    
    return normalized_dates
