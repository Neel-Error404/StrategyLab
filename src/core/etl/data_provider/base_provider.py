# src/etl/data_providers/base_provider.py
from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

class DataProvider(ABC):
    """Abstract base class for all market data providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.authenticated = False
    
    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the data provider API."""
        pass
    
    @abstractmethod
    def fetch_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                              timeframe: str) -> pd.DataFrame:
        """Fetch historical OHLCV data for the specified symbol and timeframe."""
        pass
    
    @abstractmethod
    def map_timeframe(self, standard_timeframe: str) -> str:
        """Map standardized timeframe to provider-specific timeframe."""
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols."""
        pass
    
    @abstractmethod
    def symbol_to_instrument_id(self, symbol: str) -> str:
        """Convert symbol to provider-specific instrument identifier."""
        pass
    
    @abstractmethod
    def fetch_instrument_details(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Fetch instrument details including symbol mapping."""
        pass
    
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize provider-specific data to standard format."""
        if df.empty:
            return df
            
        # Ensure consistent column naming
        column_mapping = {
            'date': 'timestamp',
            'datetime': 'timestamp',
            'time': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        }
        
        # Rename columns if they exist
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        return df