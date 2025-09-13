import pandas as pd
from config import BACKTESTER_CONFIG
import os
import logging
import glob
from typing import Optional, Dict, List, Union
from pathlib import Path

# Optional parquet support
try:
    import pyarrow
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

def load_multi_timeframe_data(
    pull_date: str, 
    ticker: str, 
    required_timeframes: List[str],
    use_ticker_first_storage: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Load multiple timeframes for a ticker from parquet/CSV files (strategy-driven).
    
    This function loads exactly the timeframes requested by the strategy,
    supporting both new ticker-first parquet structure and legacy timeframe-first CSV.
    
    Args:
        pull_date: Date range string in format 'YYYY-MM-DD_to_YYYY-MM-DD'
        ticker: Ticker symbol to load data for
        required_timeframes: List of timeframes required by the strategy (e.g., ['5m', '15m'])
        use_ticker_first_storage: Whether to use new ticker-first structure
        
    Returns:
        Dictionary mapping timeframes to DataFrames: {'5m': df5, '15m': df15}
        Returns empty dict if no data found for any required timeframe
    """
    logger = logging.getLogger("DataLoader")
    logger.info(f"Loading multi-timeframe data for {ticker}: {required_timeframes}")
    
    data_pool_dir = Path(BACKTESTER_CONFIG['DATA_POOL_DIR'])
    date_dir = data_pool_dir / pull_date
    
    if not date_dir.exists():
        logger.error(f"Date directory does not exist: {date_dir}")
        return {}
    
    loaded_data = {}
    
    for timeframe in required_timeframes:
        df = None
        
        if use_ticker_first_storage:
            # New ticker-first structure: data/pools/{date}/{ticker}/{timeframe}.parquet
            ticker_dir = date_dir / ticker
            
            # Try parquet first
            if PARQUET_AVAILABLE:
                parquet_file = ticker_dir / f"{timeframe}.parquet"
                if parquet_file.exists():
                    try:
                        df = pd.read_parquet(parquet_file)
                        logger.debug(f"✅ Loaded {len(df)} records from {parquet_file}")
                    except Exception as e:
                        logger.warning(f"Failed to read parquet {parquet_file}: {e}")
            
            # Fallback to CSV in ticker-first structure
            if df is None:
                csv_file = ticker_dir / f"{timeframe}.csv"
                if csv_file.exists():
                    try:
                        df = pd.read_csv(csv_file)
                        logger.debug(f"📄 Loaded {len(df)} records from {csv_file}")
                    except Exception as e:
                        logger.warning(f"Failed to read CSV {csv_file}: {e}")
        
        
        # Process and validate the loaded DataFrame
        if df is not None and not df.empty:
            # Standardize timestamp column
            df = _standardize_timestamp_column(df)
            if df is not None:
                loaded_data[timeframe] = df
                logger.info(f"✅ Successfully loaded {timeframe} data: {len(df)} records")
            else:
                logger.warning(f"❌ Failed to standardize {timeframe} data")
        else:
            logger.warning(f"❌ No data found for {ticker} at {timeframe} timeframe")
    
    # Final validation
    missing_timeframes = set(required_timeframes) - set(loaded_data.keys())
    if missing_timeframes:
        logger.error(f"Missing required timeframes: {missing_timeframes}")
        if len(missing_timeframes) == len(required_timeframes):
            logger.error(f"No data loaded for any required timeframe")
            return {}
    
    logger.info(f"🎯 Multi-timeframe loading complete: {list(loaded_data.keys())} / {required_timeframes}")
    return loaded_data


def _standardize_timestamp_column(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Standardize timestamp column naming and format.
    
    Args:
        df: DataFrame with potential timestamp columns
        
    Returns:
        DataFrame with standardized 'timestamp' column or None if failed
    """
    logger = logging.getLogger("DataLoader")
    
    # Find and rename timestamp column
    timestamp_cols = ['timestamp', 'datetime', 'time']
    found = False
    
    for col in timestamp_cols:
        if col in df.columns:
            if col != 'timestamp':
                df.rename(columns={col: 'timestamp'}, inplace=True)
            found = True
            break
    
    if not found:
        logger.error(f"No recognized timestamp column found. Expected one of {timestamp_cols}")
        return None
    
    # Ensure timestamp is datetime format
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        if df['timestamp'].isnull().any():
            logger.error("Some timestamp values could not be parsed")
            return None
    except Exception as e:
        logger.error(f"Error converting timestamp column: {e}")
        return None
        
    return df


def load_base_data(pull_date: str, ticker: str) -> Optional[pd.DataFrame]:
    """
    Legacy function: Load base data for a given ticker and date from available timeframes.
    
    This function is maintained for backward compatibility with existing single-timeframe strategies.
    New multi-timeframe strategies should use load_multi_timeframe_data() instead.
    
    Args:
        pull_date: Date range string in format 'YYYY-MM-DD_to_YYYY-MM-DD'
        ticker: Ticker symbol to load data for
    
    Returns:
        DataFrame with historical price data or None if no data found
    """
    # Try multiple timeframes in order of preference
    timeframes_to_try = ["1m", "day", "5m", "15m", "1h"]
    csv_files = []
    
    for tf in timeframes_to_try:
        if tf in BACKTESTER_CONFIG['TIMEFRAME_FOLDERS']:
            base_file_pattern = os.path.join(
                BACKTESTER_CONFIG['DATA_POOL_DIR'],
                pull_date,
                BACKTESTER_CONFIG['TIMEFRAME_FOLDERS'][tf],
                f"{ticker}_*.csv"
            )
            csv_files = glob.glob(base_file_pattern)
            if csv_files:
                logging.info(f"Found {len(csv_files)} CSV files for {ticker} in {tf} timeframe")
                break

    if not csv_files:
        logging.warning(f"No CSV files found for ticker '{ticker}' on date range '{pull_date}' in any supported timeframe.")
        return None

    data_frames = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Identify and rename the timestamp column
            timestamp_cols = ['timestamp', 'datetime', 'time']
            found = False
            for col in timestamp_cols:
                if col in df.columns:
                    df.rename(columns={col: 'timestamp'}, inplace=True)
                    found = True
                    break
            if not found:
                logging.error(f"No recognized timestamp column found in file '{file}'. Expected one of {timestamp_cols}.")
                return None

            # Ensure 'timestamp' is in datetime format
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            if df['timestamp'].isnull().any():
                logging.error(f"Some 'timestamp' values could not be parsed in file '{file}'.")
                return None

            data_frames.append(df)
        except Exception as e:
            logging.error(f"Error reading file '{file}': {e}")

    if not data_frames:
        logging.warning(f"No valid data loaded for ticker '{ticker}' on date range '{pull_date}'.")
        return None

    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df.drop_duplicates(subset=['timestamp'], inplace=True)
    combined_df.sort_values('timestamp', inplace=True)
    combined_df.reset_index(drop=True, inplace=True)

    return combined_df


def load_strategy_data(pull_date: str, ticker: str, strategy) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Strategy-driven data loader: Loads exactly what the strategy requires.
    
    This is the main entry point for loading data based on strategy requirements.
    It automatically detects if the strategy needs single or multi-timeframe data
    and returns the appropriate format.
    
    Args:
        pull_date: Date range string in format 'YYYY-MM-DD_to_YYYY-MM-DD'
        ticker: Ticker symbol to load data for  
        strategy: Strategy instance with required_timeframes property
        
    Returns:
        Single DataFrame (legacy strategies) or Dict of DataFrames (multi-timeframe strategies)
    """
    logger = logging.getLogger("DataLoader")
    
    # Get strategy timeframe requirements
    required_timeframes = strategy.required_timeframes
    logger.info(f"Strategy '{strategy.name}' requires timeframes: {required_timeframes}")
    
    if len(required_timeframes) == 1:
        # Single timeframe - return DataFrame for backward compatibility
        timeframe = required_timeframes[0]
        data = load_multi_timeframe_data(pull_date, ticker, [timeframe])
        
        if data:
            logger.info(f"Returning single DataFrame for {timeframe} timeframe")
            return data[timeframe]
        else:
            logger.warning(f"No data loaded for single timeframe {timeframe}")
            return pd.DataFrame()
    else:
        # Multi-timeframe - return Dict of DataFrames
        data = load_multi_timeframe_data(pull_date, ticker, required_timeframes)
        
        if data:
            logger.info(f"Returning multi-timeframe Dict: {list(data.keys())}")
            return data
        else:
            logger.warning(f"No data loaded for multi-timeframes {required_timeframes}")
            return {}