#!/usr/bin/env python3
"""
Fetch missing Nifty 100 tickers to expand existing data pool.
"""
import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.etl.data_fetcher import DataFetcher

# Missing Nifty 100 tickers (93 total)
MISSING_NIFTY_TICKERS = [
    'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'BAJAJFINSV', 'BAJAJ-AUTO',
    'BPCL', 'BHEL', 'BOSCHLTD', 'CHOLAFIN', 'COLPAL',
    'DLF', 'DABUR', 'DELHIVERY', 'ESCORTS',
    'FEDERALBNK', 'GAIL', 'GLAXO', 'GMRINFRA',
    'GODREJCP', 'GODREJPROP', 'GRANULES', 'GRAPHITE', 'GRASIM',
    'HAVELLS', 'HERCULES', 'HONEYWELL', 'IBULHSGFIN',
    'ICICIBANK', 'IDBI', 'IDFCBANK', 'INDHOTEL', 'INDIGO', 'INDUSIND',
    'IOC', 'IPCALAB', 'IRCTC', 'IRFC', 'ITC', 'ITI',
    'JKCEMENT', 'JSWSTEEL', 'JSL', 'JINDALSTEL',
    'KOTAKBANK', 'LT', 'LALPATHLAB', 'LAURUSLABS',
    'LTIM', 'LTTS', 'LUPIN', 'MANAPPURAM', 'MRF',
    'MARUTI', 'MINDTREE', 'MAXHEALTH', 'MCX', 'MOTHERSUMI',
    'MPHASIS', 'MSUMI', 'NAVNETEDUL', 'NBCC',
    'NDRAUTO', 'NESTLEIND', 'NMDC',
    'NTPC', 'ONGC', 'PAYTM',
    'PERSISTENT', 'PETRONET', 'PFC', 'PIDILITIND', 'PNB',
    'POCL', 'POLYCAB', 'POWERGRID', 'PSB',
    'PSUBANK', 'PVBANK', 'RAMCOCEM', 'RECL',
    'SBICARD', 'SBILIFE', 'SBIN', 'SHREECEM',
    'SHYAMMETL', 'SIEMENS', 'SONACOMS', 'SPAREINDS', 'STLTECH',
    'SUNPHARMA', 'SUNTV', 'SUMMITSEC', 'SYNGENE', 'TATACHEM',
    'TATASTEEL', 'TATAGLOBAL', 'TATAMOTORS', 'TATAPOWER',
    'TITAGARH', 'TORNTPHARM', 'TITAN',
    'TRIVENI', 'TVS', 'TVSMOTOR', 'UBL',
    'UNIONBANK', 'UPL', 'VGUARD', 'VINATIORGA',
    'VIMTALABS', 'WIPRO', 'YESBANK', 'ZEEJENT', 'ZEEL'
]

def fetch_missing_tickers(pool_path_str: str, start_date: str = "2022-01-03", end_date: str = "2025-11-01"):
    """Fetch missing Nifty 100 tickers."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("FetchMissing")
    
    try:
        logger.info(f"Initializing data fetcher...")
        fetcher = DataFetcher()
        
        # Convert string path to Path object
        pool_path = Path(pool_path_str)
        
        logger.info(f"Fetching {len(MISSING_NIFTY_TICKERS)} missing tickers...")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Pool path: {pool_path}")
        
        results = fetcher.fetch_historical_data(
            tickers=MISSING_NIFTY_TICKERS,
            timeframes=['15m', '5m'],
            start_date=start_date,
            end_date=end_date,
            output_dir=pool_path,
            use_ticker_first_storage=True
        )
        
        logger.info(f"Fetch completed!")
        return True
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch missing Nifty 100 tickers")
    parser.add_argument('--pool-path', default='data/pools/2022-01-01_to_2025-08-31/', help='Path to pool')
    parser.add_argument('--start-date', default='2022-01-03', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2025-11-01', help='End date (YYYY-MM-DD)')
    args = parser.parse_args()
    success = fetch_missing_tickers(args.pool_path, args.start_date, args.end_date)
    sys.exit(0 if success else 1)
