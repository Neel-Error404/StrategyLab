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

# Missing Nifty 100 tickers (93 total, excluding ADANIPORTS which is rate-limited)
MISSING_NIFTY_TICKERS = [
    'APOLLOHOSP', 'ASIANPAINT', 'BAJAJFINSV', 'BAJAJ-AUTO',
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

def fetch_missing_tickers(pool_path_str: str, start_date: str = "2022-01-03", end_date: str = "2025-11-01", batch_size: int = 2):
    """Fetch missing Nifty 100 tickers with rate limit handling."""
    import time
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("FetchMissing")
    
    try:
        logger.info(f"Initializing data fetcher...")
        fetcher = DataFetcher()
        
        # Convert string path to Path object
        pool_path = Path(pool_path_str)
        
        # Remove duplicates and sort
        unique_tickers = sorted(list(set(MISSING_NIFTY_TICKERS)))
        logger.info(f"Fetching {len(unique_tickers)} unique missing tickers...")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Pool path: {pool_path}")
        logger.info(f"Batch size: {batch_size} tickers per batch (30s pause between batches)")
        
        # Process in batches to avoid aggressive rate limiting
        total_success = 0
        total_batches = (len(unique_tickers) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(unique_tickers), batch_size):
            batch = unique_tickers[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            logger.info(f"\n{'='*80}")
            logger.info(f"BATCH {batch_num}/{total_batches} - Processing: {', '.join(batch)}")
            logger.info(f"{'='*80}")
            
            try:
                results = fetcher.fetch_historical_data(
                    tickers=batch,
                    timeframes=['15m', '5m'],
                    start_date=start_date,
                    end_date=end_date,
                    output_dir=pool_path,
                    use_ticker_first_storage=True
                )
                
                batch_success = len([t for t in results if results[t]])
                total_success += batch_success
                logger.info(f"✅ Batch {batch_num} result: {batch_success}/{len(batch)} tickers successful")
                logger.info(f"📊 Running total: {total_success}/{len(unique_tickers)} tickers fetched")
                
                # Wait between batches to avoid rate limiting
                if batch_idx + batch_size < len(unique_tickers):
                    wait_time = 30
                    logger.info(f"⏸️  Waiting {wait_time}s before next batch...")
                    time.sleep(wait_time)
            
            except KeyboardInterrupt:
                logger.error(f"❌ Batch {batch_num} interrupted by user")
                raise
            except Exception as e:
                logger.error(f"❌ Batch {batch_num} error: {e}", exc_info=True)
                continue
        
        logger.info(f"\n{'='*80}")
        logger.info(f"FETCH COMPLETED!")
        logger.info(f"Final result: {total_success}/{len(unique_tickers)} tickers successful")
        logger.info(f"{'='*80}")
        return total_success > 0
        
    except KeyboardInterrupt:
        logger.warning(f"Fetch interrupted by user. {total_success} tickers fetched so far.")
        return False
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
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
