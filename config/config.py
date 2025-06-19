#cd!/usr/bin/env python
"""
config.py

Centralized configuration for both backtester and live trading modules.
Handles environment setup, logging configuration, directory management, 
and authentication utilities for both Upstox and Zerodha.

Key features:
- Unified configuration for all trading modules
- Multi-broker support (Upstox and Zerodha)
- Environment-aware settings with fallbacks
- Custom logging setup with timezone handling
- Thread-safe token management
- Robust error handling and recovery
- Filesystem organization for data and logs
"""

from pathlib import Path
from datetime import datetime, timedelta
import os
import json
import logging
import sys
import requests
import threading
import pytz
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
from typing import Dict, Any, Optional, Union
import traceback
from urllib.parse import urlencode
import platform
import uuid

# Setup basic logging initially (will be enhanced in setup_logging functions)
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# Control logging level for third-party libraries
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)

# Define the base directory as the parent of this file's directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Create a unique system ID for this instance
SYSTEM_ID = str(uuid.uuid4())[:8]

# Define IST timezone for consistent timestamping
IST = pytz.timezone("Asia/Kolkata")

# Thread lock for token operations
_token_lock = threading.RLock()

# Environment detection
ENV = os.getenv("TRADING_ENV", "development").lower()
IS_PRODUCTION = ENV == "production"
IS_DEVELOPMENT = ENV == "development"
IS_TESTING = ENV == "testing"

# System information
SYSTEM_INFO = {
    "hostname": platform.node(),
    "platform": platform.platform(),
    "python_version": platform.python_version(),
    "cpu_count": os.cpu_count(),
    "system_id": SYSTEM_ID,
    "environment": ENV
}

# === Upstox Configuration ===
UPSTOX_CONFIG = {
    # API credentials - MUST be set via environment variables
    'CLIENT_ID': os.getenv('UPSTOX_CLIENT_ID'),
    'CLIENT_SECRET': os.getenv('UPSTOX_CLIENT_SECRET'),
    'REDIRECT_URI': os.getenv('UPSTOX_REDIRECT_URI', "https://127.0.0.1:5000/"),
     'INSTRUMENTS_CSV': BASE_DIR / 'config' / 'complete.csv',
    # API endpoints
    'AUTH_URL': "https://api.upstox.com/v2/login/authorization/dialog",
    'TOKEN_URL': "https://api.upstox.com/v2/login/authorization/token",
    
    # Data fetching parameters
    'MAX_DAYS_PER_REQUEST': 200,
    
    # Supported timeframes
    'SUPPORTED_TIMEFRAMES': ['1minute', '30minute', 'day', 'week', 'month'],
    
    # Token storage
    'ACCESS_TOKEN_DIR': BASE_DIR / 'config' / 'access_tokens' / 'upstox',
    'ACCESS_TOKEN_FILE_TEMPLATE': str(BASE_DIR / 'config' / 'access_tokens' / 'upstox' / 'access_token_{}.json'),
    
    # Error handling and recovery
    'MAX_RETRIES': 3,
    'RETRY_DELAY': 5,  # seconds
    'REQUEST_TIMEOUT': 30  # seconds
}

# === Zerodha Configuration ===
ZERODHA_CONFIG = {
    # API credentials - MUST be set via environment variables
    'API_KEY': os.getenv('ZERODHA_API_KEY'),
    'API_SECRET': os.getenv('ZERODHA_API_SECRET'),
    'REDIRECT_URI': os.getenv('ZERODHA_REDIRECT_URI', "https://127.0.0.1:5000/"),
    
    #'# Token storage paths - make sure these are defined
    'KEY_CSV_LOCATION': str(BASE_DIR / 'config' / 'access_tokens' / 'zerodha' / 'access_token.csv'),
    'ACCESS_TOKEN_DIR': str(BASE_DIR / 'config' / 'access_tokens' / 'zerodha'),
    'ACCESS_TOKEN_FILE_TEMPLATE': str(BASE_DIR / 'config' / 'access_tokens' / 'zerodha' / 'access_token_{}.json'),
    
    'SEGMENT': 'NSE',
    #'KEY_CSV_LOCATION': BASE_DIR / 'config' / 'access_tokens' / 'zerodha' / 'access_token.csv',
    'INSTRUMENTS_CSV': BASE_DIR / 'config' / 'zerodha_instruments.csv',

    
    # Supported timeframes
    'SUPPORTED_TIMEFRAMES': ['minute', '3minute', '5minute', '10minute', '15minute', '30minute', 'hour', 'day', 'week', 'month'],
    
    # Historical data limits by timeframe
    'HISTORICAL_LIMITS': {
        'minute': {'days': 60, 'candles_per_request': 60},
        '3minute': {'days': 100, 'candles_per_request': 100},
        '5minute': {'days': 100, 'candles_per_request': 100}, 
        '15minute': {'days': 200, 'candles_per_request': 200},
        '30minute': {'days': 200, 'candles_per_request': 200},
        'hour': {'days': 400, 'candles_per_request': 400},
        'day': {'days': 2000, 'candles_per_request': 2000},
        'week': {'days': 2000, 'candles_per_request': 2000},
        'month': {'days': 2000, 'candles_per_request': 2000}
    },
    
    # Token storage
    'ACCESS_TOKEN_DIR': BASE_DIR / 'config' / 'access_tokens' / 'zerodha',
    
    # Error handling and recovery
    'MAX_RETRIES': 3,
    'RETRY_DELAY': 5,  # seconds
    'REQUEST_TIMEOUT': 30  # seconds
}

# === Backtester Configuration ===
BACKTESTER_CONFIG = {
    # Primary data provider (can be 'upstox' or 'zerodha')
    'DATA_PROVIDER': os.getenv('DATA_PROVIDER', 'upstox'),
    
    # API configurations
    'UPSTOX_CONFIG': UPSTOX_CONFIG,
    'ZERODHA_CONFIG': ZERODHA_CONFIG,
    
    # Default values
    'DEFAULT_TICKER': ["RELIANCE", "TCS", "INFY"],
    'DEFAULT_TIMEFRAME': ['1m', 'day'],
    
    # Standardized timeframe definitions (used across the system)
    'STANDARD_TIMEFRAMES': {
        '1m': {'upstox': '1minute', 'zerodha': 'minute'},
        '3m': {'upstox': None, 'zerodha': '3minute'},
        '5m': {'upstox': None, 'zerodha': '5minute'},
        '10m': {'upstox': None, 'zerodha': '10minute'},
        '15m': {'upstox': None, 'zerodha': '15minute'},
        '30m': {'upstox': '30minute', 'zerodha': '30minute'},
        '1h': {'upstox': None, 'zerodha': 'hour'},
        'day': {'upstox': 'day', 'zerodha': 'day'},
        'week': {'upstox': 'week', 'zerodha': 'week'},
        'month': {'upstox': 'month', 'zerodha': 'month'}
    },
    
    # Supported timeframes in standardized format
    'SUPPORTED_TIMEFRAMES': ['1m', '3m', '5m', '10m', '15m', '30m', '1h', 'day', 'week', 'month'],
    
    # File paths and directories
    'INSTRUMENTS_CSV': BASE_DIR / 'config' / 'complete.csv',
    'DATA_POOL_DIR': BASE_DIR / 'data' / 'pools',
    'OUTPUT_FOLDER': BASE_DIR / '.runtime' / 'outputs',
    'LOG_DIR': BASE_DIR / '.runtime' / 'logs',
    'ANALYSIS_DIR': BASE_DIR / '.runtime' / 'outputs' / 'analysis',
    
    # Mapping from standard timeframes to folder names
    'TIMEFRAME_FOLDERS': {
        '1m': '1minute',
        '3m': '3minute',
        '5m': '5minute',
        '10m': '10minute', 
        '15m': '15minute',
        '30m': '30minute',
        '1h': 'hour',
        'day': 'day',
        'week': 'week', 
        'month': 'month'
    },
    
    # Strategy configuration
    'STRATEGIES': {
        "mse": "strategies.strategy_mse"
    },
    
    # Execution parameters
    'NUM_PROCESSES': min(16, os.cpu_count() or 4),
    
    # Backtesting date range
    'PROCESS_DATES': [
        "2024-12-29",
        "2024-12-31"
    ],
    
    # Analysis configuration
    'ANALYSIS': {
        'INITIAL_CAPITAL': 100000,
        'POSITION_SIZE_PCT': 0.1,
        'MAX_DRAWDOWN_PCT': 25,
        'OUTPUT_DIR': BASE_DIR / 'Backtester' / 'Analysis'
    }
    ,
    'DATA_INTEGRITY': {
        'AUTO_CHECK': True,  # Run checks automatically
        'CLEANUP_MODE': 'merge',  # How to handle overlaps
        'REPORT_DIR': BASE_DIR / 'Backtester' / 'integrity_reports'
    },
    'LOGGING': {
        'DATA_LOADER': {
            'LEVEL': logging.INFO,
            'FILE_ROTATION': True,
            'MAX_FILES': 5
        }
    },
    
    'DATA_FETCHER': {
        'INTEGRITY_CHECK': True,
        'MAX_RETRY_FETCH': 3,
        'FETCH_TIMEOUT': 60,  # seconds
        'ERROR_HANDLING': {
            'SKIP_ON_FAILURE': False,
            'LOG_LEVEL': 'WARNING'
        }
    }
}

# === Live Trading Configuration (kept unchanged) ===
LIVE_TRADING_CONFIG = {
    # API credentials - MUST be set via environment variables
    'CLIENT_ID': os.getenv('CLIENT_ID'),
    'CLIENT_SECRET': os.getenv('CLIENT_SECRET'),
    'REDIRECT_URI': os.getenv('REDIRECT_URI', "https://127.0.0.1:5000/"),
    
    # API endpoints
    'AUTH_URL': "https://api.upstox.com/v2/login/authorization/dialog",
    'TOKEN_URL': "https://api.upstox.com/v2/login/authorization/token",
    
    # Trading mode configuration
    'TRADING_MODE': os.getenv('TRADING_MODE', 'PAPER'),  # LIVE, PAPER, or DEMO
    'ORDER_EXECUTOR_MODE': os.getenv('ORDER_EXECUTOR_MODE', 'PAPER'),
    
    # Market data configuration
    'USE_FULL_MARKET_DATA': False,
    
    # File paths and directories (using proper data structure)
    'DATA_DIR': BASE_DIR / 'data' / 'live_trading',
    'LOG_DIR': BASE_DIR / 'logs' / 'live_trading',
    'REPORT_DIR': BASE_DIR / 'outputs' / 'live_trading',
    'INSTRUMENTS_CSV': BASE_DIR / 'config' / 'complete.csv',
    
    # Trading universe
    'MY_TICKERS': ['RELIANCE', 'TCS', 'INFY', 'IREDA', 'ADANIENT', 'TATAMOTORS', 'ZOMATO'],
    
    # Capital and risk management
    'TOTAL_CAPITAL': 1000000,
    'CAPITAL_RESERVE_PERCENTAGE': 10,  # Keep 10% as reserve
    'RISK_MANAGEMENT': {
        'MAX_PERCENT_PER_TRADE': 0.5,  # 50% of allocated capital per trade
        'MAX_DRAWDOWN_PERCENT': 0.05,   # 5% max drawdown before intervention
        'STOP_LOSS_PERCENT': 5.0,      # 2% stop loss per trade
        'TAKE_PROFIT_PERCENT': 5.0,    # 5% take profit target per trade
        'TRAILING_STOP_PERCENT': 1.0   # 1% trailing stop once in profit
    },
    
    # Capital allocation
    'ALLOCATION_PERCENTAGES': {
        'RELIANCE': 25,
        'ADANIENT': 25,
        'TATAMOTORS': 25,
        'ZOMATO': 10
    },
    
    # Order configuration
    'ORDER_TYPE': {
        'BUY': 'MARKET',
        'SELL': 'MARKET'
    },
    
    # Trade logging
    'TRADE_LOG_PATH': BASE_DIR / 'logs' / 'live_trading' / 'trades.log',
    
    # Time intervals
    'TIMEFRAME_LABEL': '1m',
    'RISK_CHECK_INTERVAL': 60,  # seconds
    'MARKET_SESSION_CHECK_INTERVAL': 60,  # seconds
    'HEALTH_CHECK_INTERVAL': 60,  # seconds
    'SUMMARY_INTERVAL': 60,  # seconds
    
    # Component configuration
    'CRITICAL_COMPONENTS': [
        'event_bus',
        'position_manager',
        'portfolio_handler',
        'order_executor',
        'ticker_runner',
        'risk_manager'
    ],
    
    # Token storage
    'ACCESS_TOKEN_DIR': BASE_DIR / 'config' / 'access_tokens',
    'ACCESS_TOKEN_FILE_TEMPLATE': str(BASE_DIR / 'config' / 'access_tokens' / 'access_token_{}.json'),
    
    # Error handling and recovery
    'MAX_RETRIES': 3,
    'RETRY_DELAY': 5,  # seconds
    'REQUEST_TIMEOUT': 30,  # seconds
    
    # Web dashboard (if implemented)
    'DASHBOARD_ENABLED': False,
    'DASHBOARD_PORT': 8080,
    
    # Notification configuration
    'NOTIFICATIONS_ENABLED': False,
    'NOTIFICATION_EMAIL': os.getenv('NOTIFICATION_EMAIL', ''),
    'NOTIFICATION_LEVELS': ['ERROR', 'CRITICAL']
}

# Shared configuration
COMMON_CONFIG = {
    'VERSION': '0.05',  # Strategy Lab version
    'SYSTEM_INFO': SYSTEM_INFO,
    'SYSTEM_ID': SYSTEM_ID,
    'BASE_DIR': BASE_DIR,
    'ENV': ENV,
    'IST_TIMEZONE': IST,
    
    # Market session times (IST)
    'MARKET_SESSIONS': {
        'PRE_MARKET_START': '09:00',
        'MARKET_OPEN': '09:15',
        'MARKET_CLOSE': '15:30',
        'POST_MARKET_END': '16:00'
    },
    
    # Logging configuration
    'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
    'LOG_RETENTION_DAYS': 30,
    'LOG_MAX_SIZE_MB': 50,
    
    
}


# === Custom Logging Formatter for IST ===

class ISTFormatter(logging.Formatter):
    """
    Custom formatter to ensure all logs are timestamped in IST.
    """
    def formatTime(self, record, datefmt=None):
        # Convert record.created (float -> datetime in IST)
        dt = datetime.fromtimestamp(record.created, IST)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            # Default ISO8601 format or your preferred format
            return dt.isoformat()


# === Logging Setup Function ===

def setup_logging_back(module='backtester'):
    """
    Sets up logging for the backtester.
    Logs are stored in the Backtester/logs directory, with daily rotation.
    
    Args:
        module: Name of the module for which logging is being set up.
        
    Returns:
        Configured logger instance
    """
    log_dir = BACKTESTER_CONFIG['LOG_DIR']
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create a log file name with today's date (in IST)
    current_date = datetime.now(IST).strftime('%Y-%m-%d')
    log_file = log_dir / f"{module}_{current_date}.log"
    
    logger = logging.getLogger(module)
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers to prevent duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = ISTFormatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s', '%Y-%m-%d %H:%M:%S')

    # File handler with daily rotation
    file_handler = TimedRotatingFileHandler(
        filename=str(log_file),
        when='midnight',
        backupCount=COMMON_CONFIG['LOG_RETENTION_DAYS'],
        utc=False
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler for INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    # Log system information
    logger.info(f"Logger initialized for module '{module}'. Log file: {log_file}")
    logger.info(f"System ID: {SYSTEM_ID}")
    logger.info(f"Environment: {ENV}")
    logger.info(f"Python version: {SYSTEM_INFO['python_version']}")
    
    return logger


def setup_logging(module='live_trading'):
    """
    Sets up logging for the specified module.
    Logs are written to both a file and the console.
    
    Args:
        module: Name of the module for which logging is being set up.
        
    Returns:
        Configured logger instance
    """
    # Define the log directory
    log_dir = LIVE_TRADING_CONFIG['LOG_DIR']
    log_dir.mkdir(parents=True, exist_ok=True)

    # Define the log file path
    log_file = log_dir / f'{module}.log'

    # Create a logger specific to the module
    logger = logging.getLogger(module)
    logger.setLevel(logging.DEBUG)

    # Prevent adding multiple handlers if setup_logging is called multiple times for the same module
    if logger.handlers:
        return logger

    # File handler with size-based rotation
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=COMMON_CONFIG['LOG_MAX_SIZE_MB'] * 1024 * 1024,
        backupCount=COMMON_CONFIG['LOG_RETENTION_DAYS']
    )
    file_handler.setLevel(logging.DEBUG)

    # Console handler for INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and set it for both handlers
    formatter = ISTFormatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Prevent log messages from being propagated to the root logger
    logger.propagate = False
    
    # Log system information
    logger.info(f"Logger initialized for module '{module}'. Log file: {log_file}")
    logger.info(f"System ID: {SYSTEM_ID}")
    logger.info(f"Environment: {ENV}")
    logger.info(f"Trading mode: {LIVE_TRADING_CONFIG['TRADING_MODE']}")
    
    return logger


# === Directory Initialization ===

def ensure_dirs_exist():
    """
    Ensures all required directories exist.
    Creates them if they don't exist.
    """
    directories = [
        BACKTESTER_CONFIG['DATA_POOL_DIR'],
        BACKTESTER_CONFIG['OUTPUT_FOLDER'],
        BACKTESTER_CONFIG['LOG_DIR'],
        BACKTESTER_CONFIG['ANALYSIS_DIR'],
        UPSTOX_CONFIG['ACCESS_TOKEN_DIR'],
        ZERODHA_CONFIG['ACCESS_TOKEN_DIR'],
        LIVE_TRADING_CONFIG['DATA_DIR'],
        LIVE_TRADING_CONFIG['LOG_DIR'],
        LIVE_TRADING_CONFIG['REPORT_DIR'],
        LIVE_TRADING_CONFIG['ACCESS_TOKEN_DIR']
    ]
    
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.error(f"Error creating directory {directory}: {e}")


# === Upstox Access Token Management Functions ===

def save_upstox_access_token(token_data: dict):
    """
    Saves the Upstox access token to a file with the current date in IST.
    Adds a 'token_date' field to the token data for daily validation.

    Args:
        token_data: Dictionary containing access token information.
    """
    with _token_lock:
        try:
            # Ensure access tokens directory exists
            access_tokens_dir = UPSTOX_CONFIG['ACCESS_TOKEN_DIR']
            access_tokens_dir.mkdir(parents=True, exist_ok=True)

            # Add 'token_date' to token_data
            token_date = datetime.now(IST).strftime("%Y-%m-%d")
            token_data["token_date"] = token_date
            
            # Add system and timestamp information
            token_data["system_id"] = SYSTEM_ID
            token_data["saved_at"] = datetime.now(IST).isoformat()

            # Determine filename
            date_str = token_date
            token_file = UPSTOX_CONFIG['ACCESS_TOKEN_FILE_TEMPLATE'].format(date_str)
            
            # Write token to file
            with open(token_file, 'w') as f:
                json.dump(token_data, f, indent=2)
                
            logging.info(f"Upstox access token saved to {token_file}")
            return True
        except Exception as e:
            logging.error(f"Failed to save Upstox access token: {e}")
            logging.debug(traceback.format_exc())
            return False


def load_upstox_access_token() -> Optional[str]:
    """
    Loads the latest Upstox access token from the access_tokens directory.
    Checks if the token's date matches the current IST date.
    If not, returns None to force a token refresh.

    Returns:
        Access token string if valid, else None.
    """
    with _token_lock:
        try:
            # Ensure access tokens directory exists
            access_tokens_dir = UPSTOX_CONFIG['ACCESS_TOKEN_DIR']
            if not access_tokens_dir.exists():
                logging.warning(f"Upstox access tokens directory does not exist: {access_tokens_dir}")
                return None

            # List all token files and sort them by date descending
            token_files = sorted(
                [f for f in access_tokens_dir.glob("access_token_*.json")],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            if not token_files:
                logging.warning("No Upstox access token files found.")
                return None

            # Load the latest token file
            latest_token_file = token_files[0]
            with open(latest_token_file, 'r') as f:
                token_data = json.load(f)
                
            logging.debug(f"Loaded Upstox access token from {latest_token_file}")

            # Check if 'token_date' exists
            token_date_str = token_data.get("token_date")
            if not token_date_str:
                logging.warning("token_date not found in saved Upstox token; forcing re-auth.")
                return None

            # Get today's date in IST
            today_ist = datetime.now(IST).strftime("%Y-%m-%d")
            if token_date_str != today_ist:
                logging.warning(f"Upstox token date {token_date_str} != today ({today_ist}). Forcing re-auth.")
                return None

            # If valid, return the access token
            access_token = token_data.get("access_token")
            if not access_token:
                logging.error("access_token not found in Upstox token data.")
                return None
                
            return access_token
            
        except Exception as e:
            logging.error(f"Error loading Upstox access token: {e}")
            logging.debug(traceback.format_exc())
            return None


# === Zerodha Access Token Management Functions ===

def save_zerodha_access_token(access_token: str, request_token: str = None, api_key: str = None) -> bool:
    """
    Saves the Zerodha access token to a JSON file with date information.
    
    Args:
        access_token: Zerodha access token string
        request_token: Optional request token used to generate the access token
        api_key: Optional API key used for authentication
        
    Returns:
        True if successful, False otherwise
    """
    with _token_lock:
        try:
            # Ensure access tokens directory exists
            access_tokens_dir = ZERODHA_CONFIG['ACCESS_TOKEN_DIR']
            access_tokens_dir.mkdir(parents=True, exist_ok=True)
            
            # Create token data structure similar to Upstox
            token_date = datetime.now(IST).strftime("%Y-%m-%d")
            token_data = {
                "access_token": access_token,
                "token_date": token_date,
                "system_id": SYSTEM_ID,
                "saved_at": datetime.now(IST).isoformat()
            }
            
            if request_token:
                token_data["request_token"] = request_token
                
            if api_key:
                token_data["api_key"] = api_key
            
            # Save token in JSON format using date-based filename
            token_file = ZERODHA_CONFIG['ACCESS_TOKEN_FILE_TEMPLATE'].format(token_date)
            with open(token_file, 'w') as f:
                json.dump(token_data, f, indent=2)
                
            # Also save in legacy CSV format for backward compatibility
            legacy_file = ZERODHA_CONFIG['KEY_CSV_LOCATION']
            legacy_file.parent.mkdir(parents=True, exist_ok=True)
            with open(legacy_file, 'w') as f:
                f.write(access_token)
                
            logging.info(f"Zerodha access token saved to {token_file} and {legacy_file}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save Zerodha access token: {e}")
            logging.debug(traceback.format_exc())
            return False

def load_zerodha_access_token() -> Optional[str]:
    """
    Loads the latest Zerodha access token and validates it based on date.
    
    Returns:
        Access token string if valid, else None
    """
    with _token_lock:
        try:
            # Ensure access tokens directory exists
            access_tokens_dir = ZERODHA_CONFIG['ACCESS_TOKEN_DIR']
            if not access_tokens_dir.exists():
                logging.warning(f"Zerodha access tokens directory does not exist: {access_tokens_dir}")
                return None

            # List all token files and sort them by date descending
            token_files = sorted(
                [f for f in access_tokens_dir.glob("access_token_*.json")],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            if token_files:
                # New approach: Load from JSON with date validation
                latest_token_file = token_files[0]
                with open(latest_token_file, 'r') as f:
                    token_data = json.load(f)
                
                # Check token date
                token_date_str = token_data.get("token_date")
                if not token_date_str:
                    logging.warning("token_date not found in Zerodha token; forcing re-auth.")
                    return None
                
                # Get today's date in IST
                today_ist = datetime.now(IST).strftime("%Y-%m-%d")
                
                # Zerodha tokens expire at 6:00 AM IST the next day
                # So if the token is from today, it's valid
                if token_date_str != today_ist:
                    logging.warning(f"Zerodha token from {token_date_str} != today ({today_ist}). Forcing re-auth.")
                    return None
                
                access_token = token_data.get("access_token")
                if access_token:
                    logging.debug(f"Loaded Zerodha access token from {latest_token_file}")
                    return access_token
            
            # Fallback to legacy CSV approach
            legacy_file = ZERODHA_CONFIG['KEY_CSV_LOCATION']
            if legacy_file.exists():
                # Check file modification time
                token_mtime = datetime.fromtimestamp(legacy_file.stat().st_mtime)
                today_ist = datetime.now(IST).date()
                
                # If token was modified today, assume it's valid
                if token_mtime.date() == today_ist:
                    with open(legacy_file, 'r') as f:
                        access_token = f.read().strip()
                    
                    if access_token:
                        logging.warning("Using legacy Zerodha token file. Consider updating to JSON format.")
                        return access_token
                else:
                    logging.warning(f"Legacy Zerodha token from {token_mtime.date()} is not from today ({today_ist}).")
            
            logging.warning("No valid Zerodha access token found.")
            return None
            
        except Exception as e:
            logging.error(f"Error loading Zerodha access token: {e}")
            logging.debug(traceback.format_exc())
            return None

# === Live Trading Access Token Management (kept for backward compatibility) ===
#Mental note live works only on upstock we are using the same idea in both script live and backtester\
#But Backtester has different accesstoken and differnt access token for live for same broker an issue needs to be end to end handled

def save_access_token(token_data: dict):
    """
    Saves the access token to a file with the current date in IST.
    Adds a 'token_date' field to the token data for daily validation.

    Args:
        token_data: Dictionary containing access token information.
    """
    with _token_lock:
        try:
            # Ensure access tokens directory exists
            access_tokens_dir = LIVE_TRADING_CONFIG['ACCESS_TOKEN_DIR']
            access_tokens_dir.mkdir(parents=True, exist_ok=True)

            # Add 'token_date' to token_data
            token_date = datetime.now(IST).strftime("%Y-%m-%d")
            token_data["token_date"] = token_date
            
            # Add system and timestamp information
            token_data["system_id"] = SYSTEM_ID
            token_data["saved_at"] = datetime.now(IST).isoformat()

            # Determine filename
            date_str = token_date
            token_file = LIVE_TRADING_CONFIG['ACCESS_TOKEN_FILE_TEMPLATE'].format(date_str)
            
            # Write token to file
            with open(token_file, 'w') as f:
                json.dump(token_data, f, indent=2)
                
            logging.info(f"Access token saved to {token_file}")
            return True
        except Exception as e:
            logging.error(f"Failed to save access token: {e}")
            logging.debug(traceback.format_exc())
            return False


def load_latest_access_token() -> Optional[str]:
    """
    Loads the latest access token from the access_tokens directory.
    Checks if the token's date matches the current IST date.
    If not, returns None to force a token refresh.

    Returns:
        Access token string if valid, else None.
    """
    with _token_lock:
        try:
            # Ensure access tokens directory exists
            access_tokens_dir = LIVE_TRADING_CONFIG['ACCESS_TOKEN_DIR']
            if not access_tokens_dir.exists():
                logging.warning(f"Access tokens directory does not exist: {access_tokens_dir}")
                return None

            # List all token files and sort them by date descending
            token_files = sorted(
                [f for f in access_tokens_dir.glob("access_token_*.json")],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            if not token_files:
                logging.warning("No access token files found.")
                return None

            # Load the latest token file
            latest_token_file = token_files[0]
            with open(latest_token_file, 'r') as f:
                token_data = json.load(f)
                
            logging.debug(f"Loaded access token from {latest_token_file}")

            # Check if 'token_date' exists
            token_date_str = token_data.get("token_date")
            if not token_date_str:
                logging.warning("token_date not found in saved token; forcing re-auth.")
                return None

            # Get today's date in IST
            today_ist = datetime.now(IST).strftime("%Y-%m-%d")
            if token_date_str != today_ist:
                logging.warning(f"Token date {token_date_str} != today ({today_ist}). Forcing re-auth.")
                return None

            # If valid, return the access token
            access_token = token_data.get("access_token")
            if not access_token:
                logging.error("access_token not found in token data.")
                return None
                
            return access_token
            
        except Exception as e:
            logging.error(f"Error loading access token: {e}")
            logging.debug(traceback.format_exc())
            return None


# === Session Creation Functions ===

def create_upstox_session(access_token: str) -> requests.Session:
    """
    Creates a requests.Session with the provided Upstox access token for authenticated API calls.

    Args:
        access_token: OAuth2 access token.
        
    Returns:
        Authenticated requests.Session object.
    """
    try:
        session = requests.Session()
        
        # Set default headers
        session.headers.update({
            'Authorization': f'Bearer {access_token}',
            'Accept': 'application/json',
            'User-Agent': f'TradingSystem/{COMMON_CONFIG["VERSION"]} ({SYSTEM_INFO["platform"]})'
        })
        
        # Set default timeout
        session.request = lambda method, url, **kwargs: requests.Session.request(
            session, method, url, 
            timeout=kwargs.pop('timeout', UPSTOX_CONFIG['REQUEST_TIMEOUT']), 
            **kwargs
        )
        
        return session
    except Exception as e:
        logging.error(f"Error creating Upstox session: {e}")
        logging.debug(traceback.format_exc())
        raise


def create_session(access_token: str) -> requests.Session:
    """
    Creates a requests.Session with the provided access token for authenticated API calls.
    (Maintained for backward compatibility with live trading)

    Args:
        access_token: OAuth2 access token.
        
    Returns:
        Authenticated requests.Session object.
    """
    try:
        session = requests.Session()
        
        # Set default headers
        session.headers.update({
            'Authorization': f'Bearer {access_token}',
            'Accept': 'application/json',
            'User-Agent': f'TradingSystem/{COMMON_CONFIG["VERSION"]} ({SYSTEM_INFO["platform"]})'
        })
        
        # Set default timeout
        session.request = lambda method, url, **kwargs: requests.Session.request(
            session, method, url, 
            timeout=kwargs.pop('timeout', LIVE_TRADING_CONFIG['REQUEST_TIMEOUT']), 
            **kwargs
        )
        
        return session
    except Exception as e:
        logging.error(f"Error creating session: {e}")
        logging.debug(traceback.format_exc())
        raise


# === Utility functions ===

def get_provider_timeframe(standard_timeframe: str, provider: str = None) -> Optional[str]:
    """
    Convert a standard timeframe to provider-specific timeframe.
    
    Args:
        standard_timeframe: Standard timeframe (e.g. '1m', 'day')
        provider: Provider name ('upstox' or 'zerodha', defaults to configured provider)
        
    Returns:
        Provider-specific timeframe or None if not supported
    """
    if not provider:
        provider = BACKTESTER_CONFIG['DATA_PROVIDER']
        
    provider = provider.lower()
    
    if provider not in ['upstox', 'zerodha']:
        logging.error(f"Unsupported provider: {provider}")
        return None
        
    timeframe_map = BACKTESTER_CONFIG['STANDARD_TIMEFRAMES'].get(standard_timeframe, {})
    return timeframe_map.get(provider)


def get_standard_timeframe(provider_timeframe: str, provider: str = None) -> Optional[str]:
    """
    Convert a provider-specific timeframe to standard timeframe.
    
    Args:
        provider_timeframe: Provider-specific timeframe (e.g. '1minute', 'day')
        provider: Provider name ('upstox' or 'zerodha', defaults to configured provider)
        
    Returns:
        Standard timeframe or None if not found
    """
    if not provider:
        provider = BACKTESTER_CONFIG['DATA_PROVIDER']
        
    provider = provider.lower()
    
    if provider not in ['upstox', 'zerodha']:
        logging.error(f"Unsupported provider: {provider}")
        return None
        
    for standard_tf, providers in BACKTESTER_CONFIG['STANDARD_TIMEFRAMES'].items():
        if providers.get(provider) == provider_timeframe:
            return standard_tf
            
    return None


def get_config_value(key: str, default: Any = None) -> Any:
    """
    Get a configuration value from LIVE_TRADING_CONFIG, BACKTESTER_CONFIG, or COMMON_CONFIG.
    
    Args:
        key: Configuration key to look up
        default: Default value if key is not found
        
    Returns:
        Configuration value or default
    """
    if key in LIVE_TRADING_CONFIG:
        return LIVE_TRADING_CONFIG[key]
    elif key in BACKTESTER_CONFIG:
        return BACKTESTER_CONFIG[key]
    elif key in COMMON_CONFIG:
        return COMMON_CONFIG[key]
    else:
        return default


def update_config(config_dict: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration dictionary with new values.
    
    Args:
        config_dict: Configuration dictionary to update
        updates: Dictionary of updates
        
    Returns:
        Updated configuration dictionary
    """
    for key, value in updates.items():
        if isinstance(value, dict) and key in config_dict and isinstance(config_dict[key], dict):
            # Recursively update nested dictionaries
            config_dict[key] = update_config(config_dict[key], value)
        else:
            # Update simple values
            config_dict[key] = value
    return config_dict


def load_config_from_file(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        filepath: Path to JSON configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading configuration from {filepath}: {e}")
        return {}


def save_config_to_file(config_dict: Dict[str, Any], filepath: str) -> bool:
    """
    Save configuration to a JSON file.
    
    Args:
        config_dict: Configuration dictionary
        filepath: Path to save JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Write configuration to file
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        return True
    except Exception as e:
        logging.error(f"Error saving configuration to {filepath}: {e}")
        return False


# Note: Directories are created on-demand by functions that need them
# Removed automatic creation on import to avoid unwanted directory creation

# For standalone testing and validation
if __name__ == "__main__":
    # Initialize logging
    logger = setup_logging("config_test")
    
    # Print system information
    logger.info("System Information:")
    for key, value in SYSTEM_INFO.items():
        logger.info(f"  {key}: {value}")
    
    # Verify directories
    logger.info("Checking directories:")
    for key, directory in {
        "BACKTESTER_DATA_POOL_DIR": BACKTESTER_CONFIG['DATA_POOL_DIR'],
        "BACKTESTER_OUTPUT_FOLDER": BACKTESTER_CONFIG['OUTPUT_FOLDER'],
        "BACKTESTER_LOG_DIR": BACKTESTER_CONFIG['LOG_DIR'],
        "LIVE_TRADING_DATA_DIR": LIVE_TRADING_CONFIG['DATA_DIR'],
        "LIVE_TRADING_LOG_DIR": LIVE_TRADING_CONFIG['LOG_DIR'],
        "UPSTOX_ACCESS_TOKEN_DIR": UPSTOX_CONFIG['ACCESS_TOKEN_DIR'],
        "ZERODHA_ACCESS_TOKEN_DIR": ZERODHA_CONFIG['ACCESS_TOKEN_DIR']
    }.items():
        logger.info(f"  {key}: {directory} (Exists: {directory.exists()})")
    
    # Test timeframe mapping
    logger.info("Testing timeframe mapping:")
    for std_tf in BACKTESTER_CONFIG['SUPPORTED_TIMEFRAMES']:
        upstox_tf = get_provider_timeframe(std_tf, 'upstox')
        zerodha_tf = get_provider_timeframe(std_tf, 'zerodha')
        logger.info(f"  {std_tf}: Upstox={upstox_tf}, Zerodha={zerodha_tf}")
    
    logger.info("Configuration validation completed successfully.")