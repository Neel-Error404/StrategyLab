"""
Helper utilities for the backtesting framework.

This module provides common utility functions used across the backtesting system.
"""

import logging
import json
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd


def ensure_dir_exists(directory: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_file(data: Dict[str, Any], file_path: Union[str, Path], pretty: bool = True) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save file
        pretty: Whether to format the JSON for readability
        
    Raises:
        IOError: If file can't be written
    """
    ensure_dir_exists(Path(file_path).parent)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        if pretty:
            json.dump(data, f, indent=2, sort_keys=True)
        else:
            json.dump(data, f)


def load_yaml_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML file.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Loaded YAML data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml_file(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save data to a YAML file.
    
    Args:
        data: Data to save
        file_path: Path to save file
        
    Raises:
        IOError: If file can't be written
    """
    ensure_dir_exists(Path(file_path).parent)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False)


def configure_logging(log_file: Optional[Union[str, Path]] = None, 
                     log_level: int = logging.INFO,
                     log_format: str = '%(asctime)s [%(levelname)s] %(name)s: %(message)s') -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        log_file: Optional path to log file
        log_level: Logging level
        log_format: Format for log messages
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("backtester")
    logger.setLevel(log_level)
    
    # Clear any existing handlers to prevent duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file provided)
    if log_file:
        file_path = Path(log_file)
        ensure_dir_exists(file_path.parent)
        
        file_handler = logging.FileHandler(str(file_path))
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def convert_numpy_types(obj: Any) -> Any:
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object that might contain numpy types
        
    Returns:
        Object with numpy types converted to Python native types
    """
    import numpy as np
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (dict, pd.Series)):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    else:
        return obj
