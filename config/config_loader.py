# config/config_loader.py
"""
Configuration Loader with Environment Variable Substitution

This module provides functionality to load YAML configuration files and substitute
environment variable placeholders (e.g., ${UPSTOX_CLIENT_ID}) with actual values.

Key Features:
- Load YAML configs with environment variable substitution
- Support for default values: ${VAR_NAME:default_value}
- Recursive substitution for nested dictionaries
- Integration with BacktestConfig dataclass from unified_config.py
- Backward compatibility with existing config.py structure
"""

import os
import re
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Union
from config.unified_config import BacktestConfig

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and process configuration files with environment variable substitution."""
    
    # Pattern to match  or 
    ENV_VAR_PATTERN = re.compile(r'\$\{([^}:]+)(?::([^}]*))?\}')
    
    @classmethod
    def substitute_env_vars(cls, value: Any) -> Any:
        """
        Recursively substitute environment variables in configuration values.
        
        Supports:
        - Simple: ${UPSTOX_CLIENT_ID}  value from environment
        - With default: ${UPSTOX_CLIENT_ID:default_value}  default if not set
        
        Args:
            value: Configuration value (str, dict, list, or primitive)
            
        Returns:
            Value with environment variables substituted
        """
        if isinstance(value, str):
            def replacer(match):
                var_name = match.group(1)
                default_value = match.group(2) if match.group(2) is not None else ''
                return os.getenv(var_name, default_value)
            
            return cls.ENV_VAR_PATTERN.sub(replacer, value)
        
        elif isinstance(value, dict):
            return {k: cls.substitute_env_vars(v) for k, v in value.items()}
        
        elif isinstance(value, list):
            return [cls.substitute_env_vars(item) for item in value]
        
        else:
            return value
    
    @classmethod
    def load_yaml(cls, yaml_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load YAML file and substitute environment variables.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            Dictionary with env vars substituted
            
        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f'Configuration file not found: {yaml_path}')
        
        logger.info(f'Loading configuration from: {yaml_path}')
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
            
            if raw_config is None:
                raise ValueError(f'Empty configuration file: {yaml_path}')
            
            # Substitute environment variables
            processed_config = cls.substitute_env_vars(raw_config)
            
            logger.debug(f'Configuration loaded successfully from {yaml_path}')
            return processed_config
        
        except yaml.YAMLError as e:
            logger.error(f'YAML parsing error in {yaml_path}: {e}')
            raise
        except Exception as e:
            logger.error(f'Error loading configuration from {yaml_path}: {e}')
            raise
    
    @classmethod
    def load_config(cls, config_path: Union[str, Path] = None, 
                   template: str = None) -> BacktestConfig:
        """
        Load BacktestConfig from YAML file with environment variable substitution.
        
        Priority:
        1. config_path (explicit YAML file)
        2. template (uses config/templates/{template}.yaml)
        3. Default conservative template
        
        Args:
            config_path: Path to custom YAML configuration file
            template: Template name (conservative, aggressive, minimal, etc.)
            
        Returns:
            BacktestConfig instance with all environment variables substituted
        """
        # Determine which config file to load
        base_dir = Path(__file__).resolve().parent.parent
        
        if config_path:
            yaml_path = Path(config_path)
        elif template:
            yaml_path = base_dir / 'config' / 'templates' / f'{template}.yaml'
        else:
            yaml_path = base_dir / 'config' / 'templates' / 'conservative.yaml'
            logger.info('No config specified, using default conservative template')
        
        # Load and process YAML
        config_dict = cls.load_yaml(yaml_path)
        
        # Load environment variables for broker credentials
        cls._inject_broker_env_vars(config_dict)
        
        # Create BacktestConfig from processed dictionary
        config = BacktestConfig.from_dict(config_dict)
        
        logger.info(f'Configuration loaded: {config.strategy.risk_profile} profile')
        return config
    
    @classmethod
    def _inject_broker_env_vars(cls, config_dict: Dict[str, Any]):
        """
        Inject broker credentials from environment variables into config dict.
        
        This ensures broker credentials are always loaded from environment,
        even if not explicitly specified in YAML.
        
        Args:
            config_dict: Configuration dictionary (modified in-place)
        """
        if 'broker' not in config_dict:
            config_dict['broker'] = {}
        
        broker_config = config_dict['broker']
        
        # Upstox credentials
        broker_config['upstox_client_id'] = os.getenv(
            'UPSTOX_CLIENT_ID', 
            broker_config.get('upstox_client_id', '')
        )
        broker_config['upstox_client_secret'] = os.getenv(
            'UPSTOX_CLIENT_SECRET',
            broker_config.get('upstox_client_secret', '')
        )
        broker_config['upstox_redirect_uri'] = os.getenv(
            'UPSTOX_REDIRECT_URI',
            broker_config.get('upstox_redirect_uri', 'https://127.0.0.1:5000/')
        )
        
        # Zerodha credentials
        broker_config['zerodha_api_key'] = os.getenv(
            'ZERODHA_API_KEY',
            broker_config.get('zerodha_api_key', '')
        )
        broker_config['zerodha_api_secret'] = os.getenv(
            'ZERODHA_API_SECRET',
            broker_config.get('zerodha_api_secret', '')
        )
        broker_config['zerodha_redirect_uri'] = os.getenv(
            'ZERODHA_REDIRECT_URI',
            broker_config.get('zerodha_redirect_uri', 'https://127.0.0.1:5000/')
        )
        
        # Binance credentials (optional)
        broker_config['binance_api_key'] = os.getenv(
            'BINANCE_API_KEY',
            broker_config.get('binance_api_key', '')
        )
        broker_config['binance_api_secret'] = os.getenv(
            'BINANCE_API_SECRET',
            broker_config.get('binance_api_secret', '')
        )
        
        logger.debug('Broker credentials injected from environment variables')


def load_config(config_path: str = None, template: str = None) -> BacktestConfig:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to custom YAML configuration file
        template: Template name (conservative, aggressive, minimal, etc.)
        
    Returns:
        BacktestConfig instance
    """
    return ConfigLoader.load_config(config_path, template)


if __name__ == '__main__':
    # Test the config loader
    logging.basicConfig(level=logging.DEBUG)
    
    print('Testing ConfigLoader with conservative template:')
    config = load_config(template='conservative')
    
    print(f'\nStrategy: {config.strategy.name}')
    print(f'Risk Profile: {config.strategy.risk_profile}')
    print(f'Default Provider: {config.broker.default_provider}')
    print(f'Upstox Client ID: {config.broker.upstox_client_id[:20]}...' if config.broker.upstox_client_id else 'Not set')
    print(f'Max Position Size: {config.risk.max_position_size}')
    print('\nConfig loader test complete!')
