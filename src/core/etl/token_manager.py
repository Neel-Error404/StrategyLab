#!/usr/bin/env python3
"""
Provider-Aware Token Management System

This module provides a centralized, provider-aware authentication token management
system that automatically handles token storage, retrieval, and validation for
different broker providers (Upstox, Zerodha, etc.).

Key Features:
- Automatic provider detection and routing
- Provider-specific directory management
- Unified interface across all providers
- Token validation and expiration handling
- Thread-safe operations
- Production-ready error handling

Author: Production Trading System
Created: 2025-06-07
"""

import logging
import threading
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import json

# Import provider-specific token functions
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from config.config import (
    save_upstox_access_token, load_upstox_access_token,
    save_zerodha_access_token, load_zerodha_access_token,
    UPSTOX_CONFIG, ZERODHA_CONFIG, BASE_DIR
)


class ProviderTokenManager:
    """
    Centralized token management system that automatically routes
    token operations to the correct provider-specific functions.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._lock = threading.Lock()
        
        # Provider configurations mapping
        self.provider_configs = {
            'upstox': UPSTOX_CONFIG,
            'zerodha': ZERODHA_CONFIG
        }
        
        # Provider-specific token functions
        self.token_savers = {
            'upstox': save_upstox_access_token,
            'zerodha': save_zerodha_access_token
        }
        
        self.token_loaders = {
            'upstox': load_upstox_access_token,
            'zerodha': load_zerodha_access_token
        }
        
        self._ensure_token_directories()
    
    def _ensure_token_directories(self):
        """Ensure all provider token directories exist."""
        with self._lock:
            try:
                for provider, config in self.provider_configs.items():
                    token_dir = Path(config.get('ACCESS_TOKEN_DIR'))
                    token_dir.mkdir(parents=True, exist_ok=True)
                    self.logger.debug(f"Ensured token directory exists: {token_dir}")
            except Exception as e:
                self.logger.error(f"Error creating token directories: {e}")
                raise
    
    def save_token(self, provider: str, token_data: Any, **kwargs) -> bool:
        """
        Save authentication token for the specified provider.
        
        Args:
            provider: Provider name ('upstox', 'zerodha')
            token_data: Token data (format varies by provider)
            **kwargs: Additional arguments for provider-specific saving
        
        Returns:
            True if successful, False otherwise
        """
        provider = provider.lower()
        
        if provider not in self.token_savers:
            self.logger.error(f"Unsupported provider: {provider}")
            return False
        
        with self._lock:
            try:
                saver_func = self.token_savers[provider]
                
                # Handle provider-specific token saving
                if provider == 'upstox':
                    # Upstox expects a dictionary
                    if isinstance(token_data, dict):
                        result = saver_func(token_data)
                    else:
                        self.logger.error("Upstox token_data must be a dictionary")
                        return False
                        
                elif provider == 'zerodha':
                    # Zerodha expects access_token string and optional kwargs
                    if isinstance(token_data, str):
                        result = saver_func(
                            token_data, 
                            kwargs.get('request_token'), 
                            kwargs.get('api_key')
                        )
                    elif isinstance(token_data, dict):
                        # Extract from dict if needed
                        access_token = token_data.get('access_token')
                        if access_token:
                            result = saver_func(
                                access_token,
                                kwargs.get('request_token') or token_data.get('request_token'),
                                kwargs.get('api_key') or token_data.get('api_key')
                            )
                        else:
                            self.logger.error("No access_token found in Zerodha token_data")
                            return False
                    else:
                        self.logger.error("Zerodha token_data must be string or dict")
                        return False
                
                self.logger.info(f"Successfully saved {provider} access token")
                return result if isinstance(result, bool) else True
                
            except Exception as e:
                self.logger.error(f"Error saving {provider} token: {e}")
                return False
    
    def load_token(self, provider: str) -> Optional[str]:
        """
        Load authentication token for the specified provider.
        
        Args:
            provider: Provider name ('upstox', 'zerodha')
        
        Returns:
            Access token string if found and valid, None otherwise
        """
        provider = provider.lower()
        
        if provider not in self.token_loaders:
            self.logger.error(f"Unsupported provider: {provider}")
            return None
        
        with self._lock:
            try:
                loader_func = self.token_loaders[provider]
                token = loader_func()
                
                if token:
                    self.logger.debug(f"Successfully loaded {provider} access token")
                else:
                    self.logger.warning(f"No valid {provider} access token found")
                
                return token
                
            except Exception as e:
                self.logger.error(f"Error loading {provider} token: {e}")
                return None
    
    def clear_token(self, provider: str) -> bool:
        """
        Clear/delete authentication token for the specified provider.
        
        Args:
            provider: Provider name ('upstox', 'zerodha')
        
        Returns:
            True if successful, False otherwise
        """
        provider = provider.lower()
        
        if provider not in self.provider_configs:
            self.logger.error(f"Unsupported provider: {provider}")
            return False
        
        with self._lock:
            try:
                config = self.provider_configs[provider]
                token_dir = Path(config.get('ACCESS_TOKEN_DIR'))
                
                # Remove all token files for the provider
                files_removed = 0
                if token_dir.exists():
                    # Remove JSON files
                    for json_file in token_dir.glob("access_token_*.json"):
                        json_file.unlink()
                        files_removed += 1
                        self.logger.info(f"Removed {provider} token file: {json_file}")
                    
                    # Remove legacy CSV files (mainly for Zerodha)
                    if provider == 'zerodha':
                        csv_file = Path(config.get('KEY_CSV_LOCATION'))
                        if csv_file.exists():
                            csv_file.unlink()
                            files_removed += 1
                            self.logger.info(f"Removed {provider} legacy token file: {csv_file}")
                
                if files_removed > 0:
                    self.logger.info(f"Cleared {files_removed} {provider} token file(s)")
                else:
                    self.logger.warning(f"No {provider} token files found to clear")
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error clearing {provider} token: {e}")
                return False
    
    def validate_token(self, provider: str) -> bool:
        """
        Validate if the current token for the provider is valid and not expired.
        
        Args:
            provider: Provider name ('upstox', 'zerodha')
        
        Returns:
            True if token is valid, False otherwise
        """
        token = self.load_token(provider)
        return token is not None
    
    def list_available_tokens(self) -> Dict[str, List[str]]:
        """
        List all available token files for each provider.
        
        Returns:
            Dictionary mapping provider names to lists of token file paths
        """
        result = {}
        
        for provider, config in self.provider_configs.items():
            token_files = []
            token_dir = Path(config.get('ACCESS_TOKEN_DIR'))
            
            if token_dir.exists():
                # Find JSON token files
                for json_file in token_dir.glob("access_token_*.json"):
                    token_files.append(str(json_file))
                
                # Find legacy CSV files (mainly for Zerodha)
                if provider == 'zerodha':
                    csv_file = Path(config.get('KEY_CSV_LOCATION'))
                    if csv_file.exists():
                        token_files.append(str(csv_file))
            
            result[provider] = token_files
        
        return result
    
    def detect_provider_from_token_dir(self) -> Optional[str]:
        """
        Detect which provider has valid tokens available.
        
        Returns:
            Provider name if tokens found, None otherwise
        """
        for provider in self.provider_configs.keys():
            if self.validate_token(provider):
                self.logger.info(f"Detected valid tokens for provider: {provider}")
                return provider
        
        self.logger.warning("No valid tokens found for any provider")
        return None
    
    def get_token_info(self, provider: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about the token for the specified provider.
        
        Args:
            provider: Provider name ('upstox', 'zerodha')
        
        Returns:
            Dictionary with token information or None if not found
        """
        provider = provider.lower()
        
        if provider not in self.provider_configs:
            return None
        
        config = self.provider_configs[provider]
        token_dir = Path(config.get('ACCESS_TOKEN_DIR'))
        
        # Find the latest token file
        token_files = list(token_dir.glob("access_token_*.json")) if token_dir.exists() else []
        
        if not token_files:
            return None
        
        # Sort by modification time, get the latest
        latest_file = max(token_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                token_data = json.load(f)
            
            return {
                'provider': provider,
                'token_file': str(latest_file),
                'token_date': token_data.get('token_date'),
                'saved_at': token_data.get('saved_at'),
                'system_id': token_data.get('system_id'),
                'user_id': token_data.get('user_id'),
                'user_name': token_data.get('user_name', token_data.get('user_type')),
                'expires': token_data.get('expires_in', 'Unknown')
            }
            
        except Exception as e:
            self.logger.error(f"Error reading token info for {provider}: {e}")
            return None


# Global token manager instance
_token_manager = None
_manager_lock = threading.Lock()


def get_token_manager() -> ProviderTokenManager:
    """
    Get the global token manager instance (singleton pattern).
    
    Returns:
        ProviderTokenManager instance
    """
    global _token_manager
    
    with _manager_lock:
        if _token_manager is None:
            _token_manager = ProviderTokenManager()
        return _token_manager


# Convenience functions for easy integration
def save_provider_token(provider: str, token_data: Any, **kwargs) -> bool:
    """Convenience function to save a provider token."""
    return get_token_manager().save_token(provider, token_data, **kwargs)


def load_provider_token(provider: str) -> Optional[str]:
    """Convenience function to load a provider token."""
    return get_token_manager().load_token(provider)


def clear_provider_token(provider: str) -> bool:
    """Convenience function to clear a provider token."""
    return get_token_manager().clear_token(provider)


def validate_provider_token(provider: str) -> bool:
    """Convenience function to validate a provider token."""
    return get_token_manager().validate_token(provider)


def detect_available_provider() -> Optional[str]:
    """Convenience function to detect which provider has valid tokens."""
    return get_token_manager().detect_provider_from_token_dir()


if __name__ == "__main__":
    # Test the token manager
    import sys
    logging.basicConfig(level=logging.INFO)
    
    manager = get_token_manager()
    
    print("Available token files:")
    tokens = manager.list_available_tokens()
    for provider, files in tokens.items():
        print(f"  {provider}: {files}")
    
    print("\nProvider with valid tokens:")
    provider = manager.detect_provider_from_token_dir()
    if provider:
        print(f"  {provider}")
        info = manager.get_token_info(provider)
        if info:
            print(f"  Token info: {info}")
    else:
        print("  None")
