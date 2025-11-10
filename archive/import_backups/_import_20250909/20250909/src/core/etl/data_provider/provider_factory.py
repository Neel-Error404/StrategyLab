# src/etl/data_providers/provider_factory.py
from typing import Dict, Any, Optional
import logging

from .base_provider import DataProvider
from .upstox_provider import UpstoxDataProvider
from .zerodha_provider import ZerodhaDataProvider
from ..token_manager import get_token_manager, detect_available_provider


class DataProviderFactory:
    """Enhanced factory for creating data provider instances with token management."""
    
    _providers = {
        'upstox': UpstoxDataProvider,
        'zerodha': ZerodhaDataProvider
    }
    
    @classmethod
    def get_provider(cls, provider_name: str = None, config: Dict[str, Any] = None, 
                     auto_detect: bool = False) -> Optional[DataProvider]:
        """
        Get a data provider instance by name with enhanced token management.

        Args:
            provider_name: Name of the data provider (optional if auto_detect=True)
            config: Configuration dictionary for the provider (optional)
            auto_detect: If True, automatically detect provider with valid tokens

        Returns:
            DataProvider instance or None if provider not found
        """
        logger = logging.getLogger("DataProviderFactory")
        token_manager = get_token_manager()
        
        # Auto-detect provider if requested and no provider specified
        if auto_detect and not provider_name:
            provider_name = detect_available_provider()
            if provider_name:
                logger.info(f"Auto-detected provider with valid tokens: {provider_name}")
            else:
                logger.warning("No provider with valid tokens found for auto-detection")
                return None
        
        if not provider_name:
            logger.error("No provider name specified and auto-detection disabled")
            return None
        
        provider_name = provider_name.lower()
        provider_class = cls._providers.get(provider_name)

        if not provider_class:
            logger.error(f"Data provider '{provider_name}' not found.")
            available = ", ".join(cls._providers.keys())
            logger.error(f"Available providers: {available}")
            return None

        # Use default config from the system if not provided
        if config is None:
            from config import UPSTOX_CONFIG, ZERODHA_CONFIG
            config_map = {
                'upstox': UPSTOX_CONFIG,
                'zerodha': ZERODHA_CONFIG
            }
            config = config_map.get(provider_name, {})

        try:
            provider = provider_class(config)
            
            # Validate token availability
            if not token_manager.validate_token(provider_name):
                logger.warning(f"No valid token found for {provider_name}. "
                             f"Authentication will be required during first use.")
            else:
                logger.info(f"Valid token found for {provider_name}")
            
            return provider
            
        except Exception as e:
            logger.error(f"Error creating data provider '{provider_name}': {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    @classmethod
    def get_provider_with_fallback(cls, preferred_provider: str = None, 
                                  config: Dict[str, Any] = None) -> Optional[DataProvider]:
        """
        Get a data provider with automatic fallback to any available provider.
        
        Args:
            preferred_provider: Preferred provider name (optional)
            config: Configuration dictionary (optional)
        
        Returns:
            DataProvider instance or None if no provider available
        """
        logger = logging.getLogger("DataProviderFactory")
        
        # Try preferred provider first if specified
        if preferred_provider:
            provider = cls.get_provider(preferred_provider, config)
            if provider:
                return provider
        
        # Try auto-detection
        provider = cls.get_provider(auto_detect=True, config=config)
        if provider:
            return provider
        
        # Fallback to any available provider (will require authentication)
        for provider_name in cls._providers.keys():
            try:
                provider = cls.get_provider(provider_name, config)
                if provider:
                    logger.info(f"Using fallback provider: {provider_name}")
                    return provider
            except Exception as e:
                logger.debug(f"Failed to create fallback provider {provider_name}: {e}")
        
        logger.error("No data provider could be created")
        return None
    
    @classmethod
    def list_providers(cls) -> Dict[str, str]:
        """
        List all available data providers with their token status.
        
        Returns:
            Dictionary mapping provider names to their status information
        """
        token_manager = get_token_manager()
        providers_info = {}
        
        for provider_name in cls._providers.keys():
            provider_class = cls._providers[provider_name]
            has_token = token_manager.validate_token(provider_name)
            token_info = token_manager.get_token_info(provider_name)
            
            status = "Ready" if has_token else "Needs Authentication"
            if token_info:
                user_info = token_info.get('user_name', 'Unknown')
                status += f" (User: {user_info})"
            
            providers_info[provider_name] = {
                'class': provider_class.__name__,
                'status': status,
                'has_token': has_token,
                'token_info': token_info
            }
        
        return providers_info
    
    @classmethod
    def clear_provider_tokens(cls, provider_name: str = None) -> bool:
        """
        Clear tokens for specified provider or all providers.
        
        Args:
            provider_name: Provider to clear tokens for (None = all providers)
        
        Returns:
            True if successful, False otherwise
        """
        logger = logging.getLogger("DataProviderFactory")
        token_manager = get_token_manager()
        
        if provider_name:
            # Clear specific provider
            if provider_name.lower() not in cls._providers:
                logger.error(f"Unknown provider: {provider_name}")
                return False
            
            result = token_manager.clear_token(provider_name.lower())
            if result:
                logger.info(f"Cleared tokens for {provider_name}")
            return result
        else:
            # Clear all providers
            success_count = 0
            for provider in cls._providers.keys():
                if token_manager.clear_token(provider):
                    success_count += 1
                    logger.info(f"Cleared tokens for {provider}")
            
            logger.info(f"Cleared tokens for {success_count}/{len(cls._providers)} providers")
            return success_count > 0
        """
        List all available data providers.
        
        Returns:
            Dictionary mapping provider names to their class names
        """
        return {name: provider.__name__ for name, provider in cls._providers.items()}