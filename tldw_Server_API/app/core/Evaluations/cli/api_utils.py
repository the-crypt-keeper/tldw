"""
Utilities for API discovery and configuration in the evaluation CLI.

This module provides helper functions to discover available LLM APIs
and their configuration status from the tldw config.
"""

from typing import Dict, Any, List, Tuple, Optional
from tldw_Server_API.app.core.config import load_and_log_configs
from loguru import logger


# Mapping of config keys to friendly API names that match chat_api_call endpoints
API_CONFIG_MAPPING = {
    'anthropic_api': 'anthropic',
    'cohere_api': 'cohere',
    'deepseek_api': 'deepseek',
    'google_api': 'google',
    'groq_api': 'groq',
    'huggingface_api': 'huggingface',
    'mistral_api': 'mistral',
    'openrouter_api': 'openrouter',
    'openai_api': 'openai',
    'llama_api': 'llama.cpp',
    'ooba_api': 'ooba',
    'kobold_api': 'kobold',
    'tabby_api': 'tabbyapi',
    'vllm_api': 'vllm',
    'ollama_api': 'ollama',
    'aphrodite_api': 'aphrodite',
    'custom_openai_api': 'custom-openai-api',
    'custom_openai2_api': 'custom-openai-api-2'
}

# API categories for better organization
API_CATEGORIES = {
    'Commercial': ['openai', 'anthropic', 'cohere', 'google', 'groq', 
                   'huggingface', 'mistral', 'openrouter', 'deepseek'],
    'Self-Hosted': ['llama.cpp', 'ooba', 'kobold', 'tabbyapi', 'vllm', 
                    'ollama', 'aphrodite'],
    'Custom': ['custom-openai-api', 'custom-openai-api-2']
}


def get_available_apis() -> Dict[str, Dict[str, Any]]:
    """
    Get all available APIs from configuration with their status.
    
    Returns:
        Dict mapping API names to their configuration and status
    """
    try:
        config = load_and_log_configs()
        if not config:
            logger.warning("No configuration loaded")
            return {}
        
        available_apis = {}
        
        for config_key, api_name in API_CONFIG_MAPPING.items():
            if config_key in config:
                api_config = config[config_key]
                
                # Check if API is configured (has API key or endpoint)
                is_configured = False
                config_status = []
                
                # Check for API key
                if 'api_key' in api_config:
                    if api_config['api_key'] and api_config['api_key'] not in [None, '', 'None']:
                        is_configured = True
                        config_status.append('API key set')
                    else:
                        config_status.append('API key missing')
                
                # Check for API endpoint (for self-hosted)
                if 'api_ip' in api_config or 'api_url' in api_config:
                    endpoint = api_config.get('api_ip') or api_config.get('api_url')
                    if endpoint and endpoint not in [None, '', 'None']:
                        is_configured = True
                        config_status.append(f'Endpoint: {endpoint}')
                    else:
                        config_status.append('Endpoint missing')
                
                # Get model info
                model = api_config.get('model', 'Not specified')
                
                # Determine category
                category = 'Unknown'
                for cat, apis in API_CATEGORIES.items():
                    if api_name in apis:
                        category = cat
                        break
                
                available_apis[api_name] = {
                    'configured': is_configured,
                    'model': model,
                    'category': category,
                    'status': ', '.join(config_status) if config_status else 'Not configured',
                    'config': api_config
                }
        
        return available_apis
    
    except Exception as e:
        logger.error(f"Error loading API configurations: {e}")
        return {}


def validate_api_config(api_name: str) -> Tuple[bool, str]:
    """
    Validate if an API is properly configured for use.
    
    Args:
        api_name: Name of the API to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    apis = get_available_apis()
    
    if api_name not in apis:
        available = ', '.join(sorted(apis.keys()))
        return False, f"Unknown API '{api_name}'. Available: {available}"
    
    api_info = apis[api_name]
    
    if not api_info['configured']:
        return False, f"API '{api_name}' is not properly configured. {api_info['status']}"
    
    return True, ""


def get_api_model(api_name: str, model_override: Optional[str] = None) -> str:
    """
    Get the model to use for a specific API.
    
    Args:
        api_name: Name of the API
        model_override: Optional model override
        
    Returns:
        Model string to use
    """
    if model_override:
        return model_override
    
    apis = get_available_apis()
    if api_name in apis:
        model = apis[api_name].get('config', {}).get('model')
        if model and model not in [None, '', 'None', 'Not specified']:
            return model
    
    # Return API-specific defaults if no model configured
    api_defaults = {
        'openai': 'gpt-4',
        'anthropic': 'claude-3-5-sonnet-20240620',
        'google': 'gemini-1.5-pro',
        'cohere': 'command-r-plus',
        'deepseek': 'deepseek-chat',
        'groq': 'llama3-70b-8192',
        'mistral': 'mistral-large-latest',
        'openrouter': 'microsoft/wizardlm-2-8x22b'
    }
    
    return api_defaults.get(api_name, 'default')


def get_configured_apis() -> List[str]:
    """
    Get list of APIs that are properly configured and ready to use.
    
    Returns:
        List of configured API names
    """
    apis = get_available_apis()
    return [name for name, info in apis.items() if info['configured']]


def get_default_api() -> Optional[str]:
    """
    Get the default API to use (first configured API or from config).
    
    Returns:
        Default API name or None if no APIs configured
    """
    # Check if there's a default set in config
    config = load_and_log_configs()
    if config and 'default_api' in config:
        default = config.get('default_api')
        if default and validate_api_config(default)[0]:
            return default
    
    # Otherwise return first configured API
    configured = get_configured_apis()
    return configured[0] if configured else None


def format_api_info(api_name: str, detailed: bool = False) -> str:
    """
    Format API information for display.
    
    Args:
        api_name: Name of the API
        detailed: Whether to include detailed configuration
        
    Returns:
        Formatted string with API information
    """
    apis = get_available_apis()
    if api_name not in apis:
        return f"Unknown API: {api_name}"
    
    info = apis[api_name]
    status_symbol = "✓" if info['configured'] else "✗"
    
    result = f"{status_symbol} {api_name} ({info['category']})\n"
    result += f"  Model: {info['model']}\n"
    result += f"  Status: {info['status']}\n"
    
    if detailed and info['configured']:
        result += "  Configuration:\n"
        for key, value in info['config'].items():
            if key == 'api_key' and value:
                # Mask API key for security
                display_value = "***" + value[-4:] if len(value) > 4 else "***"
            else:
                display_value = str(value)
            result += f"    {key}: {display_value}\n"
    
    return result