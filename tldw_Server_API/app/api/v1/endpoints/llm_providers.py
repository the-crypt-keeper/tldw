# llm_providers.py
# Description: API endpoints for managing LLM providers and models
#
# Imports
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from loguru import logger
from tldw_Server_API.app.core.config import load_comprehensive_config

#######################################################################################################################
#
# Functions:

router = APIRouter()

def parse_model_string(model_value: str) -> List[str]:
    """
    Parse a model string which could be a single model or comma-separated list.
    
    Args:
        model_value: Model string from config
        
    Returns:
        List of model names
    """
    if not model_value:
        return []
    
    # Handle comma-separated lists
    if ',' in model_value:
        return [m.strip() for m in model_value.split(',') if m.strip()]
    
    # Single model
    return [model_value.strip()] if model_value.strip() else []

def get_configured_providers() -> Dict[str, Any]:
    """
    Get list of configured LLM providers with their models from the config file.
    
    Returns:
        Dictionary containing provider information
    """
    try:
        config_parser = load_comprehensive_config()
        providers = []
        
        # Check if we have the required sections
        if not config_parser.has_section('API') and not config_parser.has_section('Local-API'):
            logger.warning("No API or Local-API sections found in config")
            return {
                'providers': [],
                'default_provider': None,
                'total_configured': 0,
                'message': 'No API configuration sections found in config.txt'
            }
        
        # Define provider mappings with their config keys
        provider_mappings = {
            # Commercial APIs (from API section)
            'openai': {
                'display_name': 'OpenAI',
                'api_key_field': 'openai_api_key',
                'model_field': 'openai_model',
                'type': 'commercial',
                'section': 'API'
            },
            'anthropic': {
                'display_name': 'Anthropic', 
                'api_key_field': 'anthropic_api_key',
                'model_field': 'anthropic_model',
                'type': 'commercial',
                'section': 'API'
            },
            'cohere': {
                'display_name': 'Cohere',
                'api_key_field': 'cohere_api_key',
                'model_field': 'cohere_model',
                'type': 'commercial',
                'section': 'API'
            },
            'deepseek': {
                'display_name': 'DeepSeek',
                'api_key_field': 'deepseek_api_key',
                'model_field': 'deepseek_model',
                'type': 'commercial',
                'section': 'API'
            },
            'google': {
                'display_name': 'Google',
                'api_key_field': 'google_api_key',
                'model_field': 'google_model',
                'type': 'commercial',
                'section': 'API'
            },
            'groq': {
                'display_name': 'Groq',
                'api_key_field': 'groq_api_key',
                'model_field': 'groq_model',
                'type': 'commercial',
                'section': 'API'
            },
            'mistral': {
                'display_name': 'Mistral',
                'api_key_field': 'mistral_api_key',
                'model_field': 'mistral_model',
                'type': 'commercial',
                'section': 'API'
            },
            'huggingface': {
                'display_name': 'HuggingFace',
                'api_key_field': 'huggingface_api_key',
                'model_field': 'huggingface_model',
                'type': 'commercial',
                'section': 'API'
            },
            'openrouter': {
                'display_name': 'OpenRouter',
                'api_key_field': 'openrouter_api_key',
                'model_field': 'openrouter_model',
                'type': 'commercial',
                'section': 'API'
            },
            # Local APIs (from Local-API section)
            'llama': {
                'display_name': 'Llama.cpp',
                'endpoint_field': 'llama_api_IP',
                'model_field': None,  # No model field in config
                'type': 'local',
                'section': 'Local-API'
            },
            'kobold': {
                'display_name': 'Kobold.cpp',
                'endpoint_field': 'kobold_api_IP',
                'model_field': None,  # No model field in config
                'type': 'local',
                'section': 'Local-API'
            },
            'ooba': {
                'display_name': 'Oobabooga',
                'endpoint_field': 'ooba_api_IP',
                'model_field': None,  # No model field in config
                'type': 'local',
                'section': 'Local-API'
            },
            'tabby': {
                'display_name': 'TabbyAPI',
                'endpoint_field': 'tabby_api_IP',
                'model_field': None,  # No model field in config
                'type': 'local',
                'section': 'Local-API'
            },
            'vllm': {
                'display_name': 'vLLM',
                'endpoint_field': 'vllm_api_IP',
                'model_field': 'vllm_model',
                'type': 'local',
                'section': 'Local-API'
            },
            'ollama': {
                'display_name': 'Ollama',
                'endpoint_field': 'ollama_api_IP',
                'model_field': 'ollama_model',
                'type': 'local',
                'section': 'Local-API'
            },
            'aphrodite': {
                'display_name': 'Aphrodite',
                'endpoint_field': 'aphrodite_api_IP',
                'model_field': 'aphrodite_model',
                'type': 'local',
                'section': 'Local-API'
            },
            'custom_openai_api': {
                'display_name': 'Custom OpenAI API',
                'endpoint_field': 'custom_openai_api_ip',
                'model_field': 'custom_openai_api_model',
                'type': 'local',
                'section': 'API'
            }
        }
        
        # Process each provider
        for provider_name, provider_info in provider_mappings.items():
            section_name = provider_info.get('section')
            
            # Skip if section doesn't exist
            if not section_name or not config_parser.has_section(section_name):
                continue
            
            # Check if provider is configured
            is_configured = False
            
            if provider_info['type'] == 'commercial':
                # Check for API key
                api_key_field = provider_info.get('api_key_field')
                if api_key_field and config_parser.has_option(section_name, api_key_field):
                    api_key = config_parser.get(section_name, api_key_field, fallback='')
                    # Check if API key is valid (not empty and not placeholder)
                    if api_key and not api_key.startswith('<') and not api_key.endswith('>'):
                        is_configured = True
            else:
                # Check for endpoint URL for local providers
                endpoint_field = provider_info.get('endpoint_field')
                if endpoint_field and config_parser.has_option(section_name, endpoint_field):
                    endpoint_url = config_parser.get(section_name, endpoint_field, fallback='')
                    if endpoint_url and endpoint_url.strip() and not endpoint_url.startswith('<'):
                        is_configured = True
            
            # Always include the provider, but mark if it's configured
            # Get the models from config
            model_field = provider_info.get('model_field')
            models = []
            
            if model_field and config_parser.has_option(section_name, model_field):
                model_value = config_parser.get(section_name, model_field, fallback='')
                models = parse_model_string(model_value)
            
            # If no models found in config, use defaults based on provider
            if not models:
                # Define default models for each provider
                default_models = {
                    'openai': ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'],
                    'anthropic': ['claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', 'claude-3-opus-20240229'],
                    'cohere': ['command-r-plus', 'command-r', 'command'],
                    'deepseek': ['deepseek-chat', 'deepseek-coder'],
                    'google': ['gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash'],
                    'groq': ['llama-3.3-70b-versatile', 'llama-3.1-70b-versatile', 'mixtral-8x7b-32768'],
                    'mistral': ['mistral-large-latest', 'mistral-medium-latest', 'mistral-small-latest'],
                    'huggingface': ['meta-llama/Llama-2-70b-chat-hf', 'mistralai/Mixtral-8x7B-Instruct-v0.1'],
                    'openrouter': ['anthropic/claude-3.5-sonnet', 'openai/gpt-4o', 'meta-llama/llama-3.1-405b-instruct'],
                    # Local providers
                    'llama': ['local-model'],
                    'kobold': ['local-model'],
                    'ooba': ['local-model'],
                    'tabby': ['local-model'],
                    'vllm': ['local-model'],
                    'ollama': ['llama3.2', 'mistral', 'codellama'],
                    'aphrodite': ['local-model'],
                    'custom_openai_api': ['custom-model']
                }
                models = default_models.get(provider_name, ['default-model'])
            
            provider_data = {
                'name': provider_name,
                'display_name': provider_info['display_name'],
                'models': models,
                'type': provider_info['type'],
                'default_model': models[0] if models else None,
                'is_configured': is_configured  # Add configuration status
            }
            
            # Add endpoint for local providers
            if provider_info['type'] == 'local':
                endpoint_field = provider_info.get('endpoint_field')
                if endpoint_field and config_parser.has_option(section_name, endpoint_field):
                    provider_data['endpoint'] = config_parser.get(section_name, endpoint_field, fallback='')
            
            # Add other useful config fields
            temp_field = f'{provider_name}_temperature'
            if config_parser.has_option(section_name, temp_field):
                provider_data['default_temperature'] = float(config_parser.get(section_name, temp_field, fallback='0.7'))
            
            tokens_field = f'{provider_name}_max_tokens'
            if config_parser.has_option(section_name, tokens_field):
                provider_data['max_tokens'] = int(config_parser.get(section_name, tokens_field, fallback='4096'))
            
            streaming_field = f'{provider_name}_streaming'
            if config_parser.has_option(section_name, streaming_field):
                provider_data['supports_streaming'] = config_parser.get(section_name, streaming_field, fallback='False').lower() == 'true'
            
            providers.append(provider_data)
        
        # Get the default provider from config
        default_api = 'openai'
        if config_parser.has_section('API') and config_parser.has_option('API', 'default_api'):
            default_api = config_parser.get('API', 'default_api', fallback='openai')
        
        # Also check for additional models that might be listed elsewhere
        # For example, in the RAG or Embeddings sections
        if config_parser.has_section('Embeddings') and config_parser.has_option('Embeddings', 'contextual_llm_model'):
            contextual_model = config_parser.get('Embeddings', 'contextual_llm_model', fallback='')
            # Try to determine which provider this model belongs to
            if contextual_model and 'gpt' in contextual_model.lower():
                for p in providers:
                    if p['name'] == 'openai' and contextual_model not in p['models']:
                        p['models'].append(contextual_model)
        
        return {
            'providers': providers,
            'default_provider': default_api,
            'total_configured': len(providers)
        }
        
    except Exception as e:
        logger.error(f"Error getting configured providers: {e}", exc_info=True)
        return {
            'providers': [],
            'default_provider': 'openai',
            'total_configured': 0,
            'error': 'An internal error occurred getting available providers. Please check your config.txt file for errors and try again. If the problem persists, please contact support for assistance.'
        }


def get_all_available_models() -> List[str]:
    """
    Get a flat list of all available models across all configured providers.
    
    Returns:
        List of all available model names
    """
    result = get_configured_providers()
    models = []
    
    for provider in result.get('providers', []):
        for model in provider.get('models', []):
            # Add provider prefix to make models unique
            models.append(f"{provider['name']}/{model}")
    
    return models

#######################################################################################################################
#
# Endpoints:

@router.get("/llm/providers",
    summary="Get configured LLM providers",
    description="Returns a list of all configured LLM providers with their models from config",
    response_model=Dict[str, Any])
async def get_llm_providers():
    """
    Get all configured LLM providers and their models.
    
    Returns:
        Dictionary containing:
        - providers: List of provider configurations
        - default_provider: The default provider name
        - total_configured: Number of configured providers
    """
    try:
        result = get_configured_providers()
        
        if result['total_configured'] == 0:
            logger.warning("No LLM providers are configured")
            return {
                'providers': [],
                'default_provider': None,
                'total_configured': 0,
                'message': 'No LLM providers are configured. Please check your config.txt file.'
            }
        
        logger.info(f"Found {result['total_configured']} configured LLM providers")
        return result
        
    except Exception as e:
        logger.error(f"Error in get_llm_providers endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve LLM providers: {str(e)}"
        )

@router.get("/llm/providers/{provider_name}",
    summary="Get specific provider details",
    description="Returns details for a specific LLM provider",
    response_model=Dict[str, Any])
async def get_provider_details(provider_name: str):
    """
    Get details for a specific LLM provider.
    
    Args:
        provider_name: Name of the provider (e.g., 'openai', 'anthropic')
        
    Returns:
        Provider details including models and configuration
    """
    try:
        result = get_configured_providers()
        
        # Find the specific provider
        for provider in result['providers']:
            if provider['name'] == provider_name:
                logger.info(f"Retrieved details for provider: {provider_name}")
                return provider
        
        # Provider not found or not configured
        raise HTTPException(
            status_code=404,
            detail=f"Provider '{provider_name}' is not configured or not found"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting provider details for {provider_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve provider details: {str(e)}"
        )

@router.get("/llm/models",
    summary="Get all available models",
    description="Returns a flat list of all available models across all providers",
    response_model=List[str])
async def get_all_models():
    """
    Get all available models from all configured providers.
    
    Returns:
        List of model names with provider prefix
    """
    try:
        models = get_all_available_models()
        logger.info(f"Found {len(models)} total models across all providers")
        return models
    except Exception as e:
        logger.error(f"Error getting all models: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve models: {str(e)}"
        )

# End of llm_providers.py