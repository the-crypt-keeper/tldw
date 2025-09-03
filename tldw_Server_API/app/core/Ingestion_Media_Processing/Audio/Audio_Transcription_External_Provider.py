# Audio_Transcription_External_Provider.py
#########################################
# External Transcription Provider Support
# Allows forwarding transcription requests to external OpenAI-compatible Audio APIs
#
####################
# Function List
#
# 1. transcribe_with_external_provider() - Main transcription function
# 2. validate_external_provider_config() - Validate provider configuration
# 3. format_external_request() - Format request for external API
# 4. parse_external_response() - Parse response from external API
# 5. ExternalProviderConfig - Configuration dataclass
#
####################

import os
import logging
import tempfile
from typing import Optional, Dict, Any, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import soundfile as sf
import httpx
import asyncio
import json
import base64
from urllib.parse import urlparse, urljoin

logger = logging.getLogger(__name__)


@dataclass
class ExternalProviderConfig:
    """Configuration for external transcription provider."""
    base_url: str
    api_key: Optional[str] = None
    model: str = "whisper-1"
    timeout: float = 300.0  # 5 minutes default
    max_retries: int = 3
    verify_ssl: bool = True
    custom_headers: Optional[Dict[str, str]] = None
    response_format: str = "json"
    temperature: float = 0.0
    language: Optional[str] = None
    prompt: Optional[str] = None


# Global cache for provider configurations
_provider_configs: Dict[str, ExternalProviderConfig] = {}


def load_external_provider_config(provider_name: str = "default") -> Optional[ExternalProviderConfig]:
    """
    Load external provider configuration from config file or environment.
    
    Args:
        provider_name: Name of the provider configuration to load
    
    Returns:
        ExternalProviderConfig or None if not configured
    """
    if provider_name in _provider_configs:
        return _provider_configs[provider_name]
    
    # Try to load from environment variables
    env_prefix = f"EXTERNAL_TRANSCRIPTION_{provider_name.upper()}_"
    
    # Initialize external_config
    external_config = {}
    
    base_url = os.getenv(f"{env_prefix}BASE_URL")
    if not base_url:
        # Try loading from config file
        try:
            from tldw_Server_API.app.core.Config_Management.config_utils import load_comprehensive_config
            config_data = load_comprehensive_config()
            
            external_config = config_data.get('STT-Settings', {}).get('external_providers', {}).get(provider_name, {})
            if external_config:
                base_url = external_config.get('base_url')
        except Exception as e:
            logger.debug(f"Could not load config for external provider {provider_name}: {e}")
            return None
    
    if not base_url:
        return None
    
    config = ExternalProviderConfig(
        base_url=base_url,
        api_key=os.getenv(f"{env_prefix}API_KEY") or external_config.get('api_key'),
        model=os.getenv(f"{env_prefix}MODEL") or external_config.get('model', 'whisper-1'),
        timeout=float(os.getenv(f"{env_prefix}TIMEOUT", "300")) or external_config.get('timeout', 300),
        max_retries=int(os.getenv(f"{env_prefix}MAX_RETRIES", "3")) or external_config.get('max_retries', 3),
        verify_ssl=os.getenv(f"{env_prefix}VERIFY_SSL", "true").lower() == "true",
        response_format=os.getenv(f"{env_prefix}RESPONSE_FORMAT") or external_config.get('response_format', 'json'),
        temperature=float(os.getenv(f"{env_prefix}TEMPERATURE", "0")) or external_config.get('temperature', 0),
        language=os.getenv(f"{env_prefix}LANGUAGE") or external_config.get('language'),
        prompt=os.getenv(f"{env_prefix}PROMPT") or external_config.get('prompt')
    )
    
    # Cache the configuration
    _provider_configs[provider_name] = config
    
    return config


def validate_external_provider_config(config: ExternalProviderConfig) -> Tuple[bool, Optional[str]]:
    """
    Validate external provider configuration.
    
    Args:
        config: Provider configuration to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Validate base URL
    try:
        parsed = urlparse(config.base_url)
        if not parsed.scheme or not parsed.netloc:
            return False, "Invalid base URL format"
    except Exception as e:
        return False, f"Invalid base URL: {e}"
    
    # Validate timeout
    if config.timeout <= 0:
        return False, "Timeout must be positive"
    
    # Validate temperature
    if not 0 <= config.temperature <= 2:
        return False, "Temperature must be between 0 and 2"
    
    # Validate response format
    valid_formats = ['json', 'text', 'srt', 'verbose_json', 'vtt']
    if config.response_format not in valid_formats:
        return False, f"Invalid response format. Must be one of: {valid_formats}"
    
    return True, None


async def transcribe_with_external_provider_async(
    audio_data: Union[np.ndarray, str, Path],
    sample_rate: int = 16000,
    provider_name: str = "default",
    config: Optional[ExternalProviderConfig] = None,
    **kwargs
) -> str:
    """
    Asynchronously transcribe audio using an external OpenAI-compatible API.
    
    Args:
        audio_data: Audio data as numpy array or path to audio file
        sample_rate: Sample rate of the audio
        provider_name: Name of the provider configuration to use
        config: Optional ExternalProviderConfig to use instead of loading from config
        **kwargs: Additional parameters to pass to the API
    
    Returns:
        Transcribed text or error message
    """
    # Load configuration
    if config is None:
        config = load_external_provider_config(provider_name)
        if config is None:
            return f"[Error: External provider '{provider_name}' not configured]"
    
    # Validate configuration
    is_valid, error_msg = validate_external_provider_config(config)
    if not is_valid:
        return f"[Error: Invalid configuration - {error_msg}]"
    
    # Prepare audio file
    audio_file_path = None
    temp_file = None
    
    try:
        if isinstance(audio_data, (str, Path)):
            audio_file_path = str(audio_data)
        else:
            # Save audio to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(temp_file.name, audio_data, sample_rate)
            audio_file_path = temp_file.name
        
        # Prepare the request
        endpoint = urljoin(config.base_url, "/v1/audio/transcriptions")
        if not endpoint.startswith(config.base_url):
            # Handle cases where base_url already includes the full path
            endpoint = config.base_url
        
        # Prepare headers
        headers = {}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        if config.custom_headers:
            headers.update(config.custom_headers)
        
        # Prepare form data
        with open(audio_file_path, 'rb') as audio_file:
            files = {
                'file': ('audio.wav', audio_file, 'audio/wav')
            }
            
            data = {
                'model': config.model,
                'response_format': config.response_format,
                'temperature': str(config.temperature)
            }
            
            # Add optional parameters
            if config.language:
                data['language'] = config.language
            if config.prompt:
                data['prompt'] = config.prompt
            
            # Add any additional kwargs
            for key, value in kwargs.items():
                if key not in data:
                    data[key] = str(value)
            
            # Make the request with retries
            async with httpx.AsyncClient(verify=config.verify_ssl, timeout=config.timeout) as client:
                for attempt in range(config.max_retries):
                    try:
                        response = await client.post(
                            endpoint,
                            headers=headers,
                            files=files,
                            data=data
                        )
                        
                        if response.status_code == 200:
                            # Parse response based on format
                            if config.response_format == 'text':
                                return response.text
                            elif config.response_format in ['json', 'verbose_json']:
                                result = response.json()
                                return result.get('text', '')
                            elif config.response_format in ['srt', 'vtt']:
                                return response.text
                            else:
                                return response.text
                        
                        elif response.status_code == 429:  # Rate limit
                            if attempt < config.max_retries - 1:
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                continue
                            else:
                                return f"[Error: Rate limit exceeded after {config.max_retries} attempts]"
                        
                        else:
                            error_detail = response.text
                            try:
                                error_json = response.json()
                                error_detail = error_json.get('error', {}).get('message', error_detail)
                            except:
                                pass
                            return f"[Error: API returned {response.status_code} - {error_detail}]"
                    
                    except httpx.TimeoutException:
                        if attempt < config.max_retries - 1:
                            logger.warning(f"Timeout on attempt {attempt + 1}, retrying...")
                            continue
                        else:
                            return f"[Error: Request timeout after {config.max_retries} attempts]"
                    
                    except Exception as e:
                        if attempt < config.max_retries - 1:
                            logger.warning(f"Error on attempt {attempt + 1}: {e}, retrying...")
                            await asyncio.sleep(1)
                            continue
                        else:
                            return f"[Error: {str(e)}]"
        
        return "[Error: Failed to transcribe after all retries]"
    
    except Exception as e:
        logger.error(f"Error in external provider transcription: {e}")
        return f"[Error: {str(e)}]"
    
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.remove(temp_file.name)
            except:
                pass


def transcribe_with_external_provider(
    audio_data: Union[np.ndarray, str, Path],
    sample_rate: int = 16000,
    provider_name: str = "default",
    config: Optional[ExternalProviderConfig] = None,
    **kwargs
) -> str:
    """
    Synchronous wrapper for external provider transcription.
    
    Args:
        audio_data: Audio data as numpy array or path to audio file
        sample_rate: Sample rate of the audio
        provider_name: Name of the provider configuration to use
        config: Optional ExternalProviderConfig to use instead of loading from config
        **kwargs: Additional parameters to pass to the API
    
    Returns:
        Transcribed text or error message
    """
    try:
        # Run async function in sync context
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running (e.g., in Jupyter), create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    transcribe_with_external_provider_async(
                        audio_data, sample_rate, provider_name, config, **kwargs
                    )
                )
                return future.result()
        else:
            # Normal case - run the async function
            return asyncio.run(
                transcribe_with_external_provider_async(
                    audio_data, sample_rate, provider_name, config, **kwargs
                )
            )
    except Exception as e:
        logger.error(f"Error in external provider transcription: {e}")
        return f"[Error: {str(e)}]"


def add_external_provider(name: str, config: ExternalProviderConfig) -> bool:
    """
    Add or update an external provider configuration.
    
    Args:
        name: Name of the provider
        config: Provider configuration
    
    Returns:
        True if successful
    """
    is_valid, error_msg = validate_external_provider_config(config)
    if not is_valid:
        logger.error(f"Invalid configuration for provider {name}: {error_msg}")
        return False
    
    _provider_configs[name] = config
    logger.info(f"Added external provider: {name}")
    return True


def list_external_providers() -> list[str]:
    """
    List configured external providers.
    
    Returns:
        List of provider names
    """
    return list(_provider_configs.keys())


def remove_external_provider(name: str) -> bool:
    """
    Remove an external provider configuration.
    
    Args:
        name: Name of the provider to remove
    
    Returns:
        True if removed, False if not found
    """
    if name in _provider_configs:
        del _provider_configs[name]
        logger.info(f"Removed external provider: {name}")
        return True
    return False


async def test_external_provider(
    provider_name: str = "default",
    config: Optional[ExternalProviderConfig] = None
) -> Dict[str, Any]:
    """
    Test an external provider with a simple audio sample.
    
    Args:
        provider_name: Name of the provider to test
        config: Optional configuration to test
    
    Returns:
        Dictionary with test results
    """
    import time
    
    # Generate test audio (1 second of sine wave)
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(440 * 2 * np.pi * t).astype(np.float32) * 0.5
    
    start_time = time.time()
    
    try:
        result = await transcribe_with_external_provider_async(
            audio,
            sample_rate,
            provider_name,
            config
        )
        
        elapsed = time.time() - start_time
        
        return {
            'success': not result.startswith('[Error:'),
            'result': result,
            'elapsed_time': elapsed,
            'provider': provider_name
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'elapsed_time': time.time() - start_time,
            'provider': provider_name
        }


# Integration with main transcription library
def register_external_provider_with_library():
    """
    Register external provider support with the main transcription library.
    
    This should be called during application initialization.
    """
    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (
            register_transcription_provider
        )
        
        def external_provider_wrapper(audio_data, sample_rate=16000, **kwargs):
            provider_name = kwargs.get('provider_name', 'default')
            return transcribe_with_external_provider(
                audio_data,
                sample_rate,
                provider_name=provider_name,
                **kwargs
            )
        
        register_transcription_provider('external', external_provider_wrapper)
        logger.info("External provider support registered with transcription library")
        
    except ImportError:
        logger.warning("Could not register external provider with transcription library")


#######################################################################################################################
# End of Audio_Transcription_External_Provider.py
#######################################################################################################################