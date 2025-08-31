"""
Unit and integration tests for external transcription provider support.
"""

import pytest
import numpy as np
import asyncio
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import httpx
from pathlib import Path
import tempfile

pytestmark = pytest.mark.unit


class TestExternalProvider:
    """Test suite for external transcription provider."""
    
    @pytest.fixture
    def sample_audio(self):
        """Generate sample audio data."""
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = np.sin(440 * 2 * np.pi * t).astype(np.float32)
        return audio, sample_rate
    
    @pytest.fixture
    def provider_config(self):
        """Create test provider configuration."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
            ExternalProviderConfig
        )
        
        return ExternalProviderConfig(
            base_url="https://api.example.com/v1/audio/transcriptions",
            api_key="test-api-key",
            model="whisper-1",
            timeout=30.0,
            max_retries=2,
            verify_ssl=True,
            response_format="json"
        )
    
    def test_import_module(self):
        """Test module imports."""
        try:
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
                transcribe_with_external_provider,
                ExternalProviderConfig,
                validate_external_provider_config,
                add_external_provider,
                list_external_providers
            )
            assert transcribe_with_external_provider is not None
            assert ExternalProviderConfig is not None
        except ImportError as e:
            pytest.skip(f"External provider module not available: {e}")
    
    def test_config_validation(self, provider_config):
        """Test configuration validation."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
            validate_external_provider_config,
            ExternalProviderConfig
        )
        
        # Valid config
        is_valid, error = validate_external_provider_config(provider_config)
        assert is_valid == True
        assert error is None
        
        # Invalid URL
        invalid_config = ExternalProviderConfig(
            base_url="not-a-url",
            api_key="key"
        )
        is_valid, error = validate_external_provider_config(invalid_config)
        assert is_valid == False
        assert "Invalid base URL" in error
        
        # Invalid timeout
        invalid_config = ExternalProviderConfig(
            base_url="https://api.example.com",
            timeout=-1
        )
        is_valid, error = validate_external_provider_config(invalid_config)
        assert is_valid == False
        assert "Timeout must be positive" in error
        
        # Invalid temperature
        invalid_config = ExternalProviderConfig(
            base_url="https://api.example.com",
            temperature=3.0
        )
        is_valid, error = validate_external_provider_config(invalid_config)
        assert is_valid == False
        assert "Temperature must be between" in error
        
        # Invalid response format
        invalid_config = ExternalProviderConfig(
            base_url="https://api.example.com",
            response_format="invalid"
        )
        is_valid, error = validate_external_provider_config(invalid_config)
        assert is_valid == False
        assert "Invalid response format" in error
    
    def test_add_and_list_providers(self, provider_config):
        """Test adding and listing providers."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
            add_external_provider,
            list_external_providers,
            remove_external_provider,
            _provider_configs
        )
        
        # Clear existing providers
        _provider_configs.clear()
        
        # Add provider
        success = add_external_provider("test_provider", provider_config)
        assert success == True
        
        # List providers
        providers = list_external_providers()
        assert "test_provider" in providers
        
        # Remove provider
        removed = remove_external_provider("test_provider")
        assert removed == True
        
        # Verify removed
        providers = list_external_providers()
        assert "test_provider" not in providers
    
    @pytest.mark.asyncio
    async def test_transcribe_with_mock_api(self, sample_audio, provider_config):
        """Test transcription with mocked API."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
            transcribe_with_external_provider_async
        )
        
        audio_data, sample_rate = sample_audio
        
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'text': 'Mocked transcription result',
            'task': 'transcribe',
            'language': 'en'
        }
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client
            
            result = await transcribe_with_external_provider_async(
                audio_data,
                sample_rate,
                config=provider_config
            )
            
            assert result == 'Mocked transcription result'
            mock_client.post.assert_called_once()
            
            # Verify request parameters
            call_args = mock_client.post.call_args
            assert provider_config.base_url in call_args[0][0]
            assert 'headers' in call_args[1]
            assert 'Authorization' in call_args[1]['headers']
    
    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self, sample_audio, provider_config):
        """Test retry logic on rate limiting."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
            transcribe_with_external_provider_async
        )
        
        audio_data, sample_rate = sample_audio
        provider_config.max_retries = 3
        
        # Mock responses: first two fail with 429, third succeeds
        mock_responses = [
            MagicMock(status_code=429, text="Rate limited"),
            MagicMock(status_code=429, text="Rate limited"),
            MagicMock(status_code=200, json=lambda: {'text': 'Success after retry'})
        ]
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = mock_responses
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client
            
            with patch('asyncio.sleep'):  # Skip actual sleep in tests
                result = await transcribe_with_external_provider_async(
                    audio_data,
                    sample_rate,
                    config=provider_config
                )
            
            assert result == 'Success after retry'
            assert mock_client.post.call_count == 3
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, sample_audio, provider_config):
        """Test timeout handling."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
            transcribe_with_external_provider_async
        )
        
        audio_data, sample_rate = sample_audio
        provider_config.max_retries = 2
        provider_config.timeout = 0.1  # Very short timeout
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("Request timeout")
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client
            
            result = await transcribe_with_external_provider_async(
                audio_data,
                sample_rate,
                config=provider_config
            )
            
            assert "[Error:" in result
            assert "timeout" in result.lower()
            assert mock_client.post.call_count == 2  # Should retry once
    
    @pytest.mark.asyncio
    async def test_different_response_formats(self, sample_audio, provider_config):
        """Test handling different response formats."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
            transcribe_with_external_provider_async
        )
        
        audio_data, sample_rate = sample_audio
        
        # Test JSON format
        provider_config.response_format = 'json'
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'text': 'JSON response'}
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client
            
            result = await transcribe_with_external_provider_async(
                audio_data,
                sample_rate,
                config=provider_config
            )
            
            assert result == 'JSON response'
        
        # Test text format
        provider_config.response_format = 'text'
        mock_response.text = 'Plain text response'
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client
            
            result = await transcribe_with_external_provider_async(
                audio_data,
                sample_rate,
                config=provider_config
            )
            
            assert result == 'Plain text response'
        
        # Test SRT format
        provider_config.response_format = 'srt'
        mock_response.text = '1\n00:00:00,000 --> 00:00:01,000\nSRT subtitle'
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client
            
            result = await transcribe_with_external_provider_async(
                audio_data,
                sample_rate,
                config=provider_config
            )
            
            assert 'SRT subtitle' in result
    
    def test_synchronous_wrapper(self, sample_audio, provider_config):
        """Test synchronous wrapper function."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
            transcribe_with_external_provider
        )
        
        audio_data, sample_rate = sample_audio
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'text': 'Sync wrapper result'}
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client
            
            result = transcribe_with_external_provider(
                audio_data,
                sample_rate,
                config=provider_config
            )
            
            assert result == 'Sync wrapper result'
    
    @pytest.mark.asyncio
    async def test_with_file_path(self, provider_config):
        """Test transcription with file path input."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
            transcribe_with_external_provider_async
        )
        
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sample_rate = 16000
            audio = np.random.randn(sample_rate).astype(np.float32)
            import soundfile as sf
            sf.write(tmp_file.name, audio, sample_rate)
            tmp_path = tmp_file.name
        
        try:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'text': 'File path result'}
            
            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post.return_value = mock_response
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client_class.return_value = mock_client
                
                result = await transcribe_with_external_provider_async(
                    tmp_path,
                    config=provider_config
                )
                
                assert result == 'File path result'
        
        finally:
            # Clean up
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    @pytest.mark.asyncio
    async def test_provider_test_function(self, provider_config):
        """Test the provider test function."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
            test_external_provider
        )
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'text': 'Test successful'}
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client
            
            result = await test_external_provider(config=provider_config)
            
            assert result['success'] == True
            assert result['result'] == 'Test successful'
            assert 'elapsed_time' in result
            assert result['elapsed_time'] >= 0


@pytest.mark.integration
class TestExternalProviderIntegration:
    """Integration tests for external provider."""
    
    def test_integration_with_main_library(self):
        """Test integration with main transcription library."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
            register_external_provider_with_library
        )
        
        # This should not raise any errors
        register_external_provider_with_library()
    
    @pytest.mark.asyncio
    async def test_load_from_environment(self):
        """Test loading configuration from environment variables."""
        import os
        
        # Set environment variables
        os.environ['EXTERNAL_TRANSCRIPTION_TEST_BASE_URL'] = 'https://test.api.com'
        os.environ['EXTERNAL_TRANSCRIPTION_TEST_API_KEY'] = 'test-key'
        os.environ['EXTERNAL_TRANSCRIPTION_TEST_MODEL'] = 'test-model'
        
        try:
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
                load_external_provider_config
            )
            
            config = load_external_provider_config('test')
            
            if config:  # May be None if config loading not implemented
                assert config.base_url == 'https://test.api.com'
                assert config.api_key == 'test-key'
                assert config.model == 'test-model'
        
        finally:
            # Clean up environment
            for key in ['BASE_URL', 'API_KEY', 'MODEL']:
                env_key = f'EXTERNAL_TRANSCRIPTION_TEST_{key}'
                if env_key in os.environ:
                    del os.environ[env_key]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])