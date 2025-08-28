"""
Tests for newly added LLM providers (Moonshot AI, Z.AI, HuggingFace API).
This version uses proper mocking to avoid actual API calls.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from pathlib import Path
import httpx
import sys
import os

# Add the project to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


class TestMoonshotProvider:
    """Tests for Moonshot AI provider."""
    
    @patch('tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls.requests')
    def test_moonshot_basic_chat(self, mock_requests):
        """Test basic chat functionality."""
        # Import here to ensure mock is in place
        from tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls import chat_with_moonshot
        
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "cmpl-test123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "moonshot-v1-8k",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello from Moonshot AI!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        mock_response.raise_for_status = Mock()
        
        # Mock the Session and its post method
        mock_session = Mock()
        mock_session.post.return_value = mock_response
        mock_session.mount = Mock()
        mock_requests.Session.return_value.__enter__.return_value = mock_session
        
        # Call the function
        result = chat_with_moonshot(
            input_data=[{"role": "user", "content": "Hello"}],
            api_key="test_key",
            model="moonshot-v1-8k"
        )
        
        # Assertions - function returns full response object
        assert result["choices"][0]["message"]["content"] == "Hello from Moonshot AI!"
        mock_session.post.assert_called_once()
        
        # Check the API was called with correct URL
        call_args = mock_session.post.call_args
        assert call_args[0][0] == "https://api.moonshot.cn/v1/chat/completions"
        
        # Check request payload - it uses json parameter
        payload = call_args[1]['json']
        assert payload['model'] == "moonshot-v1-8k"
        assert len(payload['messages']) == 1
        assert payload['messages'][0]['content'] == "Hello"
    
    @patch('tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls.requests')
    def test_moonshot_with_system_message(self, mock_requests):
        """Test chat with system message."""
        from tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls import chat_with_moonshot
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response with system"}}]
        }
        mock_response.raise_for_status = Mock()
        
        mock_session = Mock()
        mock_session.post.return_value = mock_response
        mock_session.mount = Mock()
        mock_requests.Session.return_value.__enter__.return_value = mock_session
        
        result = chat_with_moonshot(
            input_data=[{"role": "user", "content": "Hello"}],
            api_key="test_key",
            system_message="You are a helpful assistant."
        )
        
        assert result["choices"][0]["message"]["content"] == "Response with system"
        
        # Check that system message was added
        call_args = mock_session.post.call_args
        payload = call_args[1]['json']
        assert payload['messages'][0]['role'] == "system"
        assert payload['messages'][0]['content'] == "You are a helpful assistant."
        assert payload['messages'][1]['role'] == "user"
    
    @patch('tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls.requests')
    def test_moonshot_vision_model(self, mock_requests):
        """Test vision model with image content."""
        from tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls import chat_with_moonshot
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "I see an image"}}]
        }
        mock_response.raise_for_status = Mock()
        
        mock_session = Mock()
        mock_session.post.return_value = mock_response
        mock_session.mount = Mock()
        mock_requests.Session.return_value.__enter__.return_value = mock_session
        
        input_data = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
            ]
        }]
        
        result = chat_with_moonshot(
            input_data=input_data,
            api_key="test_key",
            model="moonshot-v1-8k-vision-preview"
        )
        
        assert result["choices"][0]["message"]["content"] == "I see an image"
        call_args = mock_session.post.call_args
        payload = call_args[1]['json']
        assert payload['model'] == "moonshot-v1-8k-vision-preview"
    
    @patch('tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls.requests')
    def test_moonshot_streaming(self, mock_requests):
        """Test streaming response."""
        from tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls import chat_with_moonshot
        
        # Mock SSE streaming response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/event-stream'}
        # iter_lines with decode_unicode=True returns strings
        mock_response.iter_lines = Mock(return_value=[
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" from"}}]}',
            'data: {"choices":[{"delta":{"content":" Moonshot!"}}]}',
            'data: [DONE]'
        ])
        mock_response.raise_for_status = Mock()
        
        mock_session = Mock()
        mock_session.post.return_value = mock_response
        mock_session.mount = Mock()
        mock_requests.Session.return_value.__enter__.return_value = mock_session
        
        chunks = []
        result_gen = chat_with_moonshot(
            input_data=[{"role": "user", "content": "Hello"}],
            api_key="test_key",
            streaming=True
        )
        
        for chunk in result_gen:
            chunks.append(chunk)
        
        assert len(chunks) == 5  # 3 content chunks + [DONE] + finally [DONE]
        assert "data: [DONE]" in chunks[-1] or "data: [DONE]" in chunks[-2]
        assert "Hello" in chunks[0]
    
    def test_moonshot_error_handling(self):
        """Test error handling."""
        import requests
        from tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls import chat_with_moonshot
        
        with patch('tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls.requests') as mock_requests:
            # Make sure the exceptions are accessible
            mock_requests.exceptions = requests.exceptions
            
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.text = "Unauthorized"
            # Use the actual HTTPError from requests
            http_error = requests.exceptions.HTTPError("401 Unauthorized")
            http_error.response = mock_response
            mock_response.raise_for_status.side_effect = http_error
            
            mock_session = Mock()
            mock_session.post.return_value = mock_response
            mock_session.mount = Mock()
            mock_requests.Session.return_value.__enter__.return_value = mock_session
            
            # The function raises HTTPError on 401
            with pytest.raises(requests.exceptions.HTTPError) as exc_info:
                chat_with_moonshot(
                    input_data=[{"role": "user", "content": "Hello"}],
                    api_key="invalid_key"
                )
            
            assert exc_info.value.response.status_code == 401


class TestZAIProvider:
    """Tests for Z.AI provider."""
    
    @patch('tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls.requests')
    def test_zai_basic_chat(self, mock_requests):
        """Test basic chat functionality."""
        from tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls import chat_with_zai
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chat-test123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "glm-4.5",
            "request_id": "req_test123",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello from Z.AI GLM!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        mock_response.raise_for_status = Mock()
        
        mock_session = Mock()
        mock_session.post.return_value = mock_response
        mock_session.mount = Mock()
        mock_requests.Session.return_value.__enter__.return_value = mock_session
        
        result = chat_with_zai(
            input_data=[{"role": "user", "content": "Hello"}],
            api_key="test_key",
            model="glm-4.5"
        )
        
        assert result["choices"][0]["message"]["content"] == "Hello from Z.AI GLM!"
        mock_session.post.assert_called_once()
        
        # Check request details
        call_args = mock_session.post.call_args
        assert call_args[0][0] == "https://api.z.ai/api/paas/v4/chat/completions"
        payload = call_args[1]['json']
        assert payload['model'] == "glm-4.5"
    
    @patch('tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls.requests')
    def test_zai_with_request_id(self, mock_requests):
        """Test chat with request_id."""
        from tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls import chat_with_zai
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}}]
        }
        mock_response.raise_for_status = Mock()
        
        mock_session = Mock()
        mock_session.post.return_value = mock_response
        mock_session.mount = Mock()
        mock_requests.Session.return_value.__enter__.return_value = mock_session
        
        result = chat_with_zai(
            input_data=[{"role": "user", "content": "Hello"}],
            api_key="test_key",
            request_id="custom_req_123"
        )
        
        assert result["choices"][0]["message"]["content"] == "Response"
        
        call_args = mock_session.post.call_args
        payload = call_args[1]['json']
        assert payload.get('request_id') == "custom_req_123"
    
    @patch('tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls.requests')
    def test_zai_model_variants(self, mock_requests):
        """Test different model variants."""
        from tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls import chat_with_zai
        
        models = ["glm-4.5", "glm-4.5-air", "glm-4.5-flash", "glm-4-32b-0414-128k"]
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}}]
        }
        mock_response.raise_for_status = Mock()
        
        mock_session = Mock()
        mock_session.post.return_value = mock_response
        mock_session.mount = Mock()
        mock_requests.Session.return_value.__enter__.return_value = mock_session
        
        for model in models:
            result = chat_with_zai(
                input_data=[{"role": "user", "content": "Test"}],
                api_key="test_key",
                model=model
            )
            
            assert result["choices"][0]["message"]["content"] == "Response"
            
            call_args = mock_session.post.call_args
            payload = call_args[1]['json']
            assert payload['model'] == model
    
    @patch('tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls.requests')
    def test_zai_streaming(self, mock_requests):
        """Test streaming response."""
        from tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls import chat_with_zai
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/event-stream'}
        # iter_lines with decode_unicode=True returns strings
        mock_response.iter_lines = Mock(return_value=[
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" GLM"}}]}',
            'data: [DONE]'
        ])
        mock_response.raise_for_status = Mock()
        
        mock_session = Mock()
        mock_session.post.return_value = mock_response
        mock_session.mount = Mock()
        mock_requests.Session.return_value.__enter__.return_value = mock_session
        
        chunks = []
        result_gen = chat_with_zai(
            input_data=[{"role": "user", "content": "Hello"}],
            api_key="test_key",
            streaming=True
        )
        
        for chunk in result_gen:
            chunks.append(chunk)
        
        assert len(chunks) == 4  # 2 content chunks + [DONE] + finally [DONE]
        assert "data: [DONE]" in chunks[-1] or "data: [DONE]" in chunks[-2]


class TestHuggingFaceAPI:
    """Tests for HuggingFace API client."""
    
    @pytest.fixture
    def api_client(self):
        """Create HuggingFace API client."""
        from tldw_Server_API.app.core.LLM_Calls.huggingface_api import HuggingFaceAPI
        return HuggingFaceAPI(token="test_token")
    
    @pytest.mark.asyncio
    async def test_search_models(self, api_client):
        """Test model search functionality."""
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = [
                {
                    "modelId": "TheBloke/Llama-2-7B-GGUF",
                    "author": "TheBloke",
                    "downloads": 100000,
                    "likes": 500,
                    "tags": ["gguf", "llama"]
                },
                {
                    "modelId": "TheBloke/Mistral-7B-GGUF",
                    "author": "TheBloke",
                    "downloads": 50000,
                    "likes": 300,
                    "tags": ["gguf", "mistral"]
                }
            ]
            mock_response.raise_for_status = Mock()
            
            # Make get return an awaitable
            async def async_get(*args, **kwargs):
                return mock_response
            mock_get.side_effect = async_get
            
            results = await api_client.search_models(
                query="llama",
                filter_tags=["gguf"],
                limit=10
            )
            
            assert len(results) == 2
            assert results[0]["modelId"] == "TheBloke/Llama-2-7B-GGUF"
    
    @pytest.mark.asyncio
    async def test_get_model_info(self, api_client):
        """Test getting model information."""
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "modelId": "TheBloke/Llama-2-7B-GGUF",
                "author": "TheBloke",
                "downloads": 100000,
                "description": "Llama 2 7B model in GGUF format"
            }
            mock_response.raise_for_status = Mock()
            
            async def async_get(*args, **kwargs):
                return mock_response
            mock_get.side_effect = async_get
        
            info = await api_client.get_model_info("TheBloke/Llama-2-7B-GGUF")
            
            assert info["modelId"] == "TheBloke/Llama-2-7B-GGUF"
            assert info["downloads"] == 100000
    
    @pytest.mark.asyncio
    async def test_list_model_files(self, api_client):
        """Test listing GGUF files in a repository."""
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = [
                {"path": "llama-2-7b.Q4_K_M.gguf", "size": 3825000000},
                {"path": "llama-2-7b.Q5_K_S.gguf", "size": 4650000000},
                {"path": "README.md", "size": 5000}
            ]
            mock_response.raise_for_status = Mock()
            
            async def async_get(*args, **kwargs):
                return mock_response
            mock_get.side_effect = async_get
        
            files = await api_client.list_model_files("TheBloke/Llama-2-7B-GGUF")
            
            # Should only return GGUF files
            assert len(files) == 2
            assert all(f["path"].endswith(".gguf") for f in files)
    
    @pytest.mark.asyncio
    async def test_download_file(self, api_client, tmp_path):
        """Test file download functionality."""
        test_content = b"This is a test GGUF file content"
        
        with patch('httpx.AsyncClient.head') as mock_head:
            with patch('httpx.AsyncClient.stream') as mock_stream:
                # Mock HEAD request for file size
                mock_head_response = AsyncMock()
                mock_head_response.headers = {"content-length": str(len(test_content))}
                mock_head.return_value = mock_head_response
                
                # Create async context manager mock
                mock_stream_context = AsyncMock()
                mock_stream_context.raise_for_status = Mock()
                
                # Create async iterator for content
                async def async_content(chunk_size=None):
                    yield test_content
                
                mock_stream_context.aiter_bytes = async_content
                
                # Setup context manager
                mock_stream.return_value.__aenter__.return_value = mock_stream_context
                mock_stream.return_value.__aexit__.return_value = None
                
                destination = tmp_path / "test_model.gguf"
                success = await api_client.download_file(
                    repo_id="TheBloke/Test-GGUF",
                    filename="test.gguf",
                    destination=destination
                )
                
                # For the test, just check that it returns True
                assert success is True
    
    @pytest.mark.asyncio
    async def test_error_handling(self, api_client):
        """Test error handling in API calls."""
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_get.side_effect = httpx.HTTPError("Connection error")
            
            results = await api_client.search_models(query="test")
            assert results == []  # Should return empty list on error
            
            info = await api_client.get_model_info("test/model")
            assert info is None  # Should return None on error
    
    @pytest.mark.asyncio
    async def test_find_best_gguf_model(self):
        """Test finding best GGUF model utility."""
        from tldw_Server_API.app.core.LLM_Calls.huggingface_api import find_best_gguf_model, HuggingFaceAPI
        
        mock_models = [
            {
                "modelId": "TheBloke/Llama-2-7B-GGUF",
                "downloads": 100000
            }
        ]
        
        with patch.object(HuggingFaceAPI, 'search_gguf_models') as mock_search:
            async def return_models(*args, **kwargs):
                return mock_models
            mock_search.side_effect = return_models
            
            best_model = await find_best_gguf_model(
                model_name="llama-2",
                max_size_gb=10.0,
                preferred_quant="Q4_K_M"
            )
            
            assert best_model["modelId"] == "TheBloke/Llama-2-7B-GGUF"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])