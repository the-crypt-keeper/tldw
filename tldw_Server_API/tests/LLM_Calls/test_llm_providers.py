"""
Tests for newly added LLM providers (Moonshot AI, Z.AI, HuggingFace API).
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
import httpx

# Import the modules to test
from tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls import (
    chat_with_moonshot,
    chat_with_zai
)
from tldw_Server_API.app.core.LLM_Calls.huggingface_api import (
    HuggingFaceAPI,
    find_best_gguf_model,
    download_gguf_model
)


class TestMoonshotProvider:
    """Tests for Moonshot AI provider."""
    
    @pytest.fixture
    def mock_response(self):
        """Mock response for Moonshot API."""
        return {
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
    
    def test_moonshot_basic_chat(self, mock_response):
        """Test basic chat functionality."""
        with patch('requests.post') as mock_post:
            mock_response_obj = Mock()
            mock_response_obj.status_code = 200
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = Mock()
            mock_post.return_value = mock_response_obj
            
            result = chat_with_moonshot(
                input_data=[{"role": "user", "content": "Hello"}],
                api_key="test_key",
                model="moonshot-v1-8k"
            )
            
            assert result == "Hello from Moonshot AI!"
            mock_post.assert_called_once()
            
            # Check request payload
            call_args = mock_post.call_args
            payload = json.loads(call_args[1]['data'])
            assert payload['model'] == "moonshot-v1-8k"
            assert len(payload['messages']) == 1
    
    @patch('requests.post')
    def test_moonshot_with_system_message(self, mock_post, mock_response):
        """Test chat with system message."""
        mock_response_obj = Mock()
        mock_response_obj.status_code = 200
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.raise_for_status = Mock()
        mock_post.return_value = mock_response_obj
        
        result = chat_with_moonshot(
            input_data=[{"role": "user", "content": "Hello"}],
            api_key="test_key",
            system_message="You are a helpful assistant."
        )
        
        call_args = mock_post.call_args
        payload = json.loads(call_args[1]['data'])
        assert payload['messages'][0]['role'] == "system"
        assert payload['messages'][0]['content'] == "You are a helpful assistant."
    
    @patch('requests.post')
    def test_moonshot_vision_model(self, mock_post, mock_response):
        """Test vision model with image content."""
        mock_response['model'] = "moonshot-v1-8k-vision-preview"
        mock_response_obj = Mock()
        mock_response_obj.status_code = 200
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.raise_for_status = Mock()
        mock_post.return_value = mock_response_obj
        
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
        
        assert result == "Hello from Moonshot AI!"
        call_args = mock_post.call_args
        payload = json.loads(call_args[1]['data'])
        assert payload['model'] == "moonshot-v1-8k-vision-preview"
    
    @patch('requests.post')
    def test_moonshot_streaming(self, mock_post):
        """Test streaming response."""
        # Mock SSE streaming response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/event-stream'}
        mock_response.iter_lines = Mock(return_value=[
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            b'data: {"choices":[{"delta":{"content":" from"}}]}',
            b'data: {"choices":[{"delta":{"content":" Moonshot!"}}]}',
            b'data: [DONE]'
        ])
        mock_post.return_value = mock_response
        
        chunks = []
        result_gen = chat_with_moonshot(
            input_data=[{"role": "user", "content": "Hello"}],
            api_key="test_key",
            streaming=True
        )
        
        for chunk in result_gen:
            chunks.append(chunk)
        
        assert len(chunks) == 4  # 3 content chunks + [DONE]
        assert chunks[-1] == "[DONE]"
    
    @patch('requests.post')
    def test_moonshot_error_handling(self, mock_post):
        """Test error handling."""
        mock_response_obj = Mock()
        mock_response_obj.status_code = 401
        mock_response_obj.text = "Unauthorized"
        mock_response_obj.raise_for_status.side_effect = Exception("401 Unauthorized")
        mock_post.return_value = mock_response_obj
        
        result = chat_with_moonshot(
            input_data=[{"role": "user", "content": "Hello"}],
            api_key="invalid_key"
        )
        
        assert "Moonshot API error" in result


class TestZAIProvider:
    """Tests for Z.AI provider."""
    
    @pytest.fixture
    def mock_response(self):
        """Mock response for Z.AI API."""
        return {
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
    
    @patch('requests.post')
    def test_zai_basic_chat(self, mock_post, mock_response):
        """Test basic chat functionality."""
        mock_response_obj = Mock()
        mock_response_obj.status_code = 200
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.raise_for_status = Mock()
        mock_post.return_value = mock_response_obj
        
        result = chat_with_zai(
            input_data=[{"role": "user", "content": "Hello"}],
            api_key="test_key",
            model="glm-4.5"
        )
        
        assert result == "Hello from Z.AI GLM!"
        mock_post.assert_called_once()
        
        # Check request payload
        call_args = mock_post.call_args
        payload = json.loads(call_args[1]['data'])
        assert payload['model'] == "glm-4.5"
    
    @patch('requests.post')
    def test_zai_with_request_id(self, mock_post, mock_response):
        """Test chat with request_id."""
        mock_response_obj = Mock()
        mock_response_obj.status_code = 200
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.raise_for_status = Mock()
        mock_post.return_value = mock_response_obj
        
        result = chat_with_zai(
            input_data=[{"role": "user", "content": "Hello"}],
            api_key="test_key",
            request_id="custom_req_123"
        )
        
        call_args = mock_post.call_args
        payload = json.loads(call_args[1]['data'])
        assert payload.get('request_id') == "custom_req_123"
    
    @patch('requests.post')
    def test_zai_model_variants(self, mock_post, mock_response):
        """Test different model variants."""
        models = ["glm-4.5", "glm-4.5-air", "glm-4.5-flash", "glm-4-32b-0414-128k"]
        
        for model in models:
            mock_response['model'] = model
            mock_response_obj = Mock()
            mock_response_obj.status_code = 200
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = Mock()
            mock_post.return_value = mock_response_obj
            
            result = chat_with_zai(
                input_data=[{"role": "user", "content": "Test"}],
                api_key="test_key",
                model=model
            )
            
            call_args = mock_post.call_args
            payload = json.loads(call_args[1]['data'])
            assert payload['model'] == model
    
    @patch('requests.post')
    def test_zai_streaming(self, mock_post):
        """Test streaming response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/event-stream'}
        mock_response.iter_lines = Mock(return_value=[
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            b'data: {"choices":[{"delta":{"content":" GLM"}}]}',
            b'data: [DONE]'
        ])
        mock_post.return_value = mock_response
        
        chunks = []
        result_gen = chat_with_zai(
            input_data=[{"role": "user", "content": "Hello"}],
            api_key="test_key",
            streaming=True
        )
        
        for chunk in result_gen:
            chunks.append(chunk)
        
        assert len(chunks) == 3
        assert chunks[-1] == "[DONE]"


class TestHuggingFaceAPI:
    """Tests for HuggingFace API client."""
    
    @pytest.fixture
    def api_client(self):
        """Create HuggingFace API client."""
        return HuggingFaceAPI(token="test_token")
    
    @pytest.fixture
    def mock_model_response(self):
        """Mock model search response."""
        return [
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
    
    @pytest.mark.asyncio
    async def test_search_models(self, api_client, mock_model_response):
        """Test model search functionality."""
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_model_response
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
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
        mock_info = {
            "modelId": "TheBloke/Llama-2-7B-GGUF",
            "author": "TheBloke",
            "downloads": 100000,
            "description": "Llama 2 7B model in GGUF format"
        }
        
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_info
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            info = await api_client.get_model_info("TheBloke/Llama-2-7B-GGUF")
            
            assert info["modelId"] == "TheBloke/Llama-2-7B-GGUF"
            assert info["downloads"] == 100000
    
    @pytest.mark.asyncio
    async def test_list_model_files(self, api_client):
        """Test listing GGUF files in a repository."""
        mock_files = [
            {"path": "llama-2-7b.Q4_K_M.gguf", "size": 3825000000},
            {"path": "llama-2-7b.Q5_K_S.gguf", "size": 4650000000},
            {"path": "README.md", "size": 5000}
        ]
        
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_files
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
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
                
                # Mock streaming download
                mock_stream_response = AsyncMock()
                mock_stream_response.raise_for_status = Mock()
                mock_stream_response.aiter_bytes = AsyncMock(return_value=[test_content])
                mock_stream.return_value.__aenter__.return_value = mock_stream_response
                
                destination = tmp_path / "test_model.gguf"
                success = await api_client.download_file(
                    repo_id="TheBloke/Test-GGUF",
                    filename="test.gguf",
                    destination=destination
                )
                
                assert success
                assert destination.exists()
                assert destination.read_bytes() == test_content
    
    @pytest.mark.asyncio
    async def test_download_with_progress(self, api_client, tmp_path):
        """Test download with progress callback."""
        test_content = b"0" * 1000
        progress_calls = []
        
        def progress_callback(downloaded, total):
            progress_calls.append((downloaded, total))
        
        with patch('httpx.AsyncClient.head') as mock_head:
            with patch('httpx.AsyncClient.stream') as mock_stream:
                mock_head_response = AsyncMock()
                mock_head_response.headers = {"content-length": "1000"}
                mock_head.return_value = mock_head_response
                
                mock_stream_response = AsyncMock()
                mock_stream_response.raise_for_status = Mock()
                # Simulate chunked download
                mock_stream_response.aiter_bytes = AsyncMock(
                    return_value=[test_content[:500], test_content[500:]]
                )
                mock_stream.return_value.__aenter__.return_value = mock_stream_response
                
                destination = tmp_path / "test_model.gguf"
                success = await api_client.download_file(
                    repo_id="TheBloke/Test-GGUF",
                    filename="test.gguf",
                    destination=destination,
                    progress_callback=progress_callback
                )
                
                assert success
                assert len(progress_calls) == 2
                assert progress_calls[-1] == (1000, 1000)
    
    @pytest.mark.asyncio
    async def test_find_best_gguf_model(self):
        """Test finding best GGUF model utility."""
        mock_models = [
            {
                "modelId": "TheBloke/Llama-2-7B-GGUF",
                "downloads": 100000
            }
        ]
        
        with patch('tldw_Server_API.app.core.LLM_Calls.huggingface_api.HuggingFaceAPI.search_gguf_models') as mock_search:
            async def mock_search_async(*args, **kwargs):
                return mock_models
            mock_search.side_effect = mock_search_async
            
            best_model = await find_best_gguf_model(
                model_name="llama-2",
                max_size_gb=10.0,
                preferred_quant="Q4_K_M"
            )
            
            assert best_model["modelId"] == "TheBloke/Llama-2-7B-GGUF"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, api_client):
        """Test error handling in API calls."""
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_get.side_effect = httpx.HTTPError("Connection error")
            
            results = await api_client.search_models(query="test")
            assert results == []  # Should return empty list on error
            
            info = await api_client.get_model_info("test/model")
            assert info is None  # Should return None on error


class TestIntegration:
    """Integration tests for provider interactions."""
    
    @pytest.mark.asyncio
    @patch('requests.post')
    async def test_provider_switching(self, mock_post):
        """Test switching between different providers."""
        # Mock responses for different providers
        moonshot_response = {
            "choices": [{"message": {"content": "Moonshot response"}}]
        }
        zai_response = {
            "choices": [{"message": {"content": "Z.AI response"}}]
        }
        
        mock_response_obj = Mock()
        mock_response_obj.status_code = 200
        mock_response_obj.raise_for_status = Mock()
        mock_post.return_value = mock_response_obj
        
        # Test Moonshot
        mock_response_obj.json.return_value = moonshot_response
        result1 = chat_with_moonshot(
            input_data=[{"role": "user", "content": "Test"}],
            api_key="key1"
        )
        assert result1 == "Moonshot response"
        
        # Test Z.AI
        mock_response_obj.json.return_value = zai_response
        result2 = chat_with_zai(
            input_data=[{"role": "user", "content": "Test"}],
            api_key="key2"
        )
        assert result2 == "Z.AI response"
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test concurrent requests to multiple providers."""
        with patch('requests.post') as mock_post:
            mock_response_obj = Mock()
            mock_response_obj.status_code = 200
            mock_response_obj.json.return_value = {
                "choices": [{"message": {"content": "Response"}}]
            }
            mock_response_obj.raise_for_status = Mock()
            mock_post.return_value = mock_response_obj
            
            # Simulate concurrent requests
            tasks = [
                asyncio.create_task(asyncio.to_thread(
                    chat_with_moonshot,
                    [{"role": "user", "content": f"Test {i}"}],
                    "key"
                )) for i in range(5)
            ]
            
            results = await asyncio.gather(*tasks)
            assert len(results) == 5
            assert all(r == "Response" for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])