"""
Unit tests for chat request/response schemas and message validation.

Tests focus on the OpenAI-compatible schema validation, message formats,
and request parameter validation without any external dependencies.
"""

import pytest
from typing import Dict, Any, List
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionRequest,
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ResponseFormat,
    ToolDefinition,
    FunctionDefinition,
)

# ========================================================================
# Message Validation Tests
# ========================================================================

class TestMessageValidation:
    """Test message parameter validation."""
    
    @pytest.mark.unit
    def test_valid_user_message(self):
        """Test valid user message creation."""
        msg = ChatCompletionUserMessageParam(
            role="user",
            content="Hello, how are you?"
        )
        assert msg.role == "user"
        assert msg.content == "Hello, how are you?"
    
    @pytest.mark.unit
    def test_valid_system_message(self):
        """Test valid system message creation."""
        msg = ChatCompletionSystemMessageParam(
            role="system",
            content="You are a helpful assistant."
        )
        assert msg.role == "system"
        assert msg.content == "You are a helpful assistant."
    
    @pytest.mark.unit
    def test_valid_assistant_message(self):
        """Test valid assistant message creation."""
        msg = ChatCompletionAssistantMessageParam(
            role="assistant",
            content="I'm doing well, thank you!"
        )
        assert msg.role == "assistant"
        assert msg.content == "I'm doing well, thank you!"
    
    @pytest.mark.unit
    def test_message_with_name(self):
        """Test message with optional name field."""
        msg = ChatCompletionUserMessageParam(
            role="user",
            content="Test message",
            name="user_123"
        )
        assert msg.name == "user_123"
    
    @pytest.mark.unit
    def test_multimodal_user_message(self):
        """Test user message with multimodal content (text + image)."""
        # User messages can have List content for multimodal
        msg = ChatCompletionUserMessageParam(
            role="user",
            content=[
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
            ]
        )
        assert msg.role == "user"
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

# ========================================================================
# ChatCompletionRequest Validation Tests  
# ========================================================================

class TestChatCompletionRequest:
    """Test ChatCompletionRequest model validation."""
    
    @pytest.mark.unit
    def test_minimal_valid_request(self):
        """Test minimal valid chat completion request."""
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Hello"}
            ]
        )
        assert request.model == "gpt-3.5-turbo"
        assert len(request.messages) == 1
        assert request.messages[0].role == "user"
    
    @pytest.mark.unit
    def test_request_with_all_parameters(self):
        """Test request with all optional parameters."""
        request = ChatCompletionRequest(
            api_provider="openai",
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"}
            ],
            temperature=0.7,
            max_tokens=150,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            stop=["\n", "END"],
            stream=False,
            n=1,
            user="user_123"
        )
        assert request.api_provider == "openai"
        assert request.temperature == 0.7
        assert request.max_tokens == 150
        assert request.stop == ["\n", "END"]
    
    @pytest.mark.unit
    def test_request_without_messages_fails(self):
        """Test that request without messages fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(model="gpt-3.5-turbo")
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("messages",) for e in errors)
    
    @pytest.mark.unit
    def test_request_with_empty_messages_fails(self):
        """Test that request with empty messages list fails."""
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(
                model="gpt-3.5-turbo",
                messages=[]
            )
        
        errors = exc_info.value.errors()
        assert any("at least 1 item" in str(e) for e in errors)
    
    @pytest.mark.unit
    def test_temperature_bounds(self):
        """Test temperature parameter bounds (0.0 to 2.0)."""
        # Valid temperatures
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            temperature=0.0
        )
        assert request.temperature == 0.0
        
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            temperature=2.0
        )
        assert request.temperature == 2.0
        
        # Invalid temperature (too high)
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                temperature=2.1
            )
        
        # Invalid temperature (negative)
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                temperature=-0.1
            )
    
    @pytest.mark.unit 
    def test_provider_validation(self):
        """Test API provider validation."""
        valid_providers = ["openai", "anthropic", "cohere", "groq", "mistral"]
        
        for provider in valid_providers:
            request = ChatCompletionRequest(
                api_provider=provider,
                model="test-model",
                messages=[{"role": "user", "content": "test"}]
            )
            assert request.api_provider == provider
    
    @pytest.mark.unit
    def test_streaming_parameter(self):
        """Test streaming parameter."""
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            stream=True
        )
        assert request.stream is True
        
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            stream=False
        )
        assert request.stream is False

# ========================================================================
# Response Format Tests
# ========================================================================

class TestResponseFormat:
    """Test response format specifications."""
    
    @pytest.mark.unit
    def test_json_response_format(self):
        """Test JSON response format specification."""
        format_spec = ResponseFormat(type="json_object")
        assert format_spec.type == "json_object"
        
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Return JSON"}],
            response_format={"type": "json_object"}
        )
        assert request.response_format.type == "json_object"
    
    @pytest.mark.unit
    def test_text_response_format(self):
        """Test text response format (default)."""
        format_spec = ResponseFormat(type="text")
        assert format_spec.type == "text"

# ========================================================================
# Tool/Function Calling Tests
# ========================================================================

class TestToolsAndFunctions:
    """Test tool and function definitions."""
    
    @pytest.mark.unit
    def test_function_definition(self):
        """Test function definition for tool calling."""
        func = FunctionDefinition(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        )
        assert func.name == "get_weather"
        assert "location" in func.parameters["required"]
    
    @pytest.mark.unit
    def test_tool_definition(self):
        """Test tool definition with function."""
        tool = ToolDefinition(
            type="function",
            function={
                "name": "search",
                "description": "Search for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        )
        assert tool.type == "function"
        assert tool.function.name == "search"
    
    @pytest.mark.unit
    def test_request_with_tools(self):
        """Test request with tool definitions."""
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"}
                            }
                        }
                    }
                }
            ],
            tool_choice="auto"
        )
        assert len(request.tools) == 1
        assert request.tool_choice == "auto"

# ========================================================================
# Edge Cases and Error Handling
# ========================================================================

class TestEdgeCasesAndErrors:
    """Test edge cases and error conditions."""
    
    @pytest.mark.unit
    def test_invalid_role_fails(self):
        """Test that invalid message role fails validation."""
        with pytest.raises(ValidationError):
            ChatCompletionUserMessageParam(
                role="invalid_role",  # Should be "user"
                content="test"
            )
    
    @pytest.mark.unit
    def test_missing_content_fails(self):
        """Test that message without content fails."""
        with pytest.raises(ValidationError):
            ChatCompletionUserMessageParam(role="user")
    
    @pytest.mark.unit
    def test_max_tokens_bounds(self):
        """Test max_tokens parameter bounds."""
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        assert request.max_tokens == 1
        
        # Negative max_tokens should fail
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=-1
            )
    
    @pytest.mark.unit
    def test_n_parameter_bounds(self):
        """Test n parameter bounds (1 to 128)."""
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            n=1
        )
        assert request.n == 1
        
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            n=128
        )
        assert request.n == 128
        
        # Too many completions
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                n=129
            )
    
    @pytest.mark.unit
    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed (for provider-specific params)."""
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            custom_param="custom_value",  # Extra field
            another_param=123  # Another extra field
        )
        # Should not raise an error due to ConfigDict(extra="allow")