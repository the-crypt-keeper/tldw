# tests/test_llm_api_calls.py
import pytest
from unittest.mock import patch, MagicMock
import os
import sys

# Add the tldw directory (one level up from Tests) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'tldw')))
from App_Function_Libraries.LLM_API_Calls import chat_with_openai, chat_with_anthropic


# Print the sys.path to debug
print("Current sys.path:", sys.path)

@pytest.fixture
def mock_openai_response():
    return {
        "choices": [
            {
                "message": {
                    "content": "This is a test response from OpenAI."
                }
            }
        ]
    }

@pytest.fixture
def mock_anthropic_response():
    return {
        "content": [
            {
                "text": "This is a test response from Anthropic."
            }
        ]
    }

@patch('App_Function_Libraries.LLM_API_Calls.requests.post')
def test_chat_with_openai(mock_post, mock_openai_response):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = mock_openai_response

    response = chat_with_openai("test_api_key", "Test input", "Test prompt")
    assert response == "This is a test response from OpenAI."

@patch('App_Function_Libraries.LLM_API_Calls.requests.post')
def test_chat_with_anthropic(mock_post, mock_anthropic_response):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = mock_anthropic_response

    response = chat_with_anthropic("test_api_key", "Test input", "Test prompt")
    assert response == "This is a test response from Anthropic."