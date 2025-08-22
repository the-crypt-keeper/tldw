#!/usr/bin/env python3
"""
Basic usage example for the Mock OpenAI API Server.

This example demonstrates how to use the mock server with the OpenAI Python client.
"""

import openai
import json
from typing import Dict, Any


def test_chat_completion():
    """Test chat completion endpoint."""
    print("Testing Chat Completion...")
    
    # Configure OpenAI client to use mock server
    openai.api_base = "http://localhost:8080/v1"
    openai.api_key = "sk-mock-key-12345"
    
    # Create a chat completion
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, this is a test message."}
        ],
        temperature=0.7,
        max_tokens=150
    )
    
    print("Response:")
    print(json.dumps(response, indent=2))
    print()
    
    # Extract the assistant's message
    assistant_message = response['choices'][0]['message']['content']
    print(f"Assistant: {assistant_message}")
    print("-" * 50)


def test_streaming_chat():
    """Test streaming chat completion."""
    print("\nTesting Streaming Chat Completion...")
    
    openai.api_base = "http://localhost:8080/v1"
    openai.api_key = "sk-mock-key-12345"
    
    # Create a streaming chat completion
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "Tell me a story about a robot."}
        ],
        stream=True
    )
    
    print("Streaming response:")
    full_response = ""
    for chunk in response:
        if 'choices' in chunk:
            choice = chunk['choices'][0]
            if 'delta' in choice:
                delta = choice['delta']
                if 'content' in delta:
                    content = delta['content']
                    print(content, end='', flush=True)
                    full_response += content
    
    print("\n" + "-" * 50)


def test_embeddings():
    """Test embeddings endpoint."""
    print("\nTesting Embeddings...")
    
    openai.api_base = "http://localhost:8080/v1"
    openai.api_key = "sk-mock-key-12345"
    
    # Create embeddings
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input="This is a test text for embedding."
    )
    
    print("Embeddings response:")
    print(f"Model: {response['model']}")
    print(f"Usage: {response['usage']}")
    print(f"Number of embeddings: {len(response['data'])}")
    
    if response['data']:
        embedding = response['data'][0]['embedding']
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
    
    print("-" * 50)


def test_models_list():
    """Test models list endpoint."""
    print("\nTesting Models List...")
    
    import requests
    
    headers = {
        "Authorization": "Bearer sk-mock-key-12345"
    }
    
    response = requests.get("http://localhost:8080/v1/models", headers=headers)
    
    if response.status_code == 200:
        models = response.json()
        print("Available models:")
        for model in models['data']:
            print(f"  - {model['id']} (owned by: {model['owned_by']})")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
    
    print("-" * 50)


def test_error_handling():
    """Test error handling with invalid API key."""
    print("\nTesting Error Handling...")
    
    import requests
    
    # Test with invalid API key
    headers = {
        "Authorization": "Bearer invalid-key"
    }
    
    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}]
    }
    
    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code != 200:
        print("Error response:")
        print(json.dumps(response.json(), indent=2))
    
    print("-" * 50)


def main():
    """Run all tests."""
    print("=" * 50)
    print("Mock OpenAI API Server - Basic Usage Examples")
    print("=" * 50)
    print("\nMake sure the mock server is running:")
    print("  python -m mock_openai.server")
    print("=" * 50)
    
    try:
        test_chat_completion()
        test_streaming_chat()
        test_embeddings()
        test_models_list()
        test_error_handling()
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure the mock server is running on http://localhost:8080")


if __name__ == "__main__":
    main()